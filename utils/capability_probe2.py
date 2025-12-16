import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import copy

class CapabilityGapProbe:
    """优化的能力监控探针 - 仅关注logits差距"""
    
    def __init__(self, alpha=0.1, cooldown_epochs=1, ema_decay=0.9):
        """
        alpha: 自适应调整的强度系数，控制权重变化幅度 (降低到0.1)
        cooldown_epochs: 调整冷却期，避免连续大幅调整
        ema_decay: 指数移动平均的衰减系数
        """
        self.alpha = alpha
        self.cooldown_epochs = cooldown_epochs
        self.ema_decay = ema_decay
        
        # 历史记录
        self.gap_history = []
        self.performance_history = []
        self.weight_adjustments = {}
        
        # 冷却期控制
        self.last_adjustment_epoch = -self.cooldown_epochs
        
        # 指数移动平均权重
        self.ema_weights = None
        
        # 权重变化限制
        self.max_weight_change = 0.10  # ±10%限制
    
    def compute_logits_gap_normalized(self, teacher_logits, student_logits):
        """计算归一化的输出层差距 [0,1]区间"""
        try:
            # 确保输入是tensor
            if not isinstance(teacher_logits, torch.Tensor):
                teacher_logits = torch.tensor(teacher_logits)
            if not isinstance(student_logits, torch.Tensor):
                student_logits = torch.tensor(student_logits)
            
            # 方法1: 使用KL散度并归一化
            teacher_probs = F.softmax(teacher_logits, dim=1)
            student_probs = F.softmax(student_logits, dim=1)
            
            # 双向KL散度 (JS散度的近似)
            kl_div_ts = F.kl_div(
                F.log_softmax(student_logits, dim=1),
                teacher_probs,
                reduction='batchmean'
            )
            kl_div_st = F.kl_div(
                F.log_softmax(teacher_logits, dim=1),
                student_probs,
                reduction='batchmean'
            )
            
            # JS散度近似
            js_div = (kl_div_ts + kl_div_st) / 2.0
            
            # 归一化到[0,1]: JS散度的最大值是log(2)
            # 确保js_div是tensor并正确处理
            if isinstance(js_div, torch.Tensor):
                js_div_value = js_div.item()
            else:
                js_div_value = float(js_div)
            
            normalized_gap = min(js_div_value / np.log(2), 1.0)
            
            return max(0.0, normalized_gap)  # 确保非负
            
        except Exception as e:
            print(f"Warning: Error in logits gap computation: {e}")
            return 0.0  # 返回默认值
    
    def compute_confidence_gap_normalized(self, teacher_probs, student_probs, true_labels):
        """计算归一化的置信度校准差距 [0,1]区间"""
        try:
            # 确保输入是tensor
            if not isinstance(teacher_probs, torch.Tensor):
                teacher_probs = torch.tensor(teacher_probs)
            if not isinstance(student_probs, torch.Tensor):
                student_probs = torch.tensor(student_probs)
            if not isinstance(true_labels, torch.Tensor):
                true_labels = torch.tensor(true_labels)
            
            # 计算预测置信度
            teacher_confidence = torch.max(teacher_probs, dim=1)[0].mean()
            student_confidence = torch.max(student_probs, dim=1)[0].mean()
            
            # 计算准确率
            teacher_pred = teacher_probs.argmax(dim=1)
            student_pred = student_probs.argmax(dim=1)
            
            teacher_acc = (teacher_pred == true_labels).float().mean()
            student_acc = (student_pred == true_labels).float().mean()
            
            # 置信度-准确率差距 (校准误差)
            teacher_calibration_error = abs(teacher_confidence - teacher_acc)
            student_calibration_error = abs(student_confidence - student_acc)
            
            # 两者校准误差的差距，已经在[0,1]区间内
            confidence_gap = abs(teacher_calibration_error - student_calibration_error)
            
            # 安全地提取数值
            if isinstance(confidence_gap, torch.Tensor):
                return confidence_gap.item()
            else:
                return float(confidence_gap)
                
        except Exception as e:
            print(f"Warning: Error in confidence gap computation: {e}")
            return 0.0  # 返回默认值
    
    def measure_capability_gaps(self, teacher_outputs, student_outputs, true_labels):
        """测量关键能力维度的差距 - 仅logits和置信度"""
        gaps = {}
        
        # 1. 输出层差距 (主要指标)
        gaps['logits_gap'] = self.compute_logits_gap_normalized(
            teacher_outputs['logits'], 
            student_outputs['logits']
        )
        
        # 2. 置信度校准差距 (辅助指标)
        teacher_probs = F.softmax(teacher_outputs['logits'], dim=1)
        student_probs = F.softmax(student_outputs['logits'], dim=1)
        gaps['confidence_gap'] = self.compute_confidence_gap_normalized(
            teacher_probs, student_probs, true_labels
        )
        
        return gaps
    
    def analyze_performance_change(self, current_acc):
        """分析性能变化趋势 - 更保守的判断"""
        # 确保current_acc是数值
        if isinstance(current_acc, torch.Tensor):
            current_acc = current_acc.item()
        
        if len(self.performance_history) == 0:
            self.performance_history.append(current_acc)
            return 'initial'
        
        # 使用最近3个epoch的平均值来判断趋势（如果有足够历史）
        if len(self.performance_history) >= 3:
            recent_avg = np.mean(self.performance_history[-3:])
            previous_avg = np.mean(self.performance_history[-3:-1]) if len(self.performance_history) > 3 else self.performance_history[-1]
        else:
            recent_avg = current_acc
            previous_avg = self.performance_history[-1]
        
        self.performance_history.append(current_acc)
        
        # 保守的变化阈值
        acc_change = recent_avg - previous_avg
        
        if acc_change > 0.1:  # 提升超过1%
            return 'improving'
        elif acc_change < -0.1:  # 下降超过1%
            return 'declining'
        else:
            return 'stable'
    
    def compute_gap_change_ratios(self, current_gaps):
        """计算差距变化比例"""
        if len(self.gap_history) == 0:
            self.gap_history.append(current_gaps)
            return {key: 0.0 for key in current_gaps.keys()}
        
        previous_gaps = self.gap_history[-1]
        self.gap_history.append(current_gaps)
        
        gap_change_ratios = {}
        for key in current_gaps.keys():
            # 确保数值类型
            curr_val = float(current_gaps[key])
            prev_val = float(previous_gaps.get(key, 0))
            
            if prev_val > 1e-8:  # 避免除零
                ratio = (curr_val - prev_val) / prev_val
                gap_change_ratios[key] = ratio
            else:
                gap_change_ratios[key] = 0.0
        
        return gap_change_ratios
    
    def compute_adaptive_weights(self, base_weights, gap_change_ratios, performance_trend, current_epoch):
        """计算自适应权重 - 改进的策略"""
        
        # 检查冷却期
        if current_epoch - self.last_adjustment_epoch < self.cooldown_epochs:
            # 在冷却期内，返回当前EMA权重或基础权重
            if self.ema_weights is not None:
                return self.ema_weights.copy()
            else:
                return base_weights.copy()
        
        # 初始化自适应权重
        adaptive_weights = base_weights.copy()
        
        # 主要基于logits差距调整alpha
        logits_gap_ratio = gap_change_ratios.get('logits_gap', 0.0)
        confidence_gap_ratio = gap_change_ratios.get('confidence_gap', 0.0)
        
        # alpha调整逻辑 (蒸馏权重)
        if 'alpha' in base_weights:
            base_alpha = base_weights['alpha']
            alpha_adjustment = 0.0
            
            if performance_trend == 'declining':
                # 性能下降时的策略
                if logits_gap_ratio > 0.05:  # logits差距显著增大
                    # 增加蒸馏权重，让学生更多学习教师
                    alpha_adjustment = self.alpha * 0.5
                elif logits_gap_ratio < -0.05:  # logits差距减小但性能仍下降
                    # 可能过度蒸馏，略微减少蒸馏权重
                    alpha_adjustment = -self.alpha * 0.2
                    
            elif performance_trend == 'improving':
                # 性能提升时的策略
                if logits_gap_ratio < -0.05:  # logits差距显著减小
                    # 差距在缩小且性能提升，可以适当增加蒸馏权重加速收敛
                    alpha_adjustment = self.alpha * 0.3
                elif logits_gap_ratio > 0.05:  # 差距增大但性能提升
                    # 可能是正常的学习过程，小幅调整
                    alpha_adjustment = -self.alpha * 0.1
                    
            else:  # stable
                # 性能稳定时，进行微调
                if abs(logits_gap_ratio) > 0.1:  # 只对显著变化做调整
                    alpha_adjustment = -self.alpha * 0.1 * np.sign(logits_gap_ratio)
            
            # 应用调整限制
            alpha_adjustment = np.clip(alpha_adjustment, 
                                     -base_alpha * self.max_weight_change,
                                     base_alpha * self.max_weight_change)
            
            adaptive_weights['alpha'] = np.clip(base_alpha + alpha_adjustment, 0.1, 0.95)
        
        # dis_weight调整逻辑 (蒸馏损失整体权重)
        if 'dis_weight' in base_weights:
            base_dis_weight = base_weights['dis_weight']
            dis_weight_adjustment = 0.0
            
            # 主要基于整体性能趋势调整
            if performance_trend == 'declining':
                # 性能下降时，增加蒸馏损失权重
                dis_weight_adjustment = self.alpha * 0.3
            elif performance_trend == 'improving':
                # 性能提升时，可以适当调整蒸馏权重
                if logits_gap_ratio < 0:  # 差距在缩小
                    dis_weight_adjustment = self.alpha * 0.2
                else:
                    dis_weight_adjustment = -self.alpha * 0.1
            
            # 应用调整限制
            dis_weight_adjustment = np.clip(dis_weight_adjustment,
                                          -base_dis_weight * self.max_weight_change,
                                          base_dis_weight * self.max_weight_change)
            
            adaptive_weights['dis_weight'] = np.clip(base_dis_weight + dis_weight_adjustment, 0.1, 3.0)
        
        # 应用指数移动平均平滑
        if self.ema_weights is None:
            self.ema_weights = adaptive_weights.copy()
        else:
            for key in adaptive_weights.keys():
                if key in self.ema_weights:
                    self.ema_weights[key] = (self.ema_decay * self.ema_weights[key] + 
                                           (1 - self.ema_decay) * adaptive_weights[key])
        
        # 记录调整时间
        self.last_adjustment_epoch = current_epoch
        
        return self.ema_weights.copy()
    
    def update_and_get_adaptive_weights(self, teacher_outputs, student_outputs, 
                                      true_labels, current_acc, base_weights, current_epoch):
        """主要接口：更新探针状态并返回自适应权重"""
        
        try:
            # 1. 测量当前能力差距
            current_gaps = self.measure_capability_gaps(
                teacher_outputs, student_outputs, true_labels
            )
            
            # 2. 分析性能变化
            performance_trend = self.analyze_performance_change(current_acc)
            
            # 3. 计算差距变化比例
            gap_change_ratios = self.compute_gap_change_ratios(current_gaps)
            
            # 4. 计算自适应权重
            adaptive_weights = self.compute_adaptive_weights(
                base_weights, gap_change_ratios, performance_trend, current_epoch
            )
            
            # 5. 记录调整信息
            self.weight_adjustments = {
                'epoch_gaps': current_gaps,
                'gap_ratios': gap_change_ratios,
                'performance_trend': performance_trend,
                'weight_changes': {
                    k: adaptive_weights.get(k, 0) - base_weights.get(k, 0) 
                    for k in base_weights.keys()
                },
                'in_cooldown': current_epoch - self.last_adjustment_epoch < self.cooldown_epochs
            }
            
            return adaptive_weights, current_gaps, performance_trend
            
        except Exception as e:
            print(f"Error in capability probe: {e}")
            print("Using default weights...")
            return base_weights.copy(), {'logits_gap': 0.0, 'confidence_gap': 0.0}, 'stable'
    
    def get_adjustment_summary(self):
        """获取调整摘要信息"""
        return self.weight_adjustments
    
    def reset_history(self, keep_recent=5):
        """重置历史记录，只保留最近几次"""
        if len(self.gap_history) > keep_recent:
            self.gap_history = self.gap_history[-keep_recent:]
        if len(self.performance_history) > keep_recent:
            self.performance_history = self.performance_history[-keep_recent:]