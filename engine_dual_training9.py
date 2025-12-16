import math
import sys
import gc
import torch
import torch.nn.functional as F
import numpy as np
from typing import Iterable
from timm.utils import accuracy 
from tqdm import tqdm
from utils import utils
from utils.center import get_center, get_weights, refine_pseudoDA, get_center_v2
from utils.multi_model3 import simple_logits_distillation_loss
from utils.capability_probe2 import CapabilityGapProbe
from scipy.stats.stats import kendalltau

gc.collect()
torch.cuda.empty_cache()

def _get_gt_mask(logits, target):
    """获取真实标签的掩码"""
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def compute_kendall_correlation_coeffs(teacher_logits, student_logits):
    """
    计算教师和学生模型输出的肯德尔相关系数（不做筛选）
    Args:
        teacher_logits: 教师模型的logits输出 [batch_size, num_classes]
        student_logits: 学生模型的logits输出 [batch_size, num_classes] 
    Returns:
        kendall_coeffs: 每个样本的肯德尔相关系数 [batch_size]
    """
    batch_size = teacher_logits.size(0)
    kendall_coeffs = torch.zeros(batch_size, device=teacher_logits.device)
    
    # 计算每个样本的肯德尔相关系数
    for i in range(batch_size):
        teacher_sample = teacher_logits[i].cpu().detach().numpy()
        student_sample = student_logits[i].cpu().detach().numpy()
        
        try:
            kendall_tau, _ = kendalltau(teacher_sample, student_sample)
            if np.isnan(kendall_tau):
                kendall_tau = 0.0
            kendall_coeffs[i] = kendall_tau
        except:
            kendall_coeffs[i] = 0.0
    
    return kendall_coeffs


def compute_pseudolabel_consistency(teacher_pseudo_labels, student_pseudo_labels):
    """
    基于伪标签计算一致性分数
    Args:
        teacher_pseudo_labels: 教师模型的伪标签 [batch_size]
        student_pseudo_labels: 学生模型的伪标签 [batch_size]
    Returns:
        consistency: 每个样本的标签一致性分数 [batch_size]
    """
    batch_size = teacher_pseudo_labels.size(0)
    
    # 计算师生伪标签一致性作为质量指标
    label_consistency = (teacher_pseudo_labels == student_pseudo_labels).float()
    
    return label_consistency


def dual_adaptive_distillation_vectorized(student_logits, teacher_logits, pseudo_labels,
                                         kendall_coeffs, base_temp=4.0, temp_range=(2.0, 8.0),
                                         alpha_range=(0.3, 0.9), weights=None):
    """
    双重自适应蒸馏：根据肯德尔系数同时调节温度和软硬标签比例
    
    Args:
        student_logits: 学生logits [B, C]
        teacher_logits: 教师logits [B, C]
        pseudo_labels: 伪标签 [B]
        kendall_coeffs: 肯德尔相关系数 [B]
        base_temp: 基础温度
        temp_range: 温度范围 (min, max)
        alpha_range: alpha范围 (min, max)
        weights: 样本权重 [B] (可选)
    
    Returns:
        total_loss: 总损失
        stats: 统计信息字典
    """
    # 1. 归一化肯德尔系数到[0, 1]
    kendall_norm = (kendall_coeffs + 1.0) / 2.0
    
    # 2. 计算自适应参数
    min_temp, max_temp = temp_range
    min_alpha, max_alpha = alpha_range
    
    temps = min_temp + (max_temp - min_temp) * kendall_norm  # [B]
    alphas = min_alpha + (max_alpha - min_alpha) * kendall_norm  # [B]
    
    # 3. 扩展维度用于广播
    temps_expanded = temps.unsqueeze(1)  # [B, 1]
    
    # 4. 计算软标签损失（每个样本不同温度）
    soft_teacher = F.softmax(teacher_logits / temps_expanded, dim=1)
    soft_student = F.log_softmax(student_logits / temps_expanded, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='none').sum(1) * (temps ** 2)
    
    # 5. 计算硬标签损失
    hard_loss = F.cross_entropy(student_logits, pseudo_labels, reduction='none')
    
    # 6. 自适应加权组合
    combined_loss = alphas * soft_loss + (1 - alphas) * hard_loss
    
    # 7. 应用样本权重（如果提供）
    if weights is not None:
        weighted_loss = (weights * combined_loss).sum() / weights.sum()
    else:
        weighted_loss = combined_loss.mean()
    
    # 8. 收集统计信息
    stats = {
        'mean_temp': temps.mean().item(),
        'mean_alpha': alphas.mean().item(),
        'temp_std': temps.std().item(),
        'alpha_std': alphas.std().item(),
        'mean_kendall': kendall_coeffs.mean().item(),
        'min_temp': temps.min().item(),
        'max_temp': temps.max().item(),
        'min_alpha': alphas.min().item(),
        'max_alpha': alphas.max().item(),
    }
    
    return weighted_loss, stats


def adaptive_dkd_loss(logits_student, logits_teacher, target, kendall_coeffs, 
                     base_temp=4.0, temp_range=(2.0, 8.0),
                     tkl_weight=1.0, ntkl_weight=8.0, weights=None):
    """
    自适应DKD损失：使用肯德尔系数调节温度
    
    Args:
        logits_student: 学生模型logits
        logits_teacher: 教师模型logits  
        target: 真实标签/伪标签
        kendall_coeffs: 肯德尔相关系数 [B]
        base_temp: 基础温度
        temp_range: 温度范围
        tkl_weight: 目标知识蒸馏权重
        ntkl_weight: 非目标知识蒸馏权重
        weights: 样本权重
    """
    # 1. 归一化肯德尔系数
    kendall_norm = (kendall_coeffs + 1.0) / 2.0
    
    # 2. 计算自适应温度
    min_temp, max_temp = temp_range
    temps = min_temp + (max_temp - min_temp) * kendall_norm  # [B]
    temps_expanded = temps.unsqueeze(1)  # [B, 1]
    
    # 3. 获取目标掩码
    gt_mask = _get_gt_mask(logits_student, target)
    
    # 4. 目标知识蒸馏 (TKL)
    logits_teacher_target = logits_teacher * gt_mask
    logits_student_target = logits_student * gt_mask
    
    pred_teacher_target = F.softmax(logits_teacher_target / temps_expanded, dim=1)
    log_pred_student_target = F.log_softmax(logits_student_target / temps_expanded, dim=1)
    
    tkl_loss = F.kl_div(log_pred_student_target, pred_teacher_target.detach(), 
                        reduction='none').sum(1) * (temps ** 2)
    
    # 5. 非目标知识蒸馏 (NTKL)
    logits_teacher_nontarget = logits_teacher * (~gt_mask)
    logits_student_nontarget = logits_student * (~gt_mask)
    
    pred_teacher_nontarget = F.softmax(logits_teacher_nontarget / temps_expanded, dim=1)
    log_pred_student_nontarget = F.log_softmax(logits_student_nontarget / temps_expanded, dim=1)
    
    ntkl_loss = F.kl_div(log_pred_student_nontarget, pred_teacher_nontarget.detach(), 
                         reduction='none').sum(1) * (temps ** 2)
    
    # 6. 组合损失
    combined_loss = tkl_weight * tkl_loss + ntkl_weight * ntkl_loss
    
    # 7. 应用样本权重
    if weights is not None:
        total_loss = (weights * combined_loss).sum() / weights.sum()
    else:
        total_loss = combined_loss.mean()
    
    # 8. 统计信息
    stats = {
        'mean_temp': temps.mean().item(),
        'temp_std': temps.std().item(),
        'mean_kendall': kendall_coeffs.mean().item(),
    }
    
    return total_loss, stats


def train_one_epoch_dual(args, dual_model: torch.nn.Module, 
                        data_loader_u: Iterable, 
                        teacher_optimizer: torch.optim.Optimizer,
                        student_optimizer: torch.optim.Optimizer,
                        amp_autocast, device: torch.device, epoch: int, 
                        loss_scaler, 
                        lr_schedule_values, 
                        train_config,
                        start_steps=None,
                        teacher_memory=None,
                        student_memory=None,
                        prob_list=None,
                        capability_probe=None
                        ):
    
    # 决定当前epoch的训练模式
    current_mode = dual_model.should_switch_mode(epoch)
    dual_model.set_training_mode(current_mode, epoch)
    
    # 获取当前活跃的模型和优化器
    active_model = dual_model.get_active_model()
    active_optimizer = teacher_optimizer if current_mode == 'T' else student_optimizer
    active_memory = teacher_memory if current_mode == 'T' else student_memory
    
    active_model.train()
    
    # 对于学生训练模式，确保教师模型处于评估模式（用于蒸馏）
    if current_mode in ['K1', 'K2']:
        dual_model.teacher.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    mode_descriptions = {
        'T': 'TEACHER',
        'S': 'STUDENT (DPA Only)',
        'K1': 'STUDENT (Adaptive-KD Only)', 
        'K2': 'STUDENT (Adaptive-KD + DPA)'
    }
    header = f'Epoch: [{epoch}] - Training {mode_descriptions.get(current_mode, current_mode)}'
    print_freq = 10

    # 损失累积变量
    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    reg_loss_sum = 0.0
    align_loss_sum = 0.0
    kd_loss_sum = 0.0
    batch_count = 0
    
    # 多重蒸馏损失统计
    kd_weak_loss_sum = 0.0
    kd_strong_loss_sum = 0.0 
    kd_prototype_loss_sum = 0.0
    kd_pseudolabel_loss_sum = 0.0
    kd_dkd_loss_sum = 0.0

    # 自适应参数统计
    adaptive_stats = {
        1: {'mean_temp': 0.0, 'mean_alpha': 0.0, 'mean_kendall': 0.0, 'batch_count': 0},  # 弱增强
        2: {'mean_temp': 0.0, 'mean_alpha': 0.0, 'mean_kendall': 0.0, 'batch_count': 0},  # 强增强
        3: {'mean_temp': 0.0, 'mean_alpha': 0.0, 'mean_kendall': 0.0, 'batch_count': 0},  # 原型
        4: {'mean_temp': 0.0, 'mean_alpha': 0.0, 'mean_kendall': 0.0, 'batch_count': 0},  # 伪标签
        5: {'mean_temp': 0.0, 'mean_kendall': 0.0, 'batch_count': 0},  # DKD (只有温度)
    }



    # 从train_config获取配置参数
    choose_logits = train_config.get('choose_logits', [1,2,5])
    logits_weights = train_config.get('logits_weights', [1.0, 1.0, 1.0, 1.0, 1.0])
    tkl_weight = train_config.get('tkl_weight', 1.0)
    ntkl_weight = train_config.get('ntkl_weight', 8.0)
    
    # 自适应蒸馏配置
    use_adaptive_distillation = train_config.get('use_adaptive_distillation', False)
    temp_range = train_config.get('temp_range', (2.0, 8.0))
    alpha_range = train_config.get('alpha_range', (0.3, 0.9))
    
    # 打印配置信息
    if current_mode in ['K1', 'K2']:
        print(f"使用的蒸馏logits类型: {choose_logits}")
        print("类型说明: 1-弱增强×文本原型, 2-强增强×文本原型, 3-视觉原型×文本原型, 4-伪标签蒸馏, 5-DKD损失")
        if use_adaptive_distillation:
            print(f"自适应蒸馏已启用 - 温度范围: {temp_range}, Alpha范围: {alpha_range}")
        else:
            print(f"使用标准蒸馏 - 固定温度: {train_config['distill_temp']}")

    for data_iter_step, (img, true_labels, idx) in enumerate(metric_logger.log_every(data_loader_u, print_freq, header)):
        img_weak = img[0].to(device, non_blocking=True)
        img_strong = img[1].to(device, non_blocking=True)
        true_labels = true_labels.to(device, non_blocking=True)
        
        # 学习率调度
        it = start_steps + data_iter_step
        if lr_schedule_values is not None:
            for i, param_group in enumerate(active_optimizer.param_groups):
                if lr_schedule_values is not None: 
                    param_group["lr"] = lr_schedule_values[it]

        # 生成伪标签 - 根据训练模式选择生成伪标签的模型
        with torch.no_grad():
            if current_mode == 'T':
                feat_weak = dual_model.teacher(img_weak)
                logit_txt = 100. * feat_weak @ dual_model.teacher.get_classifier().t()
                probs_txt = F.softmax(logit_txt, dim=-1)
                
                prob_list.append(probs_txt.mean(0))
                if len(prob_list) > 32:
                    prob_list.pop(0)
                probs, _, probs_centre = refine_pseudoDA(train_config, probs_txt, prob_list, 
                                                       dual_model.teacher.get_center().to(device), feat_weak)
                pseudo_labels = torch.argmax(probs, -1)
                
                weights_txt = get_weights(probs_txt, pseudo_labels)
                weights_centre = get_weights(probs_centre, pseudo_labels)
                weights = (weights_centre * weights_txt).to(device)
            else:
                feat_weak = dual_model.student(img_weak)
                logit_txt = 100. * feat_weak @ dual_model.student.get_classifier().t()
                probs_txt = F.softmax(logit_txt, dim=-1)
                
                prob_list.append(probs_txt.mean(0))
                if len(prob_list) > 32:
                    prob_list.pop(0)
                probs, _, probs_centre = refine_pseudoDA(train_config, probs_txt, prob_list, 
                                                       dual_model.student.get_center().to(device), feat_weak)
                pseudo_labels = torch.argmax(probs, -1)
                
                weights_txt = get_weights(probs_txt, pseudo_labels)
                weights_centre = get_weights(probs_centre, pseudo_labels)
                weights = (weights_centre * weights_txt).to(device)

        # 初始化各种蒸馏损失
        kd_weak_loss = torch.tensor(0.0, device=device)
        kd_strong_loss = torch.tensor(0.0, device=device)
        kd_prototype_loss = torch.tensor(0.0, device=device)
        kd_pseudolabel_loss = torch.tensor(0.0, device=device)
        kd_dkd_loss = torch.tensor(0.0, device=device)

        with amp_autocast():
            if current_mode == 'T':
                # 教师模型训练 - 标准DPA训练
                feat_strong = active_model(img_strong)
                logits_strong_txt = 100. * feat_strong @ active_model.get_classifier().t()
                
                cls_loss = (weights * F.cross_entropy(logits_strong_txt, pseudo_labels, reduction='none')).mean()
                reg_loss = -train_config['reg'] * (torch.log((F.softmax(logits_strong_txt, dim=-1)).mean(0))).mean()
                
                logits = train_config['w'] * active_model.get_center().to(device) @ active_model.get_classifier().t()
                labels = torch.arange(len(logits)).to(device)
                align_loss = train_config['align'] * F.cross_entropy(logits, labels)
                
                loss = cls_loss + reg_loss + align_loss
                kd_loss = torch.tensor(0.0, device=device)
                
            elif current_mode == 'S':
                # 学生模型训练 - 仅使用原始DPA损失
                student_feat_strong = dual_model.student(img_strong)
                student_logits_strong = 100. * student_feat_strong @ dual_model.student.get_classifier().t()
                
                cls_loss = (weights * F.cross_entropy(student_logits_strong, pseudo_labels, reduction='none')).mean()
                reg_loss = -train_config['reg'] * (torch.log((F.softmax(student_logits_strong, dim=-1)).mean(0))).mean()
                
                logits = train_config['w'] * active_model.get_center().to(device) @ active_model.get_classifier().t()
                labels = torch.arange(len(logits)).to(device)
                align_loss = train_config['align'] * F.cross_entropy(logits, labels)
                
                loss = cls_loss + reg_loss + align_loss
                kd_loss = torch.tensor(0.0, device=device)

            elif current_mode in ['K1', 'K2']:
                # 学生模型训练 - 使用自适应蒸馏
                
                # 获取学生模型的特征和logits
                student_feat_weak = dual_model.student(img_weak)
                student_logits_weak = 100. * student_feat_weak @ dual_model.student.get_classifier().t()

                student_feat_strong = dual_model.student(img_strong)  
                student_logits_strong = 100. * student_feat_strong @ dual_model.student.get_classifier().t()

                # 获取教师模型的logits（只进行推理，不更新梯度）
                with torch.no_grad():
                    teacher_feat_weak = dual_model.teacher(img_weak)
                    teacher_logits_weak = 100. * teacher_feat_weak @ dual_model.teacher.get_classifier().t()
                    
                    teacher_feat_strong = dual_model.teacher(img_strong)
                    teacher_logits_strong = 100. * teacher_feat_strong @ dual_model.teacher.get_classifier().t()
                    
                    # 原型logits
                    teacher_logits_prototype = train_config['w'] * dual_model.teacher.get_center().to(device) @ dual_model.teacher.get_classifier().t()
                    student_logits_prototype = train_config['w'] * dual_model.student.get_center().to(device) @ dual_model.student.get_classifier().t()

                    # 为伪标签蒸馏生成教师模型的伪标签
                    teacher_probs_txt = F.softmax(teacher_logits_weak, dim=-1)
                    teacher_pseudo_labels = torch.argmax(teacher_probs_txt, -1)
                    
                    # 生成学生模型的伪标签
                    student_probs_txt = F.softmax(student_logits_weak, dim=-1)
                    student_pseudo_labels = torch.argmax(student_probs_txt, -1)

                # 为不同的损失类型分别计算肯德尔相关系数
                kendall_coeffs_dict = {}
                
                if use_adaptive_distillation:
                    if 1 in choose_logits:  # 弱增强×文本原型蒸馏
                        kendall_coeffs_dict[1] = compute_kendall_correlation_coeffs(
                            teacher_logits_weak, student_logits_weak
                        )
                    
                    if 2 in choose_logits:  # 强增强×文本原型蒸馏
                        kendall_coeffs_dict[2] = compute_kendall_correlation_coeffs(
                            teacher_logits_strong, student_logits_strong
                        )
                    
                    if 3 in choose_logits:  # 视觉原型×文本原型蒸馏
                        kendall_coeffs_dict[3] = compute_kendall_correlation_coeffs(
                            teacher_logits_prototype, student_logits_prototype
                        )
                    
                    if 4 in choose_logits:  # 伪标签蒸馏
                        # 使用标签一致性作为"肯德尔系数"
                        kendall_coeffs_dict[4] = compute_pseudolabel_consistency(
                            teacher_pseudo_labels, student_pseudo_labels
                        )
                    
                    if 5 in choose_logits:  # DKD损失
                        kendall_coeffs_dict[5] = compute_pseudolabel_consistency(
                            teacher_pseudo_labels, student_pseudo_labels
                        )

                # 获取训练配置中的权重
                logits_weights = train_config.get('logits_weights', [1.0, 1.0, 1.0, 1.0, 1.0])
                tkl_weight = train_config.get('tkl_weight', 1.0)
                ntkl_weight = train_config.get('ntkl_weight', 8.0)

                # 计算不同类型的知识蒸馏损失
# 替换第440-520行

                # 计算不同类型的知识蒸馏损失
                kd_loss = torch.tensor(0.0, device=device)

                if 1 in choose_logits:
                    # 弱增强×文本原型蒸馏
                    if use_adaptive_distillation:
                        kd_weak_loss, weak_stats = dual_adaptive_distillation_vectorized(
                            student_logits_weak, teacher_logits_weak, pseudo_labels,
                            kendall_coeffs_dict[1],
                            base_temp=train_config['distill_temp'],
                            temp_range=temp_range,
                            alpha_range=alpha_range,
                            weights=weights
                        )
                        adaptive_stats[1]['mean_temp'] += weak_stats['mean_temp']
                        adaptive_stats[1]['mean_alpha'] += weak_stats['mean_alpha']
                        adaptive_stats[1]['mean_kendall'] += weak_stats['mean_kendall']
                        adaptive_stats[1]['batch_count'] += 1
                    else:
                        # 标准蒸馏（固定温度和alpha）
                        temp = train_config['distill_temp']
                        alpha_val = 0.7
                        
                        soft_teacher = F.softmax(teacher_logits_weak / temp, dim=1)
                        soft_student = F.log_softmax(student_logits_weak / temp, dim=1)
                        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='none').sum(1) * (temp ** 2)
                        
                        hard_loss = F.cross_entropy(student_logits_weak, pseudo_labels, reduction='none')
                        combined_loss = alpha_val * soft_loss + (1 - alpha_val) * hard_loss
                        
                        if weights is not None:
                            kd_weak_loss = (weights * combined_loss).mean()
                        else:
                            kd_weak_loss = combined_loss.mean()
                    
                    kd_loss += logits_weights[0] * kd_weak_loss

                if 2 in choose_logits:
                    # 强增强×文本原型蒸馏
                    if use_adaptive_distillation:
                        kd_strong_loss, strong_stats = dual_adaptive_distillation_vectorized(
                            student_logits_strong, teacher_logits_strong, pseudo_labels,
                            kendall_coeffs_dict[2],
                            base_temp=train_config['distill_temp'],
                            temp_range=temp_range,
                            alpha_range=alpha_range,
                            weights=weights
                        )
                        adaptive_stats[2]['mean_temp'] += strong_stats['mean_temp']
                        adaptive_stats[2]['mean_alpha'] += strong_stats['mean_alpha']
                        adaptive_stats[2]['mean_kendall'] += strong_stats['mean_kendall']
                        adaptive_stats[2]['batch_count'] += 1
                    else:
                        temp = train_config['distill_temp']
                        alpha_val = 0.7
                        
                        soft_teacher = F.softmax(teacher_logits_strong / temp, dim=1)
                        soft_student = F.log_softmax(student_logits_strong / temp, dim=1)
                        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='none').sum(1) * (temp ** 2)
                        
                        hard_loss = F.cross_entropy(student_logits_strong, pseudo_labels, reduction='none')
                        combined_loss = alpha_val * soft_loss + (1 - alpha_val) * hard_loss
                        
                        if weights is not None:
                            kd_strong_loss = (weights * combined_loss).mean()
                        else:
                            kd_strong_loss = combined_loss.mean()
                    
                    kd_loss += logits_weights[1] * kd_strong_loss

                if 3 in choose_logits:
                    # 视觉原型×文本原型蒸馏
                    labels_prototype = torch.arange(len(teacher_logits_prototype)).to(device)
                    if use_adaptive_distillation:
                        kd_prototype_loss, proto_stats = dual_adaptive_distillation_vectorized(
                            student_logits_prototype, teacher_logits_prototype, labels_prototype,
                            kendall_coeffs_dict[3],
                            base_temp=train_config['distill_temp'],
                            temp_range=temp_range,
                            alpha_range=alpha_range
                        )
                        adaptive_stats[3]['mean_temp'] += proto_stats['mean_temp']
                        adaptive_stats[3]['mean_alpha'] += proto_stats['mean_alpha']
                        adaptive_stats[3]['mean_kendall'] += proto_stats['mean_kendall']
                        adaptive_stats[3]['batch_count'] += 1
                    else:
                        temp = train_config['distill_temp']
                        alpha_val = 0.7
                        
                        soft_teacher = F.softmax(teacher_logits_prototype / temp, dim=1)
                        soft_student = F.log_softmax(student_logits_prototype / temp, dim=1)
                        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='none').sum(1) * (temp ** 2)
                        
                        hard_loss = F.cross_entropy(student_logits_prototype, labels_prototype, reduction='none')
                        combined_loss = alpha_val * soft_loss + (1 - alpha_val) * hard_loss
                        kd_prototype_loss = combined_loss.mean()
                    
                    kd_loss += logits_weights[2] * kd_prototype_loss

                if 4 in choose_logits:
                    # 伪标签蒸馏
                    if use_adaptive_distillation:
                        kd_pseudolabel_loss, pseudo_stats = dual_adaptive_distillation_vectorized(
                            student_logits_weak, teacher_logits_weak, pseudo_labels,
                            kendall_coeffs_dict[4],
                            base_temp=train_config['distill_temp'],
                            temp_range=temp_range,
                            alpha_range=(0.8, 1.0),
                            weights=weights
                        )
                        adaptive_stats[4]['mean_temp'] += pseudo_stats['mean_temp']
                        adaptive_stats[4]['mean_alpha'] += pseudo_stats['mean_alpha']
                        adaptive_stats[4]['mean_kendall'] += pseudo_stats['mean_kendall']
                        adaptive_stats[4]['batch_count'] += 1
                    else:
                        # 伪标签蒸馏只使用硬标签
                        kd_pseudolabel_loss = F.cross_entropy(student_logits_weak, pseudo_labels, reduction='none')
                        
                        if weights is not None:
                            kd_pseudolabel_loss = (weights * kd_pseudolabel_loss).mean()
                        else:
                            kd_pseudolabel_loss = kd_pseudolabel_loss.mean()
                    
                    kd_loss += logits_weights[3] * kd_pseudolabel_loss

                if 5 in choose_logits:
                    # DKD损失（解耦知识蒸馏）
                    if use_adaptive_distillation:
                        kd_dkd_loss, dkd_stats = adaptive_dkd_loss(
                            student_logits_weak, teacher_logits_weak, pseudo_labels,
                            kendall_coeffs_dict[5],
                            base_temp=train_config['distill_temp'],
                            temp_range=temp_range,
                            tkl_weight=tkl_weight,
                            ntkl_weight=ntkl_weight,
                            weights=weights
                        )
                        adaptive_stats[5]['mean_temp'] += dkd_stats['mean_temp']
                        adaptive_stats[5]['mean_kendall'] += dkd_stats['mean_kendall']
                        adaptive_stats[5]['batch_count'] += 1
                    else:
                        # 标准DKD损失
                        temp = train_config['distill_temp']
                        gt_mask = _get_gt_mask(student_logits_weak, pseudo_labels)
                        
                        # TKL
                        logits_teacher_target = teacher_logits_weak * gt_mask
                        logits_student_target = student_logits_weak * gt_mask
                        pred_teacher_target = F.softmax(logits_teacher_target / temp, dim=1)
                        log_pred_student_target = F.log_softmax(logits_student_target / temp, dim=1)
                        tkl = F.kl_div(log_pred_student_target, pred_teacher_target, reduction='none').sum(1) * (temp ** 2)
                        
                        # NTKL
                        logits_teacher_nontarget = teacher_logits_weak * (~gt_mask)
                        logits_student_nontarget = student_logits_weak * (~gt_mask)
                        pred_teacher_nontarget = F.softmax(logits_teacher_nontarget / temp, dim=1)
                        log_pred_student_nontarget = F.log_softmax(logits_student_nontarget / temp, dim=1)
                        ntkl = F.kl_div(log_pred_student_nontarget, pred_teacher_nontarget, reduction='none').sum(1) * (temp ** 2)
                        
                        # 组合
                        dkd_combined = tkl_weight * tkl + ntkl_weight * ntkl
                        
                        if weights is not None:
                            kd_dkd_loss = (weights * dkd_combined).mean()
                        else:
                            kd_dkd_loss = dkd_combined.mean()
                    
                    kd_loss += logits_weights[4] * kd_dkd_loss

                # K2模式：添加DPA损失
                if current_mode == 'K2':
                    cls_loss = (weights * F.cross_entropy(student_logits_strong, pseudo_labels, reduction='none')).mean()
                    reg_loss = -train_config['reg'] * (torch.log((F.softmax(student_logits_strong, dim=-1)).mean(0))).mean()
                    
                    logits = train_config['w'] * active_model.get_center().to(device) @ active_model.get_classifier().t()
                    labels = torch.arange(len(logits)).to(device)
                    align_loss = train_config['align'] * F.cross_entropy(logits, labels)
                    
                    # 组合损失
                    dpa_weight = train_config.get('dpa_weight', 1.0)
                    dis_weight = train_config.get('dis_weight', 1.0)
                    loss = dpa_weight * (cls_loss + reg_loss + align_loss) + dis_weight * kd_loss
                else:  # K1模式：只使用知识蒸馏损失
                    cls_loss = torch.tensor(0.0, device=device)
                    reg_loss = torch.tensor(0.0, device=device) 
                    align_loss = torch.tensor(0.0, device=device)
                    loss = train_config.get('dis_weight', 1.0) * kd_loss

        # 累积损失统计
        batch_count += 1
        total_loss_sum += loss.item()
        cls_loss_sum += cls_loss.item()
        reg_loss_sum += reg_loss.item()
        align_loss_sum += align_loss.item()
        kd_loss_sum += kd_loss.item()
        kd_weak_loss_sum += kd_weak_loss.item()
        kd_strong_loss_sum += kd_strong_loss.item()
        kd_prototype_loss_sum += kd_prototype_loss.item()
        kd_pseudolabel_loss_sum += kd_pseudolabel_loss.item()
        kd_dkd_loss_sum += kd_dkd_loss.item()

        # 更新指标
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_cls=cls_loss.item())
        metric_logger.update(loss_reg=reg_loss.item())
        metric_logger.update(loss_align=align_loss.item())
        
        if current_mode in ['K1', 'K2']:
            metric_logger.update(loss_kd=kd_loss.item())
            if 1 in choose_logits:
                metric_logger.update(kd_weak=kd_weak_loss.item())
            if 2 in choose_logits:
                metric_logger.update(kd_strong=kd_strong_loss.item())
            if 3 in choose_logits:
                metric_logger.update(kd_prototype=kd_prototype_loss.item())
            if 4 in choose_logits:
                metric_logger.update(kd_pseudolabel=kd_pseudolabel_loss.item())
            if 5 in choose_logits:
                metric_logger.update(kd_dkd=kd_dkd_loss.item())
        
        metric_logger.update(acc_selected=(pseudo_labels == true_labels).float().mean().item() * 100)
        
        # 计算学生预测准确率
        if current_mode == 'T':
            student_pred = F.softmax(logits_strong_txt, -1).argmax(-1)
        elif current_mode == 'S':
            student_pred = F.softmax(student_logits_strong, -1).argmax(-1)
        elif current_mode in ['K1', 'K2']:
            student_pred = F.softmax(student_logits_weak, -1).argmax(-1)

        metric_logger.update(acc_student=(student_pred == true_labels).float().mean().item() * 100)

        # 更新记忆库
        if current_mode == 'T':
            teacher_memory['features'][idx] = feat_weak.detach().cpu()
            teacher_memory['labels'][idx] = pseudo_labels.detach().cpu()
        else:
            with torch.no_grad():
                student_feat_weak_memory = dual_model.student(img_weak)
            student_memory['features'][idx] = student_feat_weak_memory.detach().cpu()
            student_memory['labels'][idx] = pseudo_labels.detach().cpu()

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        # 反向传播
        active_optimizer.zero_grad()
        if loss_scaler is not None:
            grad_norm = loss_scaler(loss, active_optimizer, clip_grad=1.0, 
                                  parameters=active_model.parameters(), create_graph=False)
            metric_logger.update(grad_norm=grad_norm)
        else:
            loss.backward(create_graph=False)
            active_optimizer.step()
        
        torch.cuda.synchronize()

    # 计算最终的自适应参数统计
    for logit_type in adaptive_stats:
        if adaptive_stats[logit_type]['batch_count'] > 0:
            count = adaptive_stats[logit_type]['batch_count']
            adaptive_stats[logit_type]['mean_temp'] /= count
            adaptive_stats[logit_type]['mean_kendall'] /= count
            if logit_type != 5:  # DKD没有alpha
                adaptive_stats[logit_type]['mean_alpha'] /= count

    # 计算损失占比
    loss_percentages = {}
    if total_loss_sum > 0:
        cls_avg = cls_loss_sum / batch_count
        reg_avg = reg_loss_sum / batch_count
        align_avg = align_loss_sum / batch_count
        kd_avg = kd_loss_sum / batch_count
        kd_weak_avg = kd_weak_loss_sum / batch_count
        kd_strong_avg = kd_strong_loss_sum / batch_count
        kd_prototype_avg = kd_prototype_loss_sum / batch_count
        kd_pseudolabel_avg = kd_pseudolabel_loss_sum / batch_count
        kd_dkd_avg = kd_dkd_loss_sum / batch_count
        total_avg = total_loss_sum / batch_count
        
        loss_percentages.update({
            'cls_avg': cls_avg,
            'reg_avg': reg_avg,
            'align_avg': align_avg,
            'kd_avg': kd_avg,
            'kd_weak_avg': kd_weak_avg,
            'kd_strong_avg': kd_strong_avg,
            'kd_prototype_avg': kd_prototype_avg,
            'kd_pseudolabel_avg': kd_pseudolabel_avg,
            'kd_dkd_avg': kd_dkd_avg,
            'cls_percent': (cls_avg / total_avg) * 100 if total_avg > 0 else 0,
            'reg_percent': (reg_avg / total_avg) * 100 if total_avg > 0 else 0,
            'align_percent': (align_avg / total_avg) * 100 if total_avg > 0 else 0,
            'kd_percent': (kd_avg / total_avg) * 100 if total_avg > 0 and current_mode in ['K1', 'K2'] else 0.0,
            'kd_weak_percent': (kd_weak_avg / total_avg) * 100 if total_avg > 0 and 1 in choose_logits and current_mode in ['K1', 'K2'] else 0.0,
            'kd_strong_percent': (kd_strong_avg / total_avg) * 100 if total_avg > 0 and 2 in choose_logits and current_mode in ['K1', 'K2'] else 0.0,
            'kd_prototype_percent': (kd_prototype_avg / total_avg) * 100 if total_avg > 0 and 3 in choose_logits and current_mode in ['K1', 'K2'] else 0.0,
            'kd_pseudolabel_percent': (kd_pseudolabel_avg / total_avg) * 100 if total_avg > 0 and 4 in choose_logits and current_mode in ['K1', 'K2'] else 0.0,
            'kd_dkd_percent': (kd_dkd_avg / total_avg) * 100 if total_avg > 0 and 5 in choose_logits and current_mode in ['K1', 'K2'] else 0.0,
        })

    print('-----------------------------------------------------------------------')
    print(f"Averaged stats: {epoch} ({current_mode}): {metric_logger}")
    print("损失占比分析:")
    for loss_name, percentage in loss_percentages.items():
        if 'percent' in loss_name:
            print(f"  {loss_name}: {percentage:.2f}%")
    
    # 打印蒸馏损失详细信息
    if current_mode in ['K1', 'K2'] and len(choose_logits) > 1:
        print("多重蒸馏损失分解:")
        if 1 in choose_logits:
            print(f"  弱增强蒸馏损失: {kd_weak_avg:.6f}")
        if 2 in choose_logits:
            print(f"  强增强蒸馏损失: {kd_strong_avg:.6f}")
        if 3 in choose_logits:
            print(f"  原型蒸馏损失: {kd_prototype_avg:.6f}")
        if 4 in choose_logits:
            print(f"  伪标签蒸馏损失: {kd_pseudolabel_avg:.6f}")
        if 5 in choose_logits:
            print(f"  DKD蒸馏损失: {kd_dkd_avg:.6f}")
    
    # 打印自适应参数统计
    if current_mode in ['K1', 'K2'] and use_adaptive_distillation:
        print("自适应蒸馏参数统计:")
        logits_type_descriptions = {
            1: "弱增强×文本原型",
            2: "强增强×文本原型", 
            3: "视觉原型×文本原型",
            4: "伪标签蒸馏",
            5: "DKD损失"
        }
        for logit_type in choose_logits:
            if logit_type in adaptive_stats and adaptive_stats[logit_type]['batch_count'] > 0:
                print(f"  {logits_type_descriptions[logit_type]}:")
                print(f"    平均肯德尔系数: {adaptive_stats[logit_type]['mean_kendall']:.4f}")
                print(f"    平均温度: {adaptive_stats[logit_type]['mean_temp']:.2f}")
                if logit_type != 5:
                    print(f"    平均Alpha: {adaptive_stats[logit_type]['mean_alpha']:.3f}")
        print(f"  温度范围: {temp_range}")
        if alpha_range:
            print(f"  Alpha范围: {alpha_range}")
    
    print('-----------------------------------------------------------------------')

    # 更新中心点
    if current_mode == 'T':
        dual_model.teacher.center_init_fixed(get_center(args, teacher_memory, dual_model.teacher.get_classifier()))
    else:
        dual_model.student.center_init_fixed(get_center(args, student_memory, dual_model.student.get_classifier()))
    
    metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    metrics.update(loss_percentages)
    metrics.update({'training_mode': current_mode})
    
    # 添加自适应参数统计到返回的metrics中
    if current_mode in ['K1', 'K2'] and use_adaptive_distillation:
        for logit_type in choose_logits:
            if logit_type in adaptive_stats and adaptive_stats[logit_type]['batch_count'] > 0:
                metrics.update({
                    f'adaptive_mean_kendall_type_{logit_type}': adaptive_stats[logit_type]['mean_kendall'],
                    f'adaptive_mean_temp_type_{logit_type}': adaptive_stats[logit_type]['mean_temp'],
                })
                if logit_type != 5:
                    metrics.update({
                        f'adaptive_mean_alpha_type_{logit_type}': adaptive_stats[logit_type]['mean_alpha'],
                    })
    
    return metrics, teacher_memory, student_memory, prob_list


@torch.no_grad()
def evaluate_dual(data_loader, dual_model, device):
    """评估两个模型的性能"""
    results = {}
    
    for model_name, model in [('teacher', dual_model.teacher), ('student', dual_model.student)]:
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f'Test {model_name}:'
        model.eval()
        
        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0]
            target = batch[1]
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # 前向传播
            output = model(images)
            logits = 100. * output @ model.get_classifier().t()
            
            # 计算准确率
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            metric_logger.update(acc=acc1.item())
        
        print(f"* {model_name} Acc@1 {metric_logger.acc.global_avg:.3f}")
        results[f'{model_name}_acc'] = metric_logger.acc.global_avg
    
    return results