import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model import clip_classifier
import copy

class DualModelManager(nn.Module):
    """管理教师和学生模型的类 - 支持多种Backbone组合"""
    
    def __init__(self, args):
        super().__init__()
        
        # 从args获取教师和学生的模型名称
        teacher_model_name = getattr(args, 'teacher_backbone', 'ViT-L/14')
        student_model_name = getattr(args, 'student_backbone', 'RN101')
        
        # 创建教师模型
        teacher_args = copy.deepcopy(args)
        teacher_args.clip_model = teacher_model_name
        self.teacher = clip_classifier(teacher_args)
        
        # 创建学生模型
        student_args = copy.deepcopy(args)
        student_args.clip_model = student_model_name
        self.student = clip_classifier(student_args)
        
        # 记录模型信息
        self.teacher_backbone = teacher_model_name
        self.student_backbone = student_model_name
        
        # 获取模型维度信息
        from utils.model import get_model_embed_dim
        teacher_dim = get_model_embed_dim(teacher_model_name)
        student_dim = get_model_embed_dim(student_model_name)
        
        # 训练策略配置
        self.training_strategy = getattr(args, 'training_strategy', None)
        if self.training_strategy is None:
            self.teacher_warmup_epochs = 3
            self.use_strategy_list = False
        else:
            self.use_strategy_list = True
            print(f"使用自定义训练策略: {self.training_strategy}")
        
        print(f"双模型配置:")
        print(f"  教师模型: {teacher_model_name} (特征维度: {teacher_dim})")
        print(f"  学生模型: {student_model_name} (特征维度: {student_dim})")
        print(f"  维度比例: {teacher_dim/student_dim:.2f}x")
        print(f"  简化蒸馏: 仅使用logits蒸馏，不使用特征和中心点蒸馏")
        
        # 如果维度差异大，可能需要调整蒸馏策略
        if abs(teacher_dim - student_dim) > 256:
            print(f"  ⚠️  警告: 教师和学生模型维度差异较大 ({teacher_dim} vs {student_dim})")
            print(f"  建议: 调整温度范围和alpha范围以适应维度差异")
        
    def set_training_mode(self, mode, epoch=None):
        """设置当前训练模式"""
        self.training_mode = mode
        
        if mode == 'T':  # 训练教师
            self._freeze_model(self.student)
            self._unfreeze_model(self.teacher)
            print(f"Epoch {epoch}: 训练教师模型")
        elif mode in ['S', 'K1', 'K2']:  # 训练学生
            self._freeze_model(self.teacher)
            self._unfreeze_model(self.student)
            if mode == 'S':
                print(f"Epoch {epoch}: 训练学生模型 (原始DPA损失)")
            elif mode == 'K1':
                print(f"Epoch {epoch}: 训练学生模型 (仅logits蒸馏损失)")
            elif mode == 'K2':
                print(f"Epoch {epoch}: 训练学生模型 (logits蒸馏+DPA损失)")
    
    def _freeze_model(self, model):
        """冻结模型参数"""
        for param in model.parameters():
            param.requires_grad = False
    
    def _unfreeze_model(self, model):
        """解冻特定层的参数"""
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "classifier" in name or 'ln' in name or 'bn' in name:
                param.requires_grad = True
    
    def get_active_model(self):
        """获取当前活跃的模型"""
        if self.training_mode == 'T':
            return self.teacher
        else:
            return self.student
    
    def get_teacher(self):
        return self.teacher
    
    def get_student(self):
        return self.student
    
    def should_switch_mode(self, epoch):
        """决定是否切换训练模式"""
        if self.use_strategy_list:
            if epoch < len(self.training_strategy):
                return self.training_strategy[epoch]
            else:
                return self.training_strategy[-1]
        else:
            if epoch < self.teacher_warmup_epochs:
                return 'T'
            else:
                return 'T' if epoch % 2 == 0 else 'S'


def simple_logits_distillation_loss(student_logits, teacher_logits, pseudo_labels, 
                                   temp=4.0, alpha=0.7, weights=None):
    """
    简化的logits蒸馏损失函数 - 只使用输出层蒸馏
    """
    # 蒸馏损失 - 使用教师输出作为软标签
    kd_loss = nn.KLDivLoss(reduction='none')(
        F.log_softmax(student_logits / temp, dim=1), 
        F.softmax(teacher_logits / temp, dim=1)
    ).sum(dim=1) * (temp * temp)
    
    # 硬标签损失 - 学生对伪标签的交叉熵
    ce_loss = F.cross_entropy(student_logits, pseudo_labels, reduction='none')
    
    # 组合损失
    combined_loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
    
    # 应用样本权重
    if weights is not None:
        combined_loss = weights * combined_loss
    
    return combined_loss.mean()


# 保持向后兼容的别名
self_distillation_loss = simple_logits_distillation_loss