import argparse
import datetime
import numpy as np
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import json
import os
from contextlib import suppress
import random
from pathlib import Path 
from collections import OrderedDict
import copy
import utils.utils as utils
from utils.build_dataset import build_dataset
from utils.multi_model3 import DualModelManager
from utils.utils import NativeScalerWithGradNormCount as NativeScaler
from engine_dual_training9 import train_one_epoch_dual, evaluate_dual
from utils.center14 import build_memory
import warnings
from utils.capability_probe2 import CapabilityGapProbe
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser('Dual model training with adaptive distillation', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--eval_freq', default=1, type=int) 
    
    # 训练策略配置
    parser.add_argument('--training_strategy', type=str, default=None,
                       help='Training strategy as comma-separated string, e.g., "T,T,T,S,S,K1,K1,K2,K2"')
    
    # 教师模型选择配置
    parser.add_argument('--choose_teacher_model', type=str, default=None,
                       help='Teacher model sequence to use for distillation, e.g., "1,2,3" or "8,9,10"')
    
    # 蒸馏logits类型选择配置
    parser.add_argument('--choose_logits', type=str, default='1,2,5',
                       help='Logits types for distillation as comma-separated string, e.g., "1,2,3,4,5". '
                            '1: weak×text, 2: strong×text, 3: visual×text prototype, 4: pseudo-label, 5: DKD loss')
    
    # 自适应蒸馏配置
    parser.add_argument('--use_adaptive_distillation', action='store_true',
                       help='Use adaptive temperature and alpha based on Kendall correlation')
    parser.add_argument('--no_adaptive_distillation', dest='use_adaptive_distillation', action='store_false')
    parser.set_defaults(use_adaptive_distillation=True)
    
    # 温度和Alpha范围配置
    parser.add_argument('--temp_range', type=str, default='2.0,8.0',
                       help='Temperature range for adaptive distillation as "min,max"')
    parser.add_argument('--alpha_range', type=str, default='0.3,0.9',
                       help='Alpha range for adaptive distillation as "min,max"')
    
    # 蒸馏损失权重配置
    parser.add_argument('--logits_weights', type=str, default='1.0,1.0,1.0,1.0,1.0',
                       help='Weights for different logits types as comma-separated string')
    
    # DKD特定参数
    parser.add_argument('--tkl_weight', type=float, default=1.0,
                       help='Target Knowledge Loss weight for DKD')
    parser.add_argument('--ntkl_weight', type=float, default=8.0,
                       help='Non-Target Knowledge Loss weight for DKD')
    
    parser.add_argument('--kendall_threshold', type=float, default=None,
                    help='Kendall correlation threshold for sample filtering (optional, for backward compatibility)')
    

    parser.add_argument('--teacher_backbone', type=str, default='ViT-L/14',
                       choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px',
                               'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'],
                       help='Teacher model backbone')
    
    parser.add_argument('--student_backbone', type=str, default='ViT-B/32',
                       choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px',
                               'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'],
                       help='Student model backbone')



    # CLIP parameters
    parser.add_argument("--template", default='templates.json', type=str)
    parser.add_argument("--classname", default='classes.json', type=str)
    parser.add_argument('--image_mean', default=(0.48145466, 0.4578275, 0.40821073)) 
    parser.add_argument('--image_std', default=(0.26862954, 0.26130258, 0.27577711)) 
    parser.add_argument('--input_size', default=224, type=int, help='images input size') 
    
    # training parameters
    parser.add_argument("--train_config", default='train_configs.json', type=str)
    
    # Optimizer parameters
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=0.65)
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N')
    
    # Augmentation parameters  
    parser.add_argument('--train_crop_min', default=0.3, type=float)
    parser.add_argument('--color_jitter', type=float, default=0, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME')
    parser.add_argument('--train_interpolation', type=str, default='bicubic')
    
    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int)
    parser.add_argument('--dataset', default='imagenet', type=str)
    parser.add_argument('--output_dir', default='', help='path to save checkpoint and log')
    parser.add_argument('--device', default='cuda:0', type=str, 
                       help='device to use for training (e.g., cuda:0, cuda:1)')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--amp', action='store_true')
    
    return parser.parse_args()

def parse_training_strategy(strategy_str):
    """解析训练策略字符串"""
    if strategy_str is None:
        return None
    
    strategy_list = [s.strip().upper() for s in strategy_str.split(',')]
    
    # 验证策略有效性
    valid_modes = {'T', 'S', 'K1', 'K2'}
    for mode in strategy_list:
        if mode not in valid_modes:
            raise ValueError(f"无效的训练模式: {mode}. 有效模式: {valid_modes}")
    
    return strategy_list

def parse_teacher_model_sequence(sequence_str):
    """解析教师模型选择序列"""
    if sequence_str is None:
        return None
    
    try:
        sequence_list = [int(s.strip()) for s in sequence_str.split(',')]
        return sequence_list
    except ValueError as e:
        raise ValueError(f"无效的教师模型序列: {sequence_str}. 应该是逗号分隔的数字，如 '1,2,3'")

def parse_choose_logits(logits_str):
    """解析蒸馏logits类型选择"""
    if logits_str is None:
        return [5]  # 默认使用DKD损失
    
    try:
        logits_list = [int(s.strip()) for s in logits_str.split(',')]
        # 验证logits类型有效性
        valid_logits = {1, 2, 3, 4, 5}
        for logit_type in logits_list:
            if logit_type not in valid_logits:
                raise ValueError(f"无效的logits类型: {logit_type}. 有效类型: {valid_logits}")
        return logits_list
    except ValueError as e:
        raise ValueError(f"无效的logits选择: {logits_str}. 应该是逗号分隔的数字，如 '1,2,3,4,5'")

def parse_logits_weights(weights_str):
    """解析蒸馏损失权重"""
    if weights_str is None:
        return [1.0, 1.0, 1.0, 1.0, 1.0]  # 默认权重
    
    try:
        weights_list = [float(s.strip()) for s in weights_str.split(',')]
        if len(weights_list) != 5:
            raise ValueError(f"权重数量应为5个，实际为{len(weights_list)}个")
        return weights_list
    except ValueError as e:
        raise ValueError(f"无效的权重配置: {weights_str}. 应该是5个逗号分隔的浮点数，如 '1.0,1.0,1.0,1.0,1.0'")

def parse_range(range_str):
    """解析范围字符串 (例如 "2.0,8.0")"""
    try:
        parts = [float(s.strip()) for s in range_str.split(',')]
        if len(parts) != 2:
            raise ValueError(f"范围应包含两个值，实际为{len(parts)}个")
        return tuple(parts)
    except ValueError as e:
        raise ValueError(f"无效的范围配置: {range_str}. 应该是两个逗号分隔的数字，如 '2.0,8.0'")

def get_next_teacher_sequence_number(dataset_name):
    """获取下一个教师模型的序列号"""
    pth_dir = Path(f"./pth00/{dataset_name}")
    if not pth_dir.exists():
        return 1
    
    # 查找所有教师权重文件
    teacher_files = list(pth_dir.glob("*T.pth"))
    if not teacher_files:
        return 1
    
    # 提取序列号并找到最大值
    max_sequence = 0
    for file in teacher_files:
        try:
            sequence_num = int(file.stem.replace('T', ''))
            max_sequence = max(max_sequence, sequence_num)
        except ValueError:
            continue
    
    return max_sequence + 1

def save_teacher_weights_sequential(teacher_model, dataset_name, training_mode):
    """按顺序保存教师模型权重"""
    if training_mode != 'T':
        return None
    
    # 创建保存目录
    save_dir = Path(f"./pth0/{dataset_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取下一个序列号
    sequence_num = get_next_teacher_sequence_number(dataset_name)
    
    # 保存权重文件，文件名格式：{sequence_num}T.pth
    save_path = save_dir / f"{sequence_num}T.pth"
    
    # 保存模型状态字典
    torch.save({
        'model_state_dict': teacher_model.state_dict(),
        'sequence_num': sequence_num,
        'training_mode': training_mode,
        'center': teacher_model.center.clone() if hasattr(teacher_model, 'center') else None
    }, save_path)
    
    print(f"✅ 教师模型权重已保存: {save_path} (序列号: {sequence_num})")
    return sequence_num

def load_teacher_weights_by_sequence(teacher_model, dataset_name, sequence_num):
    """根据序列号加载教师模型权重"""
    load_path = Path(f"./pth0/{dataset_name}/{sequence_num}T.pth")
    
    if not load_path.exists():
        print(f"❌ 教师权重文件不存在: {load_path}")
        return False
    
    try:
        checkpoint = torch.load(load_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 如果保存了center，也加载它
        if checkpoint.get('center') is not None and hasattr(teacher_model, 'center'):
            teacher_model.center = checkpoint['center']
        
        print(f"✅ 成功加载教师权重: {load_path} (序列号: {sequence_num})")
        return True
    except Exception as e:
        print(f"❌ 加载教师权重失败: {load_path}, 错误: {e}")
        return False

def check_and_load_existing_teacher_weights_by_sequence(dual_model, dataset_name, sequence_num):
    """检查并加载指定序列号的教师权重"""
    load_path = Path(f"./pth0/{dataset_name}/{sequence_num}T.pth")
    
    if load_path.exists():
        try:
            checkpoint = torch.load(load_path, map_location='cpu')
            dual_model.teacher.load_state_dict(checkpoint['model_state_dict'])
            
            # 如果保存了center，也加载它
            if checkpoint.get('center') is not None and hasattr(dual_model.teacher, 'center'):
                dual_model.teacher.center = checkpoint['center']
            
            print(f"✅ 发现并加载已存在的教师权重: {load_path} (序列号: {sequence_num})")
            return True
        except Exception as e:
            print(f"❌ 加载教师权重失败: {load_path}, 错误: {e}")
            return False
    else:
        print(f"📝 教师权重文件不存在: {load_path}")
        return False

def get_available_teacher_sequences(dataset_name):
    """获取可用的教师模型序列号列表"""
    pth_dir = Path(f"./pth0/{dataset_name}")
    if not pth_dir.exists():
        return []
    
    teacher_files = list(pth_dir.glob("*T.pth"))
    sequences = []
    
    for file in teacher_files:
        try:
            sequence_num = int(file.stem.replace('T', ''))
            sequences.append(sequence_num)
        except ValueError:
            continue
    
    return sorted(sequences)

def build_teacher_usage_plan(training_strategy, teacher_sequence):
    """构建教师模型使用计划"""
    if not training_strategy or not teacher_sequence:
        return {}
    
    # 找到所有需要蒸馏的epoch
    distill_epochs = []
    for epoch, mode in enumerate(training_strategy):
        if mode in ['K1', 'K2']:
            distill_epochs.append(epoch)
    
    # 检查教师序列数量是否匹配
    if len(distill_epochs) != len(teacher_sequence):
        raise ValueError(
            f"蒸馏epoch数量 ({len(distill_epochs)}) 与指定的教师模型数量 ({len(teacher_sequence)}) 不匹配！\n"
            f"需要蒸馏的epoch: {distill_epochs}\n"
            f"指定的教师序列: {teacher_sequence}\n"
            f"请确保 choose_teacher_model 参数包含 {len(distill_epochs)} 个教师模型序列号"
        )
    
    # 构建epoch到教师序列号的一一对应映射
    teacher_plan = {}
    for i, epoch in enumerate(distill_epochs):
        teacher_plan[epoch] = teacher_sequence[i]
    
    return teacher_plan

def setup_optimizers(dual_model, train_config):
    """为教师和学生模型分别设置优化器"""
    
    def get_params(model):
        params = []
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in trainable_params
                       if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.1},
            {'params': [p for n, p in trainable_params
                       if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters
    
    # 教师模型优化器
    teacher_params = get_params(dual_model.teacher)
    teacher_optimizer = optim.AdamW(teacher_params, lr=train_config['lr'])
    
    # 学生模型优化器
    student_params = get_params(dual_model.student)
    student_optimizer = optim.AdamW(student_params, lr=train_config['lr'])
    
    return teacher_optimizer, student_optimizer

def main(args):

    # GPU设置和验证
    print(f"🎯 指定使用设备: {args.device}")
    
    # 设置CUDA设备
    if 'cuda:' in args.device:
        gpu_id = int(args.device.split(':')[1])
        if torch.cuda.is_available():
            if gpu_id >= torch.cuda.device_count():
                raise ValueError(f"指定的GPU {gpu_id} 不存在！可用GPU数量: {torch.cuda.device_count()}")
            torch.cuda.set_device(gpu_id)
            print(f"🎯 设置当前CUDA设备为GPU {gpu_id}")
        else:
            raise RuntimeError("CUDA不可用！")
    
    device = torch.device(args.device)
    
    # 验证GPU使用情况
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"🎯 当前使用GPU {current_device}: {device_name}")
        print(f"🎯 可用GPU数量: {torch.cuda.device_count()}")
        
        # 显示GPU内存使用情况
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        print(f"🎯 GPU {current_device} 内存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
    
    # 解析训练策略和教师模型使用计划
    if args.training_strategy:
        args.training_strategy = parse_training_strategy(args.training_strategy)
        print(f"\n使用自定义训练策略: {args.training_strategy}")
        
        # 统计需要蒸馏的epoch数量和教师训练的epoch数量
        distill_epochs = [i for i, mode in enumerate(args.training_strategy) if mode in ['K1', 'K2']]
        teacher_epochs = [i for i, mode in enumerate(args.training_strategy) if mode == 'T']
        
        if distill_epochs:
            print(f"需要蒸馏的epoch: {distill_epochs} (共{len(distill_epochs)}个)")
        if teacher_epochs:
            print(f"需要教师训练的epoch: {teacher_epochs} (共{len(teacher_epochs)}个)")

    # 解析教师模型选择序列
    teacher_sequence = None
    teacher_usage_plan = {}
    teacher_training_plan = {}
    
    if args.choose_teacher_model:
        teacher_sequence = parse_teacher_model_sequence(args.choose_teacher_model)
        print(f"\n指定教师模型序列: {teacher_sequence} (共{len(teacher_sequence)}个)")
        
        if args.training_strategy:
            try:
                teacher_usage_plan = build_teacher_usage_plan(args.training_strategy, teacher_sequence)
                print("教师模型使用计划:")
                for epoch, seq_num in teacher_usage_plan.items():
                    mode = args.training_strategy[epoch]
                    print(f"  Epoch {epoch} ({mode}): 使用教师模型 {seq_num}T.pth")
                
                # 构建教师训练计划
                teacher_epochs = [i for i, mode in enumerate(args.training_strategy) if mode == 'T']
                if teacher_epochs:
                    needed_teachers = set(teacher_sequence)
                    available_teachers = set(get_available_teacher_sequences(args.dataset))
                    missing_teachers = needed_teachers - available_teachers
                    
                    if missing_teachers:
                        missing_teachers = sorted(missing_teachers)
                        print(f"\n需要训练的教师模型: {missing_teachers}")
                        
                        for i, epoch in enumerate(teacher_epochs):
                            if i < len(missing_teachers):
                                teacher_training_plan[epoch] = missing_teachers[i]
                        
                        print("教师训练计划:")
                        for epoch, seq_num in teacher_training_plan.items():
                            print(f"  Epoch {epoch} (T): 训练教师模型 {seq_num}T.pth")
                    else:
                        print("\n✅ 所有需要的教师模型都已存在，无需训练新的教师模型")
                        
            except ValueError as e:
                print(f"❌ 教师模型配置错误: {e}")
                return

    # 解析蒸馏logits类型选择
    choose_logits = parse_choose_logits(args.choose_logits)
    print(f"\n🎯 蒸馏logits类型配置: {choose_logits}")
    logits_type_descriptions = {
        1: "弱增强×文本原型蒸馏",
        2: "强增强×文本原型蒸馏", 
        3: "视觉原型×文本原型蒸馏",
        4: "伪标签蒸馏",
        5: "DKD损失(解耦知识蒸馏)"
    }
    for logit_type in choose_logits:
        print(f"  类型 {logit_type}: {logits_type_descriptions[logit_type]}")

    # 解析蒸馏损失权重
    logits_weights = parse_logits_weights(args.logits_weights)
    print(f"\n🎯 蒸馏损失权重配置: {logits_weights}")
    for i, weight in enumerate(logits_weights, 1):
        if i in choose_logits:
            print(f"  类型 {i} 权重: {weight} ✓ (启用)")
        else:
            print(f"  类型 {i} 权重: {weight} (未启用)")

    # 解析温度和alpha范围
    temp_range = parse_range(args.temp_range)
    alpha_range = parse_range(args.alpha_range)
    
    print(f"\n🎯 自适应蒸馏配置:")
    print(f"  启用自适应蒸馏: {args.use_adaptive_distillation}")
    if args.use_adaptive_distillation:
        print(f"  温度范围: {temp_range}")
        print(f"  Alpha范围: {alpha_range}")

    # 保存日志
    log_path = os.path.join(args.output_dir, f"{args.dataset}_dual_adaptive_trainlog.txt")
    Path(args.output_dir).parent.mkdir(parents=True, exist_ok=True)
    
    # 记录训练策略到日志
    log_args = dict(args._get_kwargs())
    if args.training_strategy:
        log_args['training_strategy'] = ','.join(args.training_strategy)
    if args.choose_teacher_model:
        log_args['choose_teacher_model'] = args.choose_teacher_model
    log_args['choose_logits'] = ','.join(map(str, choose_logits))
    log_args['logits_weights'] = ','.join(map(str, logits_weights))
    log_args['use_adaptive_distillation'] = args.use_adaptive_distillation
    log_args['temp_range'] = ','.join(map(str, temp_range))
    log_args['alpha_range'] = ','.join(map(str, alpha_range))
    
    with open(log_path, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(log_args) + "\n")
    


    # 训练配置
    train_config_path = os.path.join("./json_files", args.train_config)
    with open(train_config_path, 'r') as train_config_file:
        train_config_data = json.load(train_config_file)
    
    # 使用ViT-L/14的配置作为基础配置
    train_config = train_config_data[args.dataset + '_ViT-L/14']
    
    # 添加蒸馏配置（保持与train16.py相同的逻辑）
    train_config.update({
        'distill_temp': 4.0,           # 蒸馏温度
        'dis_weight': 1.0,             # 蒸馏损失权重
        'dpa_weight': 1.0,             # K2模式下DPA损失的权重
        'choose_logits': choose_logits,  # 选择的蒸馏logits类型
        'logits_weights': logits_weights,  # 蒸馏损失权重
        'tkl_weight': args.tkl_weight,   # DKD目标知识损失权重
        'ntkl_weight': args.ntkl_weight,  # DKD非目标知识损失权重
        
        # 新增：自适应蒸馏配置
        'use_adaptive_distillation': args.use_adaptive_distillation,
        'temp_range': temp_range,      # 温度范围 (min, max)
        'alpha_range': alpha_range,    # Alpha范围 (min, max)
    })

    # 设置肯德尔相关系数阈值（如果需要的话，保持兼容）
    if args.kendall_threshold is not None:
        train_config['kendall_threshold'] = args.kendall_threshold
    else:
        dataset_kendall_thresholds = {
            'cars': 0.5, 'dtd': 0.0, 'fgvc': 0.6, 'food101': 0.0,
            'flowers': 0.5, 'pets': 0.5, 'aircraft': 0.0,
            'caltech101': 0.0, 'eurosat': 0.0, 'ucf101': 0.0,
        }
        train_config['kendall_threshold'] = dataset_kendall_thresholds.get(args.dataset, 0.0)

    # 打印配置摘要
    print("\n" + "="*80)
    print("🎯 自适应知识蒸馏配置摘要:")
    print(f"  数据集: {args.dataset}")
    print(f"  启用自适应蒸馏: {train_config['use_adaptive_distillation']}")
    if train_config['use_adaptive_distillation']:
        print(f"  温度范围: {train_config['temp_range']}")
        print(f"  Alpha范围: {train_config['alpha_range']}")
    print(f"  蒸馏温度: {train_config['distill_temp']}")
    print(f"  蒸馏权重: {train_config['dis_weight']}")
    print(f"  DPA权重(K2模式): {train_config['dpa_weight']}")
    if 5 in choose_logits:
        print(f"  DKD-TKL权重: {train_config['tkl_weight']}")
        print(f"  DKD-NTKL权重: {train_config['ntkl_weight']}")
    print("="*80 + "\n")



    # 训练配置 - 根据Backbone选择配置
    # 修正：保持斜杠，与 JSON 文件键格式一致
    config_key = f"{args.dataset}_{args.teacher_backbone}"

    if config_key not in train_config_data:
        config_key_no_slash = f"{args.dataset}_{args.teacher_backbone.replace('/', '')}"
        if config_key_no_slash in train_config_data:
            config_key = config_key_no_slash
        else:
            config_key = f"{args.dataset}_ViT-L/14"
            print(f"⚠️  未找到 {args.dataset}_{args.teacher_backbone} 的配置，使用默认配置 {config_key}")
    else:
        print(f"✅ 使用配置: {config_key}")

    train_config = train_config_data[config_key]
    
    if not args.output_dir:
        teacher_name = args.teacher_backbone.replace('/', '-')
        student_name = args.student_backbone.replace('/', '-')
        backbone_suffix = f"_T{teacher_name}_S{student_name}"
        
        # 简化策略后缀
        if args.training_strategy:
            # 统计各模式的数量
            from collections import Counter
            mode_counts = Counter(args.training_strategy)
            strategy_suffix = f"_st_{''.join([f'{k}{v}' for k, v in sorted(mode_counts.items())])}"
        else:
            strategy_suffix = ""
        
        # 简化教师序列后缀
        if args.choose_teacher_model:
            teacher_nums = args.choose_teacher_model.replace(',', '')
            # 只保留前3个和后3个教师编号
            if len(teacher_nums) > 6:
                teacher_suffix = f"_tea_{teacher_nums[:3]}..{teacher_nums[-3:]}"
            else:
                teacher_suffix = f"_tea_{teacher_nums}"
        else:
            teacher_suffix = ""
        
        # 简化logits后缀
        logits_suffix = f"_lg_{''.join(map(str, choose_logits))}"
        
        # 简化自适应参数后缀
        if args.use_adaptive_distillation:
            adaptive_suffix = "_adp"
            temp_suffix = f"_T{int(temp_range[0])}-{int(temp_range[1])}"
            alpha_suffix = f"_A{int(alpha_range[0]*10)}-{int(alpha_range[1]*10)}"
        else:
            adaptive_suffix = ""
            temp_suffix = ""
            alpha_suffix = ""
        
        # 组合成简短的目录名
        args.output_dir = os.path.join(
            'output', 
            args.dataset,
            f"Dual{backbone_suffix}_e{train_config['epochs']}_lr{train_config['lr']:.6f}"
            f"{strategy_suffix}{teacher_suffix}{logits_suffix}{adaptive_suffix}{temp_suffix}{alpha_suffix}"
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # 构建数据集
    batch_size = train_config["model_patch_size"]
    dataset_train, len_original = build_dataset(is_train=True, args=args)
    
    print(f"\n训练集信息:")
    print(f"  原始长度: {len_original}")
    print(f"  实际长度: {len(dataset_train)}")
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    len_data_loader_train = len(data_loader_train)
    args.len_original = len_original
    
    # 验证集
    dataset_val, _ = build_dataset(is_train=False, args=args)  
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        batch_size=4*batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # 确保在创建模型前设置正确的设备
    with torch.cuda.device(device):
        print(f"\n🔧 在设备 {device} 上创建双模型管理器...")
        # 创建双模型管理器
        dual_model = DualModelManager(args)
        dual_model.to(device) 
        print(f"✅ 双模型已成功移动到设备 {device}")
    
    args.nb_classes = len(dual_model.teacher.classnames)
    
    print(f"\n双模型信息:")
    print(f"  类别数: {args.nb_classes}")
    teacher_model_name = dual_model.teacher.model.visual.__class__.__name__
    student_model_name = dual_model.student.model.visual.__class__.__name__
    print(f"  教师模型架构: {teacher_model_name}")
    print(f"  学生模型架构: {student_model_name}")
    
    # 检查现有教师权重和使用计划
    print("\n" + "="*80)
    print("检查现有教师权重...")
    available_sequences = get_available_teacher_sequences(args.dataset)
    if available_sequences:
        print(f"可用的教师权重序列: {available_sequences}")
        for seq in available_sequences:
            print(f"  ✅ {seq}T.pth")
    else:
        print("  📝 暂无可用的教师权重")
    
    # 检查教师使用计划的可行性
    if teacher_usage_plan:
        missing_teachers = []
        for epoch, seq_num in teacher_usage_plan.items():
            if seq_num not in available_sequences:
                missing_teachers.append(seq_num)
        
        if missing_teachers:
            missing_teachers = sorted(set(missing_teachers))
            print(f"⚠️  警告: 计划使用但不存在的教师权重: {missing_teachers}")
            print("   这些epoch将使用当前教师模型状态")
        else:
            print("✅ 所有计划使用的教师权重都已准备就绪")
    
    print("="*80 + "\n")
    
    # 构建记忆库
    print("构建教师模型记忆库...")
    teacher_memory_args = copy.deepcopy(args)
    teacher_memory_args.clip_model = args.teacher_backbone  # ✅ 使用教师模型的backbone
    teacher_center, teacher_memory = build_memory(
        teacher_memory_args, 
        dual_model.teacher,
        args.dataset, 
        data_loader_train, 
        len_original, 
        dual_model.teacher.model.embed_dim
    )
    dual_model.teacher.center_init_fixed(teacher_center)

    print("构建学生模型记忆库...")
    student_memory_args = copy.deepcopy(args)
    student_memory_args.clip_model = args.student_backbone  # ✅ 使用学生模型的backbone
    student_center, student_memory = build_memory(
        student_memory_args, 
        dual_model.student, 
        args.dataset, 
        data_loader_train, 
        len_original, 
        dual_model.student.model.embed_dim
    )
    dual_model.student.center_init_fixed(student_center)

    
    prob_list = []
    
    # 设置优化器
    teacher_optimizer, student_optimizer = setup_optimizers(dual_model, train_config)
    
    # 初始化能力监控探针（可选）
    capability_probe = CapabilityGapProbe(
        alpha=0.1,
        cooldown_epochs=3,
        ema_decay=0.9
    )
    
    # 训练配置
    args.lr = train_config['lr']
    args.min_lr = args.min_lr * 2
    args.epochs = train_config['epochs']
    args.eval_freq = train_config['eval_freq']
    
    # 如果使用自定义策略，确保epochs足够
    if args.training_strategy:
        min_epochs = len(args.training_strategy)
        if args.epochs < min_epochs:
            print(f"警告: 设置的epochs ({args.epochs}) 小于策略长度 ({min_epochs})，自动调整为 {min_epochs}")
            args.epochs = min_epochs
    
    n_parameters_teacher = sum(p.numel() for p in dual_model.teacher.parameters() if p.requires_grad)
    n_parameters_student = sum(p.numel() for p in dual_model.student.parameters() if p.requires_grad)
    
    print('-----------------------------------------------------------------------')
    print(f'Teacher parameters: {n_parameters_teacher}')
    print(f'Student parameters: {n_parameters_student}')
    print('-----------------------------------------------------------------------')
    
    loss_scaler = None
    amp_autocast = suppress
    
    # 学习率调度
    num_training_steps_per_epoch = len_data_loader_train
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    
    # 开始训练
    print(f"\n开始双模型自适应蒸馏训练，总共 {args.epochs} 个epochs")
    if args.training_strategy:
        print("\n训练策略概览:")
        for i, mode in enumerate(args.training_strategy):
            if i < args.epochs:
                mode_desc = {
                    'T': '教师训练',
                    'S': '学生(DPA)',
                    'K1': '学生(自适应蒸馏)' if args.use_adaptive_distillation else '学生(标准蒸馏)',
                    'K2': '学生(自适应蒸馏+DPA)' if args.use_adaptive_distillation else '学生(标准蒸馏+DPA)'
                }
                teacher_info = f" [使用教师{teacher_usage_plan.get(i, '当前状态')}]" if i in teacher_usage_plan else ""
                print(f"  Epoch {i}: {mode_desc[mode]}{teacher_info}")
    
    start_time = time.time()
    max_teacher_accuracy = 0.0
    max_student_accuracy = 0.0
    
    evaluation_accuracies = []
    
    for epoch in range(args.start_epoch, args.epochs):
        # 确定当前epoch的训练模式
        current_mode = dual_model.should_switch_mode(epoch)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}: 训练模式 {current_mode}")
        
        # 处理教师训练模式
        if current_mode == 'T':
            # 检查是否需要训练教师模型
            if epoch in teacher_training_plan:
                target_sequence = teacher_training_plan[epoch]
                
                # 检查是否已有该序列号的教师权重
                if check_and_load_existing_teacher_weights_by_sequence(dual_model, args.dataset, target_sequence):
                    print(f"🚀 跳过教师训练，使用已有权重 (序列号: {target_sequence})")
                    
                    # 直接进行评估
                    test_stats = evaluate_dual(data_loader_val, dual_model, device)
                    teacher_acc = test_stats['teacher_acc']
                    student_acc = test_stats['student_acc']
                    
                    print(f"Epoch {epoch} ({current_mode}) - Teacher Acc: {teacher_acc:.1f}%, Student Acc: {student_acc:.1f}% (使用已有权重)")
                    
                    # 记录评估结果
                    evaluation_accuracies.append({
                        'epoch': epoch,
                        'teacher_acc': teacher_acc,
                        'student_acc': student_acc,
                        'training_mode': current_mode,
                        'teacher_sequence': target_sequence
                    })
                    
                    # 记录到日志
                    log_data = {
                        'epoch': epoch,
                        'teacher_acc': teacher_acc,
                        'student_acc': student_acc,
                        'training_mode': current_mode,
                        'train_loss': 0.0,
                        'skipped_training': True,
                        'teacher_sequence': target_sequence
                    }
                    
                    with open(log_path, mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_data) + "\n")
                    
                    # 更新最佳准确率
                    if max_teacher_accuracy < teacher_acc:
                        max_teacher_accuracy = teacher_acc
                        if args.output_dir:
                            utils.save_model(args=args, model=dual_model.teacher, 
                                           model_without_ddp=dual_model.teacher, 
                                           optimizer=teacher_optimizer,
                                           loss_scaler=loss_scaler, epoch="best_teacher")
                    
                    if max_student_accuracy < student_acc:
                        max_student_accuracy = student_acc
                        if args.output_dir:
                            utils.save_model(args=args, model=dual_model.student, 
                                           model_without_ddp=dual_model.student, 
                                           optimizer=student_optimizer,
                                           loss_scaler=loss_scaler, epoch="best_student")
                    
                    print('-----------------------------------------------------------------------')
                    print(f'Max Teacher accuracy: {max_teacher_accuracy:.2f}%')
                    print(f'Max Student accuracy: {max_student_accuracy:.2f}%')
                    print('-----------------------------------------------------------------------')
                    
                    continue
                else:
                    print(f"📝 开始训练教师模型 (Epoch {epoch}, 将保存为 {target_sequence}T.pth)")
            else:
                print(f"⏭️  当前epoch无需训练教师模型，跳过 Epoch {epoch}")
                
                # 进行评估
                test_stats = evaluate_dual(data_loader_val, dual_model, device)
                teacher_acc = test_stats['teacher_acc']
                student_acc = test_stats['student_acc']
                
                print(f"Epoch {epoch} ({current_mode}) - Teacher Acc: {teacher_acc:.1f}%, Student Acc: {student_acc:.1f}% (跳过训练)")
                
                log_data = {
                    'epoch': epoch,
                    'teacher_acc': teacher_acc,
                    'student_acc': student_acc,
                    'training_mode': current_mode,
                    'train_loss': 0.0,
                    'skipped_epoch': True
                }
                
                with open(log_path, mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_data) + "\n")
                
                continue
        
        # 处理学生训练模式
        elif current_mode in ['K1', 'K2']:
            teacher_loaded = False
            used_teacher_sequence = None
            
            # 如果有教师使用计划，按计划加载
            if epoch in teacher_usage_plan:
                target_sequence = teacher_usage_plan[epoch]
                success = load_teacher_weights_by_sequence(dual_model.teacher, args.dataset, target_sequence)
                if success:
                    print(f"🔄 按计划使用教师权重 {target_sequence}T.pth 进行蒸馏")
                    teacher_loaded = True
                    used_teacher_sequence = target_sequence
                else:
                    print(f"⚠️  无法加载计划的教师权重 {target_sequence}T.pth，使用当前教师模型")
            else:
                print(f"ℹ️  未指定教师模型，使用当前教师模型状态进行蒸馏")
        
        print(f"{'='*80}\n")
        
        # 执行训练
        train_stats, teacher_memory, student_memory, prob_list = train_one_epoch_dual(
            args, dual_model,
            data_loader_train, teacher_optimizer, student_optimizer, 
            amp_autocast, device, epoch,
            loss_scaler=loss_scaler,
            lr_schedule_values=lr_schedule_values,
            train_config=train_config,
            start_steps=epoch * num_training_steps_per_epoch,
            teacher_memory=teacher_memory,
            student_memory=student_memory,
            prob_list=prob_list,
            capability_probe=capability_probe
        )
        
        # 如果是教师训练且在训练计划中，保存权重
        saved_sequence = None
        if current_mode == 'T' and epoch in teacher_training_plan:
            target_sequence = teacher_training_plan[epoch]
            save_dir = Path(f"./pth0/{args.dataset}")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{target_sequence}T.pth"
            
            torch.save({
                'model_state_dict': dual_model.teacher.state_dict(),
                'sequence_num': target_sequence,
                'training_mode': current_mode,
                'center': dual_model.teacher.center.clone() if hasattr(dual_model.teacher, 'center') else None
            }, save_path)
            
            print(f"✅ 教师模型权重已保存: {save_path} (序列号: {target_sequence})")
            saved_sequence = target_sequence
        
        # 评估两个模型
        test_stats = evaluate_dual(data_loader_val, dual_model, device)
        
        teacher_acc = test_stats['teacher_acc']
        student_acc = test_stats['student_acc']
        
        print(f"\nEpoch {epoch} ({current_mode}) - Teacher Acc: {teacher_acc:.1f}%, Student Acc: {student_acc:.1f}%")
        
        # 记录评估结果
        eval_data = {
            'epoch': epoch,
            'teacher_acc': teacher_acc,
            'student_acc': student_acc,
            'training_mode': current_mode
        }
        if saved_sequence:
            eval_data['teacher_sequence'] = saved_sequence
        if current_mode in ['K1', 'K2'] and 'used_teacher_sequence' in locals():
            eval_data['used_teacher_sequence'] = used_teacher_sequence
        
        evaluation_accuracies.append(eval_data)
        
        # 在日志记录中添加自适应蒸馏信息
        log_data = {
            'epoch': epoch,
            'teacher_acc': teacher_acc,
            'student_acc': student_acc,
            'training_mode': current_mode,
            'train_loss': train_stats.get('loss', None),
        }
        
        # 添加教师序列信息
        if saved_sequence:
            log_data['teacher_sequence'] = saved_sequence
        if current_mode in ['K1', 'K2'] and 'used_teacher_sequence' in locals():
            log_data['used_teacher_sequence'] = used_teacher_sequence
        
        # 添加自适应参数统计到日志
        for key, value in train_stats.items():
            if ('loss' in key or 'percent' in key or 'avg' in key or 
                'adaptive_' in key or 'gap_' in key) and key not in log_data:
                log_data[key] = value
        
        # 保存到日志文件
        with open(log_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_data) + "\n")
        
        # 保存最佳模型
        if max_teacher_accuracy < teacher_acc:
            max_teacher_accuracy = teacher_acc
            if args.output_dir:
                utils.save_model(args=args, model=dual_model.teacher, 
                               model_without_ddp=dual_model.teacher, 
                               optimizer=teacher_optimizer,
                               loss_scaler=loss_scaler, epoch="best_teacher")
        
        if max_student_accuracy < student_acc:
            max_student_accuracy = student_acc
            if args.output_dir:
                utils.save_model(args=args, model=dual_model.student, 
                               model_without_ddp=dual_model.student, 
                               optimizer=student_optimizer,
                               loss_scaler=loss_scaler, epoch="best_student")
        
        print('-----------------------------------------------------------------------')
        print(f'Max Teacher accuracy: {max_teacher_accuracy:.2f}%')
        print(f'Max Student accuracy: {max_student_accuracy:.2f}%')
        print('-----------------------------------------------------------------------')
        
        # 保存详细日志
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters_teacher': n_parameters_teacher,
            'n_parameters_student': n_parameters_student
        }
        
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\nTraining time {total_time_str}')
    
    # 输出最终结果摘要
    print("\n" + "="*80)
    print("训练完成摘要:")
    print(f"  最佳教师模型准确率: {max_teacher_accuracy:.2f}%")
    print(f"  最佳学生模型准确率: {max_student_accuracy:.2f}%")
    print(f"  总训练时间: {total_time_str}")
    if args.training_strategy:
        print(f"  使用的训练策略: {','.join(args.training_strategy)}")
    if args.choose_teacher_model:
        print(f"  指定的教师序列: {args.choose_teacher_model}")
    print(f"  蒸馏logits类型: {','.join(map(str, choose_logits))}")
    print(f"  使用自适应蒸馏: {args.use_adaptive_distillation}")
    if args.use_adaptive_distillation:
        print(f"  温度范围: {temp_range}")
        print(f"  Alpha范围: {alpha_range}")
    
    # 显示保存的教师权重信息
    pth_dir = Path(f"./pth0/{args.dataset}")
    if pth_dir.exists():
        teacher_files = list(pth_dir.glob("*T.pth"))
        if teacher_files:
            print(f"  保存的教师权重文件: {len(teacher_files)} 个")
            sorted_files = sorted(teacher_files, key=lambda x: int(x.stem.replace('T', '')))
            for file in sorted_files:
                print(f"    {file}")
    print("="*80)

if __name__ == '__main__':
    opts = get_args()
    main(opts)