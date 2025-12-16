import argparse
import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# 导入现有模块
from utils.build_dataset import build_dataset
from utils.multi_model2 import DualModelManager
from utils.center14 import build_memory

class BatchDatasetDifficultyAnalyzer:
    """批量数据集难度分析器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # 定义要分析的数据集列表
        self.datasets_to_analyze = [
            'caltech101',
            'dtd',
            # 'eurosat', 
            'fgvc',
            'food101',
            # 'flowers',
            'pets',
            'cars',
            # 'sun397',
            'ucf101'
        ]
        
        # 存储所有数据集的分析结果
        self.all_results = {}
        
        # 定义要分析的核心指标 - 添加新指标
        self.core_metrics = [
            'accuracy',
            # 类间分离性指标
            'avg_inter_class_similarity',
            'max_inter_class_similarity',
            'min_inter_class_similarity',
            'avg_inter_class_euclidean_distance',
            'max_inter_class_euclidean_distance', 
            'min_inter_class_euclidean_distance',
            'avg_inter_class_cosine_similarity',  # 新增：平均类间余弦相似性
            'avg_inter_class_cosine_distance',    # 新增：平均类间余弦距离
            # 类内紧致性指标
            'avg_intra_class_similarity',
            'max_intra_class_similarity',
            'min_intra_class_similarity',
            'avg_intra_class_distance',
            'max_intra_class_distance',
            'min_intra_class_distance',
            'avg_intra_class_cosine_similarity',  # 新增：平均类内余弦相似性
            'avg_intra_class_euclidean_distance', # 新增：平均类内欧式距离
            # 综合分析指标
            'combined_separability',
            'intrinsic_dimension_95',
            'dimension_efficiency_95',
            'silhouette_score'
        ]
        
    def analyze_single_dataset(self, dataset_name):
        """分析单个数据集"""
        print(f"\n{'='*80}")
        print(f"开始分析数据集: {dataset_name}")
        print(f"{'='*80}")
        
        # 更新args中的数据集名称
        self.args.dataset = dataset_name
        
        try:
            # 创建双模型管理器
            dual_model = DualModelManager(self.args)
            dual_model.to(self.device)
            
            teacher = dual_model.teacher
            student = dual_model.student
            
            # 加载数据集
            dataset_train, len_original = build_dataset(is_train=True, args=self.args)
            dataset_val, _ = build_dataset(is_train=False, args=self.args)
            
            # 创建数据加载器
            batch_size = 32
            data_loader = torch.utils.data.DataLoader(
                dataset_train if self.args.use_train_set else dataset_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
            
            nb_classes = len(teacher.classnames)
            
            print(f"数据集信息:")
            print(f"  类别数: {nb_classes}")
            print(f"  样本数: {len(dataset_train if self.args.use_train_set else dataset_val)}")
            
            # 提取师生模型特征
            teacher_features, teacher_labels, _, teacher_acc = self.extract_features_and_labels(
                teacher, data_loader, "教师模型"
            )
            
            student_features, student_labels, _, student_acc = self.extract_features_and_labels(
                student, data_loader, "学生模型"
            )
            
            # 分析各项指标
            results = self.analyze_all_metrics(
                teacher_features, teacher_labels, teacher_acc,
                student_features, student_labels, student_acc,
                nb_classes
            )
            
            return results
            
        except Exception as e:
            print(f"分析数据集 {dataset_name} 时出错: {e}")
            return None
    
    @torch.no_grad()
    def extract_features_and_labels(self, model, data_loader, model_name):
        """提取特征和标签"""
        model.eval()
        all_features = []
        all_labels = []
        correct = 0
        total = 0
        
        print(f"正在提取 {model_name} 特征...")
        
        for batch in tqdm(data_loader):
            if isinstance(batch[0], list):
                images = batch[0][0].to(self.device)
            else:
                images = batch[0].to(self.device)
            
            labels = batch[1].to(self.device)
            
            # 提取特征
            features = model(images)
            
            # 计算预测
            logits = 100. * features @ model.get_classifier().t()
            predictions = torch.argmax(logits, dim=1)
            
            # 统计准确率
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # 收集数据
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
        
        # 合并所有数据
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        accuracy = correct / total * 100
        
        print(f"{model_name} 准确率: {accuracy:.2f}%")
        
        return all_features, all_labels, None, accuracy
    
 



    def analyze_all_metrics(self, teacher_features, teacher_labels, teacher_acc,
                        student_features, student_labels, student_acc, nb_classes):
        """分析所有核心指标"""
        
        results = {}
        
        # 1. 准确率
        results['accuracy'] = {
            'teacher': teacher_acc,
            'student': student_acc
        }
        
        # 2. 类间度量分析
        teacher_inter_metrics, _ = self.analyze_inter_class_metrics(teacher_features, teacher_labels, nb_classes)
        student_inter_metrics, _ = self.analyze_inter_class_metrics(student_features, student_labels, nb_classes)
        
        # 类间相似度指标（现有的）
        results['avg_inter_class_similarity'] = {
            'teacher': teacher_inter_metrics['avg_inter_class_similarity'],
            'student': student_inter_metrics['avg_inter_class_similarity']
        }
        
        results['max_inter_class_similarity'] = {
            'teacher': teacher_inter_metrics['max_inter_class_similarity'],
            'student': student_inter_metrics['max_inter_class_similarity']
        }
        
        results['min_inter_class_similarity'] = {
            'teacher': teacher_inter_metrics['min_inter_class_similarity'],
            'student': student_inter_metrics['min_inter_class_similarity']
        }
        
        # 类间距离指标（现有的）
        results['avg_inter_class_euclidean_distance'] = {
            'teacher': teacher_inter_metrics['avg_inter_class_euclidean_distance'],
            'student': student_inter_metrics['avg_inter_class_euclidean_distance']
        }
        
        results['max_inter_class_euclidean_distance'] = {
            'teacher': teacher_inter_metrics['max_inter_class_euclidean_distance'],
            'student': student_inter_metrics['max_inter_class_euclidean_distance']
        }
        
        results['min_inter_class_euclidean_distance'] = {
            'teacher': teacher_inter_metrics['min_inter_class_euclidean_distance'],
            'student': student_inter_metrics['min_inter_class_euclidean_distance']
        }
        
        # 新增：类间分离性指标
        results['avg_inter_class_cosine_similarity'] = {
            'teacher': teacher_inter_metrics['avg_inter_class_similarity'],  # 余弦相似性
            'student': student_inter_metrics['avg_inter_class_similarity']
        }
        
        results['avg_inter_class_cosine_distance'] = {
            'teacher': teacher_inter_metrics['avg_inter_class_cosine_distance'],  # 余弦距离
            'student': student_inter_metrics['avg_inter_class_cosine_distance']
        }
        
        # 3. 类内度量分析
        teacher_intra_metrics = self.analyze_intra_class_metrics(teacher_features, teacher_labels, nb_classes)
        student_intra_metrics = self.analyze_intra_class_metrics(student_features, student_labels, nb_classes)
        
        # 类内相似度指标（现有的）
        results['avg_intra_class_similarity'] = {
            'teacher': teacher_intra_metrics['avg_intra_class_similarity'],
            'student': student_intra_metrics['avg_intra_class_similarity']
        }
        
        results['max_intra_class_similarity'] = {
            'teacher': teacher_intra_metrics['max_intra_class_similarity'],
            'student': student_intra_metrics['max_intra_class_similarity']
        }
        
        results['min_intra_class_similarity'] = {
            'teacher': teacher_intra_metrics['min_intra_class_similarity'],
            'student': student_intra_metrics['min_intra_class_similarity']
        }
        
        # 类内距离指标（现有的）
        results['avg_intra_class_distance'] = {
            'teacher': teacher_intra_metrics['avg_intra_class_euclidean_distance'],
            'student': student_intra_metrics['avg_intra_class_euclidean_distance']
        }
        
        results['max_intra_class_distance'] = {
            'teacher': teacher_intra_metrics['max_intra_class_euclidean_distance'],
            'student': student_intra_metrics['max_intra_class_euclidean_distance']
        }
        
        results['min_intra_class_distance'] = {
            'teacher': teacher_intra_metrics['min_intra_class_euclidean_distance'],
            'student': student_intra_metrics['min_intra_class_euclidean_distance']
        }
        
        # 新增：类内紧致性指标
        results['avg_intra_class_cosine_similarity'] = {
            'teacher': teacher_intra_metrics['avg_intra_class_similarity'],  # 余弦相似性
            'student': student_intra_metrics['avg_intra_class_similarity']
        }
        
        results['avg_intra_class_euclidean_distance'] = {
            'teacher': teacher_intra_metrics['avg_intra_class_euclidean_distance'],  # 欧式距离
            'student': student_intra_metrics['avg_intra_class_euclidean_distance']
        }
        
        # 4. 可分离性分析
        teacher_separability = self.compute_separability_score(teacher_inter_metrics, teacher_intra_metrics)
        student_separability = self.compute_separability_score(student_inter_metrics, student_intra_metrics)
        
        results['combined_separability'] = {
            'teacher': teacher_separability['combined_separability'],
            'student': student_separability['combined_separability']
        }
        
        # 5. 特征复杂度分析
        teacher_complexity = self.analyze_feature_space_complexity(teacher_features, teacher_labels, nb_classes)
        student_complexity = self.analyze_feature_space_complexity(student_features, student_labels, nb_classes)
        
        results['intrinsic_dimension_95'] = {
            'teacher': teacher_complexity['intrinsic_dimension_95'],
            'student': student_complexity['intrinsic_dimension_95']
        }
        
        results['dimension_efficiency_95'] = {
            'teacher': teacher_complexity['dimension_efficiency_95'],
            'student': student_complexity['dimension_efficiency_95']
        }
        
        results['silhouette_score'] = {
            'teacher': teacher_complexity['silhouette_score'],
            'student': student_complexity['silhouette_score']
        }
        
        return results







    def analyze_inter_class_metrics(self, features, labels, nb_classes):
        """分析类间度量"""
        # 计算每个类别的中心特征
        class_centers = {}
        for class_id in range(nb_classes):
            mask = labels == class_id
            if mask.sum() > 0:
                class_features = features[mask]
                class_features = F.normalize(class_features, p=2, dim=1)
                class_center = class_features.mean(dim=0)
                class_center = F.normalize(class_center.unsqueeze(0), p=2, dim=1).squeeze(0)
                class_centers[class_id] = class_center
        
        available_classes = list(class_centers.keys())
        n_classes = len(available_classes)
        
        similarities = []
        euclidean_distances = []
        cosine_distances = []
        
        for i, class_i in enumerate(available_classes):
            for j, class_j in enumerate(available_classes):
                if i != j:
                    center_i = class_centers[class_i]
                    center_j = class_centers[class_j]
                    
                    # 计算余弦相似度
                    cosine_sim = F.cosine_similarity(
                        center_i.unsqueeze(0), center_j.unsqueeze(0)
                    ).item()
                    similarities.append(cosine_sim)
                    
                    # 计算余弦距离
                    cosine_dist = 1 - cosine_sim
                    cosine_distances.append(cosine_dist)
                    
                    # 计算欧氏距离
                    euclidean_dist = torch.dist(center_i, center_j, p=2).item()
                    euclidean_distances.append(euclidean_dist)
        
        inter_class_metrics = {
            # 类间相似度指标
            'avg_inter_class_similarity': np.mean(similarities) if similarities else 0.0,
            'max_inter_class_similarity': np.max(similarities) if similarities else 0.0,
            'min_inter_class_similarity': np.min(similarities) if similarities else 0.0,
            'std_inter_class_similarity': np.std(similarities) if similarities else 0.0,
            
            # 类间距离指标
            'avg_inter_class_euclidean_distance': np.mean(euclidean_distances) if euclidean_distances else 0.0,
            'max_inter_class_euclidean_distance': np.max(euclidean_distances) if euclidean_distances else 0.0,
            'min_inter_class_euclidean_distance': np.min(euclidean_distances) if euclidean_distances else 0.0,
            'std_inter_class_euclidean_distance': np.std(euclidean_distances) if euclidean_distances else 0.0,
            
            'avg_inter_class_cosine_distance': np.mean(cosine_distances) if cosine_distances else 0.0,
            'max_inter_class_cosine_distance': np.max(cosine_distances) if cosine_distances else 0.0,
            'min_inter_class_cosine_distance': np.min(cosine_distances) if cosine_distances else 0.0,
            'std_inter_class_cosine_distance': np.std(cosine_distances) if cosine_distances else 0.0,
        }
        
        return inter_class_metrics, class_centers

    # 修改 analyze_intra_class_metrics 函数
    def analyze_intra_class_metrics(self, features, labels, nb_classes):
        """分析类内度量"""
        features_norm = F.normalize(features, p=2, dim=1)
        
        intra_similarities = []
        intra_euclidean_distances = []
        intra_cosine_distances = []
        
        # 收集所有类别的类内度量值
        all_class_similarities = []
        all_class_euclidean_distances = []
        all_class_cosine_distances = []
        
        for class_id in range(nb_classes):
            mask = labels == class_id
            if mask.sum() > 1:
                class_features = features_norm[mask]
                n_samples = class_features.shape[0]
                
                # 限制计算数量避免过慢
                max_pairs = min(100, n_samples * (n_samples - 1) // 2)
                class_similarities = []
                class_euclidean_distances = []
                class_cosine_distances = []
                
                count = 0
                for i in range(n_samples):
                    for j in range(i+1, n_samples):
                        if count >= max_pairs:
                            break
                        
                        # 余弦相似度
                        cosine_sim = F.cosine_similarity(
                            class_features[i:i+1], class_features[j:j+1]
                        ).item()
                        class_similarities.append(cosine_sim)
                        
                        # 余弦距离
                        cosine_dist = 1 - cosine_sim
                        class_cosine_distances.append(cosine_dist)
                        
                        # 欧氏距离
                        euclidean_dist = torch.dist(
                            class_features[i], class_features[j], p=2
                        ).item()
                        class_euclidean_distances.append(euclidean_dist)
                        
                        count += 1
                    if count >= max_pairs:
                        break
                
                # 记录该类别的平均值
                if class_similarities:
                    class_avg_sim = np.mean(class_similarities)
                    class_avg_euc_dist = np.mean(class_euclidean_distances)
                    class_avg_cos_dist = np.mean(class_cosine_distances)
                    
                    intra_similarities.append(class_avg_sim)
                    intra_euclidean_distances.append(class_avg_euc_dist)
                    intra_cosine_distances.append(class_avg_cos_dist)
                    
                    # 收集所有样本对的值用于计算全局最大最小值
                    all_class_similarities.extend(class_similarities)
                    all_class_euclidean_distances.extend(class_euclidean_distances)
                    all_class_cosine_distances.extend(class_cosine_distances)
        
        intra_class_metrics = {
            # 类内相似度指标
            'avg_intra_class_similarity': np.mean(intra_similarities) if intra_similarities else 0.0,
            'max_intra_class_similarity': np.max(all_class_similarities) if all_class_similarities else 0.0,
            'min_intra_class_similarity': np.min(all_class_similarities) if all_class_similarities else 0.0,
            'std_intra_class_similarity': np.std(intra_similarities) if intra_similarities else 0.0,
            
            # 类内距离指标
            'avg_intra_class_euclidean_distance': np.mean(intra_euclidean_distances) if intra_euclidean_distances else 0.0,
            'max_intra_class_euclidean_distance': np.max(all_class_euclidean_distances) if all_class_euclidean_distances else 0.0,
            'min_intra_class_euclidean_distance': np.min(all_class_euclidean_distances) if all_class_euclidean_distances else 0.0,
            'std_intra_class_euclidean_distance': np.std(intra_euclidean_distances) if intra_euclidean_distances else 0.0,
            
            'avg_intra_class_cosine_distance': np.mean(intra_cosine_distances) if intra_cosine_distances else 0.0,
            'max_intra_class_cosine_distance': np.max(all_class_cosine_distances) if all_class_cosine_distances else 0.0,
            'min_intra_class_cosine_distance': np.min(all_class_cosine_distances) if all_class_cosine_distances else 0.0,
            'std_intra_class_cosine_distance': np.std(intra_cosine_distances) if intra_cosine_distances else 0.0,
        }
        
        return intra_class_metrics 
    
    
    
    def compute_separability_score(self, inter_metrics, intra_metrics):
        """计算类别可分离性分数"""
        inter_similarity = inter_metrics['avg_inter_class_similarity']
        intra_similarity = intra_metrics['avg_intra_class_similarity']
        
        similarity_separability = (intra_similarity - inter_similarity) / (intra_similarity + inter_similarity + 1e-8)
        
        separability_stats = {
            'combined_separability': similarity_separability,
        }
        
        return separability_stats
    
    def analyze_feature_space_complexity(self, features, labels, nb_classes):
        """分析特征空间复杂度"""
        features_norm = F.normalize(features, p=2, dim=1)
        
        # PCA分析
        try:
            # 为了加速，采样部分数据
            sample_size = min(2000, len(features_norm))
            indices = torch.randperm(len(features_norm))[:sample_size]
            sample_features = features_norm[indices].numpy()
            sample_labels = labels[indices].numpy()
            
            pca = PCA()
            pca.fit(sample_features)
            
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            intrinsic_dim_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            effective_rank = np.sum(pca.explained_variance_ratio_ ** 2) ** -1
            
        except Exception as e:
            print(f"PCA分析失败: {e}")
            intrinsic_dim_95 = features.shape[1]
            effective_rank = features.shape[1]
        
        # 轮廓系数
        try:
            from sklearn.metrics import silhouette_score
            sample_size = min(1000, len(features_norm))
            indices = torch.randperm(len(features_norm))[:sample_size]
            sample_features = features_norm[indices].numpy()
            sample_labels = labels[indices].numpy()
            
            silhouette_avg = silhouette_score(sample_features, sample_labels)
        except Exception as e:
            print(f"轮廓系数计算失败: {e}")
            silhouette_avg = 0.0
        
        complexity_stats = {
            'intrinsic_dimension_95': intrinsic_dim_95,
            'dimension_efficiency_95': intrinsic_dim_95 / features.shape[1],
            'silhouette_score': silhouette_avg,
        }
        
        return complexity_stats
    
    def calculate_metric_statistics(self, metric_name, all_results):
        """计算单个指标的统计信息"""
        teacher_values = []
        student_values = []
        differences = []
        ratios = []
        
        for dataset, results in all_results.items():
            if results and metric_name in results:
                teacher_val = results[metric_name]['teacher']
                student_val = results[metric_name]['student']
                
                teacher_values.append(teacher_val)
                student_values.append(student_val)
                
                diff = teacher_val - student_val
                differences.append(diff)
                
                # 计算比例 (差值相对于学生值的比例)
                if student_val != 0:
                    ratio = diff / student_val
                else:
                    ratio = 0 if diff == 0 else (1 if diff > 0 else -1)
                ratios.append(ratio)
        
        # 计算平均值
        avg_teacher = np.mean(teacher_values) if teacher_values else 0
        avg_student = np.mean(student_values) if student_values else 0
        avg_diff = np.mean(differences) if differences else 0
        avg_ratio = np.mean(ratios) if ratios else 0
        
        return {
            'teacher_values': teacher_values,
            'student_values': student_values,
            'differences': differences,
            'ratios': ratios,
            'avg_teacher': avg_teacher,
            'avg_student': avg_student,
            'avg_diff': avg_diff,
            'avg_ratio': avg_ratio
        }
    

    def generate_summary_table(self, all_results):
        """生成汇总表格 - 数据集为行，指标为列"""
        print("生成汇总表格...")
        
        # 准备表格数据 - 每个数据集一行
        table_data = []
        
        # 为每个数据集创建一行数据
        for dataset in self.datasets_to_analyze:
            if dataset in all_results and all_results[dataset]:
                row_data = {'Dataset': dataset}
                
                # 为每个指标添加三个要素
                for metric in self.core_metrics:
                    if metric in all_results[dataset]:
                        teacher_val = all_results[dataset][metric]['teacher']
                        student_val = all_results[dataset][metric]['student']
                        
                        # 计算差值和相对比例
                        diff = teacher_val - student_val
                        if student_val != 0:
                            ratio = diff / student_val
                        else:
                            ratio = 0 if diff == 0 else (1 if diff > 0 else -1)
                        
                        # 添加三个要素到行数据
                        row_data[f'{metric}_Teacher'] = teacher_val
                        row_data[f'{metric}_Student'] = student_val
                        row_data[f'{metric}_Ratio'] = ratio
                    else:
                        # 如果指标不存在，设为0
                        row_data[f'{metric}_Teacher'] = 0
                        row_data[f'{metric}_Student'] = 0
                        row_data[f'{metric}_Ratio'] = 0
                
                table_data.append(row_data)
        
        # 计算平均值行
        avg_row = {'Dataset': 'AVERAGE'}
        for metric in self.core_metrics:
            # 收集所有数据集的该指标值
            teacher_values = []
            student_values = []
            ratios = []
            
            for dataset in self.datasets_to_analyze:
                if dataset in all_results and all_results[dataset] and metric in all_results[dataset]:
                    teacher_val = all_results[dataset][metric]['teacher']
                    student_val = all_results[dataset][metric]['student']
                    
                    teacher_values.append(teacher_val)
                    student_values.append(student_val)
                    
                    diff = teacher_val - student_val
                    if student_val != 0:
                        ratio = diff / student_val
                    else:
                        ratio = 0 if diff == 0 else (1 if diff > 0 else -1)
                    ratios.append(ratio)
            
            # 计算平均值
            avg_row[f'{metric}_Teacher'] = np.mean(teacher_values) if teacher_values else 0
            avg_row[f'{metric}_Student'] = np.mean(student_values) if student_values else 0
            avg_row[f'{metric}_Ratio'] = np.mean(ratios) if ratios else 0
        
        # 添加平均值行到最后
        table_data.append(avg_row)
        
        # 创建DataFrame
        df = pd.DataFrame(table_data)
        
        # 重新排序列：Dataset列在最前面，然后按指标顺序排列三要素
        column_order = ['Dataset']
        for metric in self.core_metrics:
            column_order.extend([f'{metric}_Teacher', f'{metric}_Student', f'{metric}_Ratio'])
        
        df = df[column_order]
        
        return df



    def run_batch_analysis(self):
        """运行批量分析"""
        print("开始批量数据集难度分析...")
        print(f"待分析数据集: {self.datasets_to_analyze}")
        
        all_results = {}
        
        for dataset_name in self.datasets_to_analyze:
            print(f"\n正在分析数据集: {dataset_name}")
            results = self.analyze_single_dataset(dataset_name)
            all_results[dataset_name] = results
            
            # 清理GPU内存
            torch.cuda.empty_cache()
        
        # 生成汇总表格
        summary_df = self.generate_summary_table(all_results)
        
        # 保存结果
        self.save_batch_results(all_results, summary_df)
        
        return all_results, summary_df
    
    # 修改 save_batch_results 函数，添加更清晰的表格格式

    def save_batch_results(self, all_results, summary_df):
        """保存批量分析结果 - 增强版"""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存基础CSV表格
        summary_file = output_dir / "dataset_difficulty_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"汇总表格已保存到: {summary_file}")
        
        # 2. 保存格式化的Excel表格，确保正确的行列方向
        formatted_summary_file = output_dir / "dataset_difficulty_summary_formatted.xlsx"
        
        with pd.ExcelWriter(formatted_summary_file, engine='openpyxl') as writer:
            # 主汇总表 - 数据集为行，指标为列
            summary_df.to_excel(writer, sheet_name='Main_Summary', index=False)
            
            # 获取工作表并设置格式
            workbook = writer.book
            worksheet = workbook['Main_Summary']
            
            # 冻结首行和首列
            worksheet.freeze_panes = 'B2'
            
            # 设置列宽
            for col in worksheet.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 20)
                worksheet.column_dimensions[column].width = adjusted_width
            
            # 为每个指标创建单独的工作表
            for metric in self.core_metrics:
                metric_columns = ['Dataset', f'{metric}_Teacher', f'{metric}_Student', f'{metric}_Ratio']
                metric_data = summary_df[metric_columns].copy()
                
                # 重命名列以便于阅读
                metric_data.columns = ['Dataset', 'Teacher_Value', 'Student_Value', 'Teacher_Student_Ratio']
                metric_data.to_excel(writer, sheet_name=metric[:30], index=False)
            
            # 创建对比分析表（相对于平均值的偏差）
            self.create_deviation_analysis(writer, summary_df)
            
            # 创建转置表格（如果您需要指标为行，数据集为列的版本）
            self.create_transposed_summary(writer, summary_df)
        
        print(f"格式化汇总表格已保存到: {formatted_summary_file}")
        
        # 3. 保存详细的JSON结果
        json_file = output_dir / "detailed_results.json"
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(all_results)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"详细结果已保存到: {json_file}")
        
        # 4. 生成可读性报告
        self.generate_readable_report(output_dir, summary_df, all_results)
        
        # 5. 打印汇总统计
        self.print_summary_statistics(summary_df)

    def create_transposed_summary(self, writer, summary_df):
        """创建转置的汇总表格 - 指标为行，数据集为列"""
        
        # 准备转置数据
        transposed_data = []
        
        # 为每个指标的每个要素创建一行
        for metric in self.core_metrics:
            for element in ['Teacher', 'Student', 'Ratio']:
                row_data = {'Metric_Element': f'{metric}_{element}'}
                
                # 为每个数据集添加数据
                for _, dataset_row in summary_df.iterrows():
                    dataset_name = dataset_row['Dataset']
                    value = dataset_row[f'{metric}_{element}']
                    row_data[dataset_name] = value
                
                transposed_data.append(row_data)
        
        # 创建转置DataFrame
        transposed_df = pd.DataFrame(transposed_data)
        
        # 调整列顺序，将AVERAGE列放到最后
        cols = list(transposed_df.columns)
        if 'AVERAGE' in cols:
            cols.remove('AVERAGE')
            cols.append('AVERAGE')
        transposed_df = transposed_df[cols]
        
        # 保存转置表格
        transposed_df.to_excel(writer, sheet_name='Transposed_Summary', index=False)



    def create_deviation_analysis(self, writer, summary_df):
        """创建相对于平均值的偏差分析表"""
        
        # 获取平均值行
        avg_row = summary_df[summary_df['Dataset'] == 'AVERAGE'].iloc[0]
        
        # 创建偏差分析数据
        deviation_data = []
        
        for _, row in summary_df.iterrows():
            if row['Dataset'] != 'AVERAGE':
                dev_row = {'Dataset': row['Dataset']}
                
                for metric in self.core_metrics:
                    # 计算相对于平均值的偏差
                    teacher_dev = row[f'{metric}_Teacher'] - avg_row[f'{metric}_Teacher']
                    student_dev = row[f'{metric}_Student'] - avg_row[f'{metric}_Student']
                    ratio_dev = row[f'{metric}_Ratio'] - avg_row[f'{metric}_Ratio']
                    
                    dev_row[f'{metric}_Teacher_Dev'] = teacher_dev
                    dev_row[f'{metric}_Student_Dev'] = student_dev
                    dev_row[f'{metric}_Ratio_Dev'] = ratio_dev
                    
                    # 标记是否高于/低于平均值
                    dev_row[f'{metric}_Teacher_Status'] = 'Above' if teacher_dev > 0 else 'Below'
                    dev_row[f'{metric}_Student_Status'] = 'Above' if student_dev > 0 else 'Below'
                    dev_row[f'{metric}_Ratio_Status'] = 'Above' if ratio_dev > 0 else 'Below'
                
                deviation_data.append(dev_row)
        
        deviation_df = pd.DataFrame(deviation_data)
        deviation_df.to_excel(writer, sheet_name='Deviation_Analysis', index=False)



    def generate_readable_report(self, output_dir, summary_df, all_results):
        """生成可读性报告"""
        report_file = output_dir / "analysis_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("数据集难度分析报告\n")
            f.write("="*80 + "\n\n")
            
            # 基本信息
            f.write("1. 分析概览\n")
            f.write("-"*40 + "\n")
            f.write(f"分析数据集数量: {len(self.datasets_to_analyze)}\n")
            f.write(f"分析指标数量: {len(self.core_metrics)}\n")
            f.write(f"数据集列表: {', '.join(self.datasets_to_analyze)}\n\n")
            
            # 平均值统计
            avg_row = summary_df[summary_df['Dataset'] == 'AVERAGE'].iloc[0]
            f.write("2. 整体平均值\n")
            f.write("-"*40 + "\n")
            
            # 分类展示平均值
            f.write("2.1 准确率指标:\n")
            f.write(f"  准确率: Teacher={avg_row['accuracy_Teacher']:.4f}, Student={avg_row['accuracy_Student']:.4f}, Ratio={avg_row['accuracy_Ratio']:.4f}\n\n")
            
            f.write("2.2 类间分离性指标:\n")
            f.write(f"  平均类间余弦相似性: Teacher={avg_row['avg_inter_class_cosine_similarity_Teacher']:.4f}, Student={avg_row['avg_inter_class_cosine_similarity_Student']:.4f}, Ratio={avg_row['avg_inter_class_cosine_similarity_Ratio']:.4f}\n")
            f.write(f"  平均类间欧式距离: Teacher={avg_row['avg_inter_class_euclidean_distance_Teacher']:.4f}, Student={avg_row['avg_inter_class_euclidean_distance_Student']:.4f}, Ratio={avg_row['avg_inter_class_euclidean_distance_Ratio']:.4f}\n")
            f.write(f"  平均类间余弦距离: Teacher={avg_row['avg_inter_class_cosine_distance_Teacher']:.4f}, Student={avg_row['avg_inter_class_cosine_distance_Student']:.4f}, Ratio={avg_row['avg_inter_class_cosine_distance_Ratio']:.4f}\n\n")
            
            f.write("2.3 类内紧致性指标:\n")
            f.write(f"  平均类内余弦相似性: Teacher={avg_row['avg_intra_class_cosine_similarity_Teacher']:.4f}, Student={avg_row['avg_intra_class_cosine_similarity_Student']:.4f}, Ratio={avg_row['avg_intra_class_cosine_similarity_Ratio']:.4f}\n")
            f.write(f"  平均类内欧式距离: Teacher={avg_row['avg_intra_class_euclidean_distance_Teacher']:.4f}, Student={avg_row['avg_intra_class_euclidean_distance_Student']:.4f}, Ratio={avg_row['avg_intra_class_euclidean_distance_Ratio']:.4f}\n\n")
            
            f.write("2.4 综合分析指标:\n")
            f.write(f"  综合可分离性: Teacher={avg_row['combined_separability_Teacher']:.4f}, Student={avg_row['combined_separability_Student']:.4f}, Ratio={avg_row['combined_separability_Ratio']:.4f}\n")
            f.write(f"  轮廓系数: Teacher={avg_row['silhouette_score_Teacher']:.4f}, Student={avg_row['silhouette_score_Student']:.4f}, Ratio={avg_row['silhouette_score_Ratio']:.4f}\n\n")
            
            # 数据集排名分析
            f.write("3. 数据集难度排名分析\n")
            f.write("-"*40 + "\n")
            
            # 基于准确率差值的排名
            accuracy_data = []
            for _, row in summary_df.iterrows():
                if row['Dataset'] != 'AVERAGE':
                    accuracy_gap = row['accuracy_Teacher'] - row['accuracy_Student']
                    accuracy_data.append((row['Dataset'], accuracy_gap))
            
            accuracy_data.sort(key=lambda x: x[1], reverse=True)
            
            f.write("3.1 基于准确率差距的排名 (差距越大，蒸馏难度越高):\n")
            for i, (dataset, gap) in enumerate(accuracy_data, 1):
                f.write(f"  {i}. {dataset}: {gap:.2f}%\n")
            
            # 基于类间分离性的排名
            f.write("\n3.2 基于类间分离性的排名 (相似度越高，区分越困难):\n")
            separation_data = []
            for _, row in summary_df.iterrows():
                if row['Dataset'] != 'AVERAGE':
                    avg_separation = (row['avg_inter_class_cosine_similarity_Teacher'] + row['avg_inter_class_cosine_similarity_Student']) / 2
                    separation_data.append((row['Dataset'], avg_separation))
            
            separation_data.sort(key=lambda x: x[1], reverse=True)
            for i, (dataset, sep) in enumerate(separation_data, 1):
                f.write(f"  {i}. {dataset}: {sep:.4f}\n")
            
            # 基于类内紧致性的排名
            f.write("\n3.3 基于类内紧致性的排名 (相似度越低，类内越分散):\n")
            cohesion_data = []
            for _, row in summary_df.iterrows():
                if row['Dataset'] != 'AVERAGE':
                    avg_cohesion = (row['avg_intra_class_cosine_similarity_Teacher'] + row['avg_intra_class_cosine_similarity_Student']) / 2
                    cohesion_data.append((row['Dataset'], avg_cohesion))
            
            cohesion_data.sort(key=lambda x: x[1])  # 升序，相似度越低排名越高
            for i, (dataset, coh) in enumerate(cohesion_data, 1):
                f.write(f"  {i}. {dataset}: {coh:.4f}\n")
            
            f.write("\n")
            
            # 异常值分析
            f.write("4. 关键指标异常值分析\n")
            f.write("-"*40 + "\n")
            
            # 重点关注新增的4个指标
            key_metrics = [
                'avg_inter_class_cosine_similarity',
                'avg_inter_class_euclidean_distance', 
                'avg_intra_class_cosine_similarity',
                'avg_intra_class_euclidean_distance'
            ]
            
            for metric in key_metrics:
                metric_values = []
                for _, row in summary_df.iterrows():
                    if row['Dataset'] != 'AVERAGE':
                        metric_values.append((row['Dataset'], row[f'{metric}_Ratio']))
                
                metric_values.sort(key=lambda x: x[1])
                
                f.write(f"{metric} 师生比例排名:\n")
                f.write(f"  最低: {metric_values[0][0]} ({metric_values[0][1]:.4f})\n")
                f.write(f"  最高: {metric_values[-1][0]} ({metric_values[-1][1]:.4f})\n\n")
            
            f.write("5. 指标解释\n")
            f.write("-"*40 + "\n")
            f.write("类间分离性指标 (数值越大，类别区分度越好):\n")
            f.write("  - 平均类间余弦相似性: 类别中心间的平均相似度，越小越好\n")
            f.write("  - 平均类间欧式距离: 类别中心间的平均距离，越大越好\n\n")
            f.write("类内紧致性指标 (数值反映类内聚合度):\n")
            f.write("  - 平均类内余弦相似性: 同类样本间的平均相似度，越大越好\n")
            f.write("  - 平均类内欧式距离: 同类样本间的平均距离，越小越好\n\n")
            f.write("师生比例 (Ratio = (Teacher - Student) / Student):\n")
            f.write("  - 正值: 教师模型表现更好\n")
            f.write("  - 负值: 学生模型表现更好\n")
            f.write("  - 接近0: 师生模型表现相似\n")
        
        print(f"可读性报告已保存到: {report_file}")




    def print_summary_statistics(self, summary_df):
        """打印汇总统计信息"""
        print("\n" + "="*80)
        print("批量分析完成! 汇总统计:")
        print("="*80)
        
        print(f"\n表格结构说明:")
        print(f"  行数: {len(summary_df)} (包含 {len(self.datasets_to_analyze)} 个数据集 + 1 个平均值行)")
        print(f"  列数: {len(summary_df.columns)} (1个数据集列 + {len(self.core_metrics)} × 3要素)")
        print(f"  表格布局: 数据集为行，指标三要素为列")
        
        avg_row = summary_df[summary_df['Dataset'] == 'AVERAGE'].iloc[0]
        
        print(f"\n核心指标汇总:")
        print(f"=" * 60)
        
        # 分类展示指标
        print(f"\n【准确率指标】")
        print(f"  accuracy: T={avg_row['accuracy_Teacher']:.4f}, S={avg_row['accuracy_Student']:.4f}, R={avg_row['accuracy_Ratio']:.4f}")
        
        print(f"\n【类间分离性指标 - 相似度】")
        print(f"  平均类间余弦相似性: T={avg_row['avg_inter_class_cosine_similarity_Teacher']:.4f}, S={avg_row['avg_inter_class_cosine_similarity_Student']:.4f}, R={avg_row['avg_inter_class_cosine_similarity_Ratio']:.4f}")
        print(f"  最大类间相似性: T={avg_row['max_inter_class_similarity_Teacher']:.4f}, S={avg_row['max_inter_class_similarity_Student']:.4f}, R={avg_row['max_inter_class_similarity_Ratio']:.4f}")
        print(f"  最小类间相似性: T={avg_row['min_inter_class_similarity_Teacher']:.4f}, S={avg_row['min_inter_class_similarity_Student']:.4f}, R={avg_row['min_inter_class_similarity_Ratio']:.4f}")
        
        print(f"\n【类间分离性指标 - 距离】")
        print(f"  平均类间欧式距离: T={avg_row['avg_inter_class_euclidean_distance_Teacher']:.4f}, S={avg_row['avg_inter_class_euclidean_distance_Student']:.4f}, R={avg_row['avg_inter_class_euclidean_distance_Ratio']:.4f}")
        print(f"  平均类间余弦距离: T={avg_row['avg_inter_class_cosine_distance_Teacher']:.4f}, S={avg_row['avg_inter_class_cosine_distance_Student']:.4f}, R={avg_row['avg_inter_class_cosine_distance_Ratio']:.4f}")
        
        print(f"\n【类内紧致性指标 - 相似度】")
        print(f"  平均类内余弦相似性: T={avg_row['avg_intra_class_cosine_similarity_Teacher']:.4f}, S={avg_row['avg_intra_class_cosine_similarity_Student']:.4f}, R={avg_row['avg_intra_class_cosine_similarity_Ratio']:.4f}")
        print(f"  最大类内相似性: T={avg_row['max_intra_class_similarity_Teacher']:.4f}, S={avg_row['max_intra_class_similarity_Student']:.4f}, R={avg_row['max_intra_class_similarity_Ratio']:.4f}")
        print(f"  最小类内相似性: T={avg_row['min_intra_class_similarity_Teacher']:.4f}, S={avg_row['min_intra_class_similarity_Student']:.4f}, R={avg_row['min_intra_class_similarity_Ratio']:.4f}")
        
        print(f"\n【类内紧致性指标 - 距离】")
        print(f"  平均类内欧式距离: T={avg_row['avg_intra_class_euclidean_distance_Teacher']:.4f}, S={avg_row['avg_intra_class_euclidean_distance_Student']:.4f}, R={avg_row['avg_intra_class_euclidean_distance_Ratio']:.4f}")
        print(f"  最大类内距离: T={avg_row['max_intra_class_distance_Teacher']:.4f}, S={avg_row['max_intra_class_distance_Student']:.4f}, R={avg_row['max_intra_class_distance_Ratio']:.4f}")
        print(f"  最小类内距离: T={avg_row['min_intra_class_distance_Teacher']:.4f}, S={avg_row['min_intra_class_distance_Student']:.4f}, R={avg_row['min_intra_class_distance_Ratio']:.4f}")
        
        print(f"\n【综合分析指标】")
        print(f"  综合可分离性: T={avg_row['combined_separability_Teacher']:.4f}, S={avg_row['combined_separability_Student']:.4f}, R={avg_row['combined_separability_Ratio']:.4f}")
        print(f"  内在维度95%: T={avg_row['intrinsic_dimension_95_Teacher']:.1f}, S={avg_row['intrinsic_dimension_95_Student']:.1f}, R={avg_row['intrinsic_dimension_95_Ratio']:.4f}")
        print(f"  维度效率95%: T={avg_row['dimension_efficiency_95_Teacher']:.4f}, S={avg_row['dimension_efficiency_95_Student']:.4f}, R={avg_row['dimension_efficiency_95_Ratio']:.4f}")
        print(f"  轮廓系数: T={avg_row['silhouette_score_Teacher']:.4f}, S={avg_row['silhouette_score_Student']:.4f}, R={avg_row['silhouette_score_Ratio']:.4f}")
        
        # 显示表格预览
        print(f"\n表格预览 (前3行，前4列):")
        print("-"*80)
        display_df = summary_df.head(3)
        # 只显示前几列避免过宽
        preview_columns = ['Dataset'] + [f'{self.core_metrics[0]}_Teacher', f'{self.core_metrics[0]}_Student', f'{self.core_metrics[0]}_Ratio']
        if len(preview_columns) <= len(display_df.columns):
            print(display_df[preview_columns].to_string(index=False))
        else:
            print(display_df[['Dataset']].to_string(index=False))
        
        print(f"\n指标说明:")
        print(f"  T = Teacher (教师模型值)")
        print(f"  S = Student (学生模型值)")  
        print(f"  R = Ratio (差值相对学生比值 = (T-S)/S)")
        print(f"\n理想情况:")
        print(f"  类间分离性: 距离大、相似度小 → 易区分")
        print(f"  类内紧致性: 距离小、相似度大 → 聚类好")
        print(f"  师生差异: R接近0 → 蒸馏效果好")




def get_args():
    parser = argparse.ArgumentParser('Batch Dataset Difficulty Analyzer', add_help=False)
    
    # Dataset parameters
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', default=4, type=int)
    
    # CLIP parameters
    parser.add_argument("--template", default='templates.json', type=str)
    parser.add_argument("--classname", default='classes.json', type=str)
    parser.add_argument('--image_mean', default=(0.48145466, 0.4578275, 0.40821073))
    parser.add_argument('--image_std', default=(0.26862954, 0.26130258, 0.27577711))
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    
    # Analysis parameters
    parser.add_argument('--use_train_set', action='store_true', default=True, 
                       help='Use training set for analysis (default: True)')
    parser.add_argument('--output_dir', default='batch_dataset_analysis', 
                       help='Output directory for analysis results')
    
    # Data augmentation parameters (required by build_dataset)
    parser.add_argument('--train_crop_min', default=0.3, type=float)
    parser.add_argument('--color_jitter', type=float, default=0, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME')
    parser.add_argument('--train_interpolation', type=str, default='bicubic')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    print("批量数据集难度分析器")
    print("="*80)
    print(f"设备: {args.device}")
    print(f"使用训练集: {args.use_train_set}")
    print("="*80)
    
    # 创建批量分析器
    analyzer = BatchDatasetDifficultyAnalyzer(args)
    
    # 运行批量分析
    all_results, summary_df = analyzer.run_batch_analysis()
    
    print("\n" + "="*80)
    print("批量分析完成!")
    print("="*80)


if __name__ == '__main__':
    main()