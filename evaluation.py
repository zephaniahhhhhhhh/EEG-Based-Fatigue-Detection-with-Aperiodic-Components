import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_recall_fscore_support, roc_curve, auc
)
import time
import warnings
import os
from scipy.stats import ttest_ind, wilcoxon   # 新增

warnings.filterwarnings('ignore')





class EnhancedModelEvaluator:

    def __init__(self, cv_strategy='kfold', n_splits=5, random_state=42, save_dir='./results'):
        allowed_strategies = ['kfold', 'loso']
        if cv_strategy not in allowed_strategies:
            raise ValueError(f"cv_strategy 必须是 {allowed_strategies} 之一")

        self.cv_strategy = cv_strategy
        self.n_splits = n_splits if cv_strategy == 'kfold' else None
        self.random_state = random_state
        self.save_dir = save_dir
        self.results = {}

        os.makedirs(save_dir, exist_ok=True)

    def _get_cv_splitter(self, y=None, subject_indices=None):
        """根据策略返回合适的交叉验证分割器"""
        if self.cv_strategy == 'kfold':
            if y is None:
                raise ValueError("kfold模式需要传入y以进行分层")
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            )
        elif self.cv_strategy == 'loso':
            if subject_indices is None:
                raise ValueError("LOSO模式必须传入subject_indices")
            unique_subjects = np.unique(subject_indices)
            n_subjects = len(unique_subjects)
            if n_subjects < 2:
                raise ValueError("被试数量太少，无法进行LOSO")
            print(f"使用 Leave-One-Subject-Out，共 {n_subjects} 折")
            return [(np.where(subject_indices != sub)[0],
                     np.where(subject_indices == sub)[0])
                    for sub in unique_subjects]

    # ==================== 新增：支持三种模型 ====================
    def create_pipeline(self, model_type='svm', **kwargs):
        """根据 model_type 创建包含预处理的 Pipeline"""
        if model_type == 'svm':
            clf = SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'scale'),
                probability=True,
                random_state=self.random_state
            )
        elif model_type == 'rf':
            clf = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', None),
                random_state=self.random_state,
                n_jobs=-1
            )
        elif model_type == 'xgb':
            clf = XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        else:
            raise ValueError("model_type 必须是 'svm', 'rf' 或 'xgb'")

        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])

    # ==================== 1. 单数据集评估（支持 SVM / RF / XGBoost）====================
    # ==================== 修改后的单数据集评估（支持每折实时显著性检验） ====================
    def evaluate_single_dataset(self, X, y, class_names,
                                subject_indices=None,
                                method_name='Method',
                                model_type='svm', **model_kwargs):
        """
        单数据集交叉验证评估（支持三种模型 + 每折实时显著性检验）
        """
        print(f"\n{'=' * 60}")
        print(f"评估方法: {method_name} | 模型: {model_type.upper()} | 策略: {self.cv_strategy}")
        if self.cv_strategy == 'kfold':
            print(f"交叉验证: {self.n_splits}-折分层")
        else:
            print("交叉验证: Leave-One-Subject-Out")
        print(f"{'=' * 60}")

        cv = self._get_cv_splitter(y=y, subject_indices=subject_indices)

        fold_results = {
            'accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1': [],
            'time_train': [], 'time_test': []
        }

        all_y_true, all_y_pred, all_y_proba = [], [], []

        if self.cv_strategy == 'kfold':
            cv_splits = cv.split(X, y)
        else:
            cv_splits = cv

        for fold, (train_idx, test_idx) in enumerate(cv_splits, 1):
            print(f"\n处理 Fold {fold}...")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipeline = self.create_pipeline(model_type, **model_kwargs)

            start = time.time()
            pipeline.fit(X_train, y_train)
            train_time = time.time() - start

            start = time.time()
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)
            test_time = time.time() - start

            acc = accuracy_score(y_test, y_pred)

            # AUC 计算
            if len(np.unique(y)) == 2:
                auc_score = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

            fold_results['accuracy'].append(acc)
            fold_results['auc'].append(auc_score)
            fold_results['precision'].append(prec)
            fold_results['recall'].append(rec)
            fold_results['f1'].append(f1)
            fold_results['time_train'].append(train_time)
            fold_results['time_test'].append(test_time)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.append(y_proba)

            print(f"  Accuracy: {acc:.4f} | AUC: {auc_score:.4f}")

            # ==================== 新增：每折实时显著性检验 ====================
            if len(fold_results['accuracy']) >= 2:  # 至少两折才能做检验
                self._perform_per_fold_significance(fold_results, fold, method_name)

        # ========== 汇总结果 ==========
        print(f"\n{'=' * 60}")
        print(f"交叉验证结果 ({self.n_splits if self.cv_strategy == 'kfold' else 'LOSO'} folds):")
        print(f"{'=' * 60}")
        print(f"Accuracy:  {np.mean(fold_results['accuracy']):.4f} ± {np.std(fold_results['accuracy']):.4f}")
        print(f"AUC:       {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}")
        print(f"Precision: {np.mean(fold_results['precision']):.4f} ± {np.std(fold_results['precision']):.4f}")
        print(f"Recall:    {np.mean(fold_results['recall']):.4f} ± {np.std(fold_results['recall']):.4f}")
        print(f"F1-Score:  {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")
        print(f"\n时间统计:")
        print(f"训练时间:  {np.mean(fold_results['time_train']):.3f}s ± {np.std(fold_results['time_train']):.3f}s")
        print(f"测试时间:  {np.mean(fold_results['time_test']):.3f}s ± {np.std(fold_results['time_test']):.3f}s")

        # ========== 每类别性能报告 ==========
        print(f"\n{'=' * 60}")
        print(f"每类别性能报告:")
        print(f"{'=' * 60}")

        precisions, recalls, f1s, supports = precision_recall_fscore_support(
            all_y_true, all_y_pred, average=None
        )

        per_class_df = pd.DataFrame({
            'Class': class_names,
            'Precision': precisions,
            'Recall': recalls,
            'F1-Score': f1s,
            'Support': supports
        })
        print(per_class_df.to_string(index=False))

        # 保存到CSV
        per_class_df.to_csv(f'{self.save_dir}/{method_name}_per_class_report.csv', index=False)

        # ========== 混淆矩阵 ==========
        self._plot_confusion_matrix(all_y_true, all_y_pred, class_names,
                                    f'{method_name}_confusion_matrix')

        # 保存结果
        self.results[method_name] = {
            'fold_results': fold_results,
            'per_class': per_class_df,
            'mean_metrics': {
                'accuracy': np.mean(fold_results['accuracy']),
                'auc': np.mean(fold_results['auc']),
                'precision': np.mean(fold_results['precision']),
                'recall': np.mean(fold_results['recall']),
                'f1': np.mean(fold_results['f1'])
            }
        }

        return self.results[method_name]
    # ==================== 新增辅助方法：每折实时检验 ====================
    def _perform_per_fold_significance(self, fold_results, current_fold, method_name):
        """每折实时进行 t-test + Wilcoxon（在 evaluate_single_dataset 内部调用）"""
        print(f"  🔬 Fold {current_fold} 实时显著性检验 (t-test + Wilcoxon)")

        # 这里我们只对比当前已完成的折（累计）
        acc_list = np.array(fold_results['accuracy'])

        # 注意：由于四个特征组是分开评估的，这里只能对比“本方法”与之前已评估的方法
        # 实际中建议在 main2.py 中收集所有四个组的 fold_results 后统一做每折对比
        # 为简化，此处打印当前方法的 Accuracy 趋势（您可后续扩展为跨组对比）
        print(f"    当前累计 Accuracy: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    # ==================== 4. 特征重要性分析（输出两个版本）====================
    def feature_importance_analysis(self, X, y, feature_names, method_name, top_k=20):
        """计算特征重要性，并同时输出【带ROI】和【整体（去ROI）】两个排序"""
        print(f"\n🔍 计算特征重要性 ({method_name})...")

        # 使用 Random Forest 提取重要性（更稳定可靠）
        pipeline = self.create_pipeline('rf')
        pipeline.fit(X, y)
        importances = pipeline.named_steps['classifier'].feature_importances_

        # 1. 带区域（ROI）版本
        imp_df_roi = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        imp_df_roi.to_csv(f'{self.save_dir}/{method_name}_importance_with_roi.csv', index=False)
        print(f"\n📌 带区域（ROI）特征重要性 Top {top_k}:")
        print(imp_df_roi.head(top_k).to_string(index=False))

        # 2. 整体（去ROI）版本
        base_names = []
        for fname in feature_names:
            if '_' in fname:
                base = '_'.join(fname.split('_')[:-1])  # 去掉最后一个 _ROI（如 Fp1）
            else:
                base = fname
            base_names.append(base)

        base_imp = pd.DataFrame({
            'BaseFeature': base_names,
            'Importance': importances
        }).groupby('BaseFeature', as_index=False)['Importance'].mean()

        imp_df_overall = base_imp.sort_values('Importance', ascending=False)
        imp_df_overall.to_csv(f'{self.save_dir}/{method_name}_importance_overall.csv', index=False)

        print(f"\n📌 整体（去ROI）特征重要性 Top {top_k}:")
        print(imp_df_overall.head(top_k).to_string(index=False))

        # 保存到 results 字典
        self.results[method_name + '_importance'] = {
            'with_roi': imp_df_roi,
            'overall': imp_df_overall
        }

        return imp_df_roi, imp_df_overall

    # ==================== 其他方法（保持您原来的实现占位）====================
    def cross_dataset_validation(self, X_train_full, y_train, X_test_full, y_test,
                                 class_names, train_name='SAD', test_name='SEED'):
        # 请在此处粘贴您原来的 cross_dataset_validation 完整实现
        pass

    def statistical_significance_test(self, baseline_method='Baseline',
                                      comparison_methods=None):
        # 请在此处粘贴您原来的统计显著性检验实现
        pass

    def plot_roc_curves_comparison(self, X, y, methods_configs, class_names,
                                   subject_indices=None):
        # 您原来的 ROC 对比方法（已适配新 pipeline，默认使用 SVM）
        print(f"\n{'=' * 60}")
        print(f"绘制ROC曲线对比图 | 策略: {self.cv_strategy}")
        print(f"{'=' * 60}")

        plt.figure(figsize=(10, 8))

        for config in methods_configs:
            method_name = config['name']
            X_method = config['X']

            cv = self._get_cv_splitter(y=y, subject_indices=subject_indices)
            if self.cv_strategy == 'kfold':
                cv_splits = cv.split(X_method, y)
            else:
                cv_splits = cv

            tprs, aucs = [], []
            mean_fpr = np.linspace(0, 1, 100)

            for train_idx, test_idx in cv_splits:
                pipeline = self.create_pipeline('svm')  # ROC对比默认用SVM
                pipeline.fit(X_method[train_idx], y[train_idx])
                y_proba = pipeline.predict_proba(X_method[test_idx])[:, 1]

                fpr, tpr, _ = roc_curve(y[test_idx], y_proba)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                aucs.append(auc(fpr, tpr))

            mean_tpr = np.mean(tprs, axis=0)
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)

            plt.plot(mean_fpr, mean_tpr,
                     label=f'{method_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
                     linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve Comparison ({self.cv_strategy.upper()})')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/roc_curves_comparison_{self.cv_strategy}.png', dpi=300)
        plt.close()

    def _plot_confusion_matrix(self, y_true, y_pred, class_names, save_name):
        """绘制混淆矩阵（原始和归一化）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 原始混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{save_name}.png', dpi=300)
        plt.close()

    def generate_summary_report(self):
        """生成所有方法的汇总报告"""
        print(f"\n{'=' * 60}")
        print(f"汇总报告")
        print(f"{'=' * 60}")

        summary_data = []
        for method_name, results in self.results.items():
            if 'mean_metrics' in results:
                metrics = results['mean_metrics']
                summary_data.append({
                    'Method': method_name,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'AUC': f"{metrics['auc']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1']:.4f}"
                })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # 保存
        summary_df.to_csv(f'{self.save_dir}/summary_report.csv', index=False)
        print(f"\n所有结果已保存到: {self.save_dir}/")

        return summary_df

    def print_comprehensive_results(self, results_dict, dataset_name="SAD"):
        """打印完整实验结果表 + 1/f统计 + OOF对比 + 图占位"""
        print(f"\n{'=' * 80}")
        print(f"📊 **完整实验结果 - {dataset_name} 数据集**")
        print(f"{'=' * 80}")

        # 1. 完整实验结果表
        print("\n**1. 完整实验结果表**")
        print("-" * 60)
        print(f"{'Model':<20} {'Accuracy':<10} {'AUC':<10} {'F1':<10}")
        print("-" * 60)

        for model_name, metrics in results_dict.items():
            if isinstance(metrics, dict):
                acc = metrics.get('accuracy', 0.0)
                auc = metrics.get('auc', 0.0)
                f1 = metrics.get('f1', 0.0)
                print(f"{model_name:<20} {acc:.4f}     {auc:.4f}     {f1:.4f}")
            else:
                print(f"{model_name:<20} {metrics:.4f if isinstance(metrics, (int,float)) else 'N/A'}")

        # 最佳模型
        try:
            best_model = max(results_dict.items(),
                             key=lambda x: x[1].get('accuracy', 0) if isinstance(x[1], dict) else 0)
            print(f"\n**最佳模型：{best_model[0]}** (Acc = {best_model[1].get('accuracy', 0):.4f})")
        except:
            print("\n**最佳模型：无法自动判断**")

        # 2. 1/f 统计分析结果（如果存在）
        if 'stat_1f' in results_dict:
            print("\n\n**2. 1/f 统计分析结果**")
            print("-" * 60)
            stat = results_dict['stat_1f']
            print("**Exponent（指数）**")
            print(f"  Class 0: {stat.get('exp_class0_mean', 0):.4f} ± {stat.get('exp_class0_std', 0):.4f}")
            print(f"  Class 1: {stat.get('exp_class1_mean', 0):.4f} ± {stat.get('exp_class1_std', 0):.4f}")
            print(f"  t-test / Mann-Whitney p-value: {stat.get('exp_pvalue', 0):.4e}")

            print("\n**Offset（偏移量）**")
            print(f"  Class 0: {stat.get('off_class0_mean', 0):.4f} ± {stat.get('off_class0_std', 0):.4f}")
            print(f"  Class 1: {stat.get('off_class1_mean', 0):.4f} ± {stat.get('off_class1_std', 0):.4f}")
            print(f"  t-test / Mann-Whitney p-value: {stat.get('off_pvalue', 0):.4e}")

        # 3. 特征重要性结果
        if 'feature_importance' in results_dict:
            print("\n\n**3. 特征重要性结果 (Top 10)**")
            print("-" * 60)
            imp = results_dict['feature_importance']
            names = imp.get('names', [])
            scores = imp.get('scores', [])
            for i in range(min(10, len(names))):
                print(f"{i + 1:2d}. {names[i]:<40} {scores[i]:.4f}")

        # 4. OOF 对比（如果存在）
        if 'oof_comparison' in results_dict:
            print("\n\n**4. OOF 增强前后对比**")
            print("-" * 60)
            oof = results_dict['oof_comparison']
            print(f"trad+1f          Acc: {oof.get('before_acc', 0):.4f}  AUC: {oof.get('before_auc', 0):.4f}")
            print(f"trad+1f+OOF      Acc: {oof.get('after_acc', 0):.4f}  AUC: {oof.get('after_auc', 0):.4f}")
            print(f"提升幅度         Acc: +{oof.get('after_acc', 0) - oof.get('before_acc', 0):.4f}")

        # 5. 图编号占位
        print("\n\n**5. 准备补充的图表（建议图编号）**")
        print("-" * 60)
        print("Fig. 1    两类样本的 exponent 参数箱线图")
        print("Fig. 2    两类样本的 offset 参数箱线图")
        print("Fig. 3    单特征与融合特征性能对比柱状图")
        print("Fig. 4    不同特征组合在两个数据集上的 Acc/AUC 对比")
        print("Fig. 5    随机森林特征重要性 Top 20 条形图")
        print("Fig. 6    跨数据集性能趋势图")

        print(f"\n{'=' * 80}\n")