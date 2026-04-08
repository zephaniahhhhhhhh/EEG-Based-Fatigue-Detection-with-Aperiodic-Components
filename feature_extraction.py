# -*- coding: utf-8 -*-
"""
feature_extraction.py
通用特征提取模块 - 传统特征 + 稳定版 1/f 特征
"""

from joblib import Parallel, delayed
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis, ttest_ind
from specparam import SpectralModel
import time
import os
import warnings

warnings.filterwarnings('ignore')

# 优先使用 antropy（更快、更稳定），若未安装则 fallback 到 utils 中的自定义版本
try:
    from antropy import sample_entropy as antropy_sampen, app_entropy as antropy_appen
    ANTRPY_AVAILABLE = True
except ImportError:
    ANTRPY_AVAILABLE = False
    print("警告：未找到 antropy 库，将使用 utils.py 中的自定义 entropy 函数")

from utils import (
    bandpower, sample_entropy, approximate_entropy,
)


def _compute_one_trial(trial, sfreq, use_entropy=True):
    """单 trial 传统特征计算"""
    feat_list = []
    n_channels = trial.shape[0]

    for ch_idx in range(n_channels):
        ch_data = trial[ch_idx]

        # 频带功率特征
        theta = bandpower(ch_data, sfreq, (4, 8))
        alpha = bandpower(ch_data, sfreq, (8, 13))
        ratio = theta / (alpha + 1e-6)

        # 熵特征
        if use_entropy:
            if ANTRPY_AVAILABLE:
                se = antropy_sampen(ch_data, order=2, metric='chebyshev')
                ae = antropy_appen(ch_data, order=2, metric='chebyshev')
            else:
                se = sample_entropy(ch_data)
                ae = approximate_entropy(ch_data)
        else:
            se = ae = 0.0

        # 时域统计特征
        skew_val   = skew(ch_data, bias=False, nan_policy='omit')
        kurt_val   = kurtosis(ch_data, fisher=True, bias=False, nan_policy='omit')
        median_val = np.median(ch_data)
        max_val    = np.max(ch_data)
        min_val    = np.min(ch_data)
        fourth_val = np.mean((ch_data - np.mean(ch_data)) ** 4)

        feat_list.extend([
            ratio, se, ae, skew_val, kurt_val,
            median_val, max_val, min_val, fourth_val
        ])

    return feat_list


def extract_traditional_features(EEGsample, config, use_entropy=True):
    """提取传统特征"""
    start_time = time.time()
    if EEGsample.ndim != 3:
        raise ValueError(f"预期 3 维输入，实际 {EEGsample.shape}")

    n_trials, n_channels, n_times = EEGsample.shape
    print(f"传统特征提取 - 输入形状: ({n_trials}, {n_channels}, {n_times})")

    base_names = [
        "theta_alpha_ratio", "sample_entropy", "approx_entropy",
        "skewness", "kurtosis", "median", "max", "min", "fourth_moment"
    ]
    if not use_entropy:
        base_names = base_names[:1] + base_names[3:]

    feature_names = [f"{fname}_{ch}" for ch in config.ch_names for fname in base_names]
    n_features_expected = len(feature_names)

    print(f"预期特征维度：{n_features_expected}")

    all_features = Parallel(n_jobs=-1, verbose=10, backend='loky', prefer='threads')(
        delayed(_compute_one_trial)(EEGsample[i], config.sfreq, use_entropy)
        for i in range(n_trials)
    )

    X = np.array(all_features, dtype=float)

    if X.shape[1] != n_features_expected:
        raise ValueError(f"特征维度不匹配！实际 {X.shape[1]} vs 预期 {n_features_expected}")

    elapsed = time.time() - start_time
    print(f"传统特征提取完成 | 耗时 {elapsed:.1f}s | 形状 {X.shape}")

    feature_info = {
        'total': X.shape[1],
        'names': feature_names,
        'per_channel': len(base_names),
        'description': f"传统特征（entropy={'启用' if use_entropy else '禁用'}）"
    }

    return X, feature_info


def extract_1f_features_single_band_enhanced(EEGsample, config, fmin=1, fmax=40):
    """稳定版 1/f 特征提取（exponent + offset）"""
    n_trials, n_channels, n_times = EEGsample.shape

    all_exps = np.full((n_trials, n_channels), np.nan)
    all_offs = np.full((n_trials, n_channels), np.nan)

    failure_count = 0
    printed = 0

    should_plot = True
    plot_done = False
    plot_save_dir = "./psd_1f_plots"
    os.makedirs(plot_save_dir, exist_ok=True)

    for i in range(n_trials):
        trial = EEGsample[i]
        for ch_idx in range(n_channels):
            ch_data = trial[ch_idx]

            if np.all(ch_data == 0):
                continue

            try:
                nperseg = min(256, n_times // 2) if n_times > 256 else n_times
                f, psd = welch(ch_data, fs=config.sfreq,
                               nperseg=nperseg, noverlap=nperseg // 2)

                mask = (f >= fmin) & (f <= fmax)
                if np.sum(mask) < 8:
                    failure_count += 1
                    continue

                psd_safe = np.maximum(psd, 1e-12)

                sm = SpectralModel(
                    peak_width_limits=(2.0, 12.0),
                    aperiodic_mode='fixed',
                    max_n_peaks=6
                )

                sm.fit(f[mask], psd_safe[mask])

                aperiodic_params = sm.get_params('aperiodic')
                offs = aperiodic_params[0]
                exp  = aperiodic_params[1]

                all_exps[i, ch_idx] = exp
                all_offs[i, ch_idx] = offs

            except Exception as e:
                failure_count += 1
                if printed < 5:
                    print(f"[trial {i}, ch {ch_idx}] fit 失败：{str(e)}")
                    printed += 1

    print(f"总通道数：{n_trials * n_channels} | fit 失败：{failure_count} ({failure_count/(n_trials*n_channels):.2%})")

    # 中位数填补
    for i in range(n_trials):
        valid_exps = all_exps[i][~np.isnan(all_exps[i])]
        valid_offs = all_offs[i][~np.isnan(all_offs[i])]
        if len(valid_exps) > 0:
            all_exps[i] = np.nan_to_num(all_exps[i], nan=np.median(valid_exps))
            all_offs[i] = np.nan_to_num(all_offs[i], nan=np.median(valid_offs))
        else:
            all_exps[i] = 1.2
            all_offs[i] = -15.0

    X_1f = np.concatenate([all_exps, all_offs], axis=1)

    # 补充 names 键（关键修复）
    names = [f"exp_{ch}" for ch in config.ch_names] + [f"offset_{ch}" for ch in config.ch_names]

    info = {
        'total_features': n_channels * 2,
        'per_channel': 2,
        'names': names,
        'description': 'exponent & offset (1-40 Hz), flattened',
    }

    return X_1f, info


def oof_probs(X, y, base_clf=None, n_splits=5, random_state=42):
    """Out-of-Fold 概率增强特征"""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC

    if base_clf is None:
        base_clf = SVC(kernel='rbf', probability=True, random_state=random_state)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', base_clf)
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros((X.shape[0], len(np.unique(y))), dtype=float)

    for tr, te in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])

    return oof


def compute_1f_statistics(X_1f, y, ch_names):
    """计算 exponent 和 offset 在两类中的均值±标准差 + 统计检验 p 值"""
    n_ch = len(ch_names)
    exp_features = X_1f[:, :n_ch]
    off_features = X_1f[:, n_ch:]

    stat_dict = {
        'exp_class0_mean': float(exp_features[y == 0].mean()),
        'exp_class0_std': float(exp_features[y == 0].std()),
        'exp_class1_mean': float(exp_features[y == 1].mean()),
        'exp_class1_std': float(exp_features[y == 1].std()),
        'exp_pvalue': float(ttest_ind(exp_features[y == 0].flatten(), exp_features[y == 1].flatten(), equal_var=False).pvalue),

        'off_class0_mean': float(off_features[y == 0].mean()),
        'off_class0_std': float(off_features[y == 0].std()),
        'off_class1_mean': float(off_features[y == 1].mean()),
        'off_class1_std': float(off_features[y == 1].std()),
        'off_pvalue': float(ttest_ind(off_features[y == 0].flatten(), off_features[y == 1].flatten(), equal_var=False).pvalue),
    }

    print(f"  1/f 统计完成 - Exponent p={stat_dict['exp_pvalue']:.2e} | Offset p={stat_dict['off_pvalue']:.2e}")
    return stat_dict


def get_1f_feature_names(ch_names):
    """生成 1/f 特征名称（供特征重要性使用）"""
    return [f"exp_{ch}" for ch in ch_names] + [f"offset_{ch}" for ch in ch_names]