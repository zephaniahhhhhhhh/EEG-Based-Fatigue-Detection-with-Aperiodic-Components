# -*- coding: utf-8 -*-
"""
utils.py
基础工具函数（bandpower、熵、相关系数等）
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import pearsonr


def bandpower(data, sf, band):
    """计算指定频段的功率"""
    f, Pxx = welch(data, sf, nperseg=min(len(data), sf * 2))
    if len(f) < 2:
        return 0.0
    freq_res = f[1] - f[0]
    idx = (f >= band[0]) & (f <= band[1])
    return float(np.trapz(Pxx[idx], dx=freq_res)) if np.any(idx) else 0.0


def sample_entropy(signal, m=2, r=0.2):
    """样本熵（自定义版本，当 antropy 不可用时备用）"""
    N = len(signal)
    if N <= m + 1:
        return 0.0
    r *= (np.std(signal) + 1e-12)

    def _phi(mm):
        x = np.array([signal[i:i + mm] for i in range(N - mm + 1)])
        dist = np.max(np.abs(x[:, None] - x[None, :]), axis=2)
        C = np.sum(dist <= r, axis=0) - 1
        denom = (N - mm + 1) * (N - mm)
        return np.sum(C) / max(denom, 1)

    try:
        return -np.log((_phi(m + 1) + 1e-12) / (_phi(m) + 1e-12))
    except Exception:
        return 0.0


def approximate_entropy(signal, m=2, r=0.2):
    """近似熵（自定义版本，当 antropy 不可用时备用）"""
    N = len(signal)
    if N <= m + 1:
        return 0.0
    r *= (np.std(signal) + 1e-12)

    def _phi(mm):
        x = np.array([signal[i:i + mm] for i in range(N - mm + 1)])
        dist = np.max(np.abs(x[:, None] - x[None, :]), axis=2)
        C = np.mean((dist <= r).astype(float), axis=1) + 1e-12
        return np.mean(np.log(C))

    try:
        return abs(_phi(m) - _phi(m + 1))
    except Exception:
        return 0.0


def safe_pearsonr(x, y):
    """安全的Pearson相关"""
    try:
        r, _ = pearsonr(x, y)
        return float(r) if np.isfinite(r) else 0.0
    except Exception:
        return 0.0