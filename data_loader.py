import time
import numpy as np
import scipy.io as sio
import h5py
from pathlib import Path
import joblib
import hashlib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def get_cache_path(config, cache_dir="./data_cache"):
    """生成唯一缓存路径（基于文件路径、修改时间、关键配置）"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    file_path = Path(config.data_path)
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    mtime = file_path.stat().st_mtime
    fingerprint = f"{file_path}_{mtime}_{config.data_key}_{config.label_key}_{config.n_channels}"
    hash_str = hashlib.md5(fingerprint.encode()).hexdigest()[:12]

    return cache_dir / f"{config.name}_{hash_str}.joblib"


def loadmat_hdf5(filename):
    """加载 MATLAB v7.3+ HDF5 格式文件"""
    with h5py.File(filename, 'r') as f:
        data = {}
        for k in f.keys():
            item = f[k]
            if isinstance(item, h5py.Dataset):
                val = item[()]
                # 处理字节字符串
                if val.dtype.kind == 'S':
                    val = np.vectorize(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)(val)
                if val.shape == ():
                    val = val[()]
                data[k] = val
            elif isinstance(item, h5py.Group):
                data[k] = {subk: f[k][subk][()] for subk in f[k].keys()}
        return data


def load_dataset(config, use_cache=True, force_refresh=False, cache_dir="./data_cache"):
    """加载 EEG 数据集（支持 SAD 和 SEED），自动标准化形状并缓存"""
    start_total = time.time()

    cache_path = get_cache_path(config, cache_dir)

    # 尝试从缓存加载
    if use_cache and cache_path.exists() and not force_refresh:
        print(f"  ⚡ 从缓存加载: {cache_path.name}")
        try:
            data_dict = joblib.load(cache_path)
            elapsed = time.time() - start_total
            print(f"  缓存加载完成，耗时 {elapsed:.2f}s")

            X = data_dict['X']
            y = data_dict['y']
            print(f"\n  从缓存读取的数据摘要:")
            print(f"    样本数:       {X.shape[0]:>6}")
            print(f"    通道数:       {X.shape[1]:>6}")
            print(f"    时间点数:     {X.shape[2]:>6}")
            counts = np.bincount(y, minlength=2)
            total = len(y)
            print(f"    Alert (0) :   {counts[0]:>6}  ({counts[0]/total:.1%})")
            print(f"    Drowsy(1) :   {counts[1]:>6}  ({counts[1]/total:.1%})")
            if 'subject' in data_dict:
                print(f"    被试数量:     {len(np.unique(data_dict['subject']))}")
            print(f"{'═' * 60}\n")
            return data_dict
        except Exception as e:
            print(f"  缓存加载失败 ({str(e)}) → 回退到原始读取...")

    # 原始加载流程
    start_load = time.time()
    print(f"\n{'═' * 60}")
    print(f"📂 读取原始文件: {config.name}")
    print(f"   路径: {config.data_path}")
    print(f"{'═' * 60}")

    path = Path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    # 尝试两种加载方式
    try:
        data = sio.loadmat(str(path))
        print("  使用 scipy.io.loadmat 加载成功（旧版格式）")
    except NotImplementedError:
        print("  检测到 MATLAB v7.3+ (HDF5) 格式，使用 h5py...")
        data = loadmat_hdf5(str(path))

    # ── 提取 EEG 数据 ────────────────────────────────────────────────
    if config.data_key not in data:
        raise KeyError(f"未找到数据键: {config.data_key}，可用键: {list(data.keys())}")

    EEG = data[config.data_key]
    if EEG.dtype == np.object_:
        EEG = np.array([np.squeeze(x) for x in EEG.flat], dtype=np.float64)

    if EEG.ndim != 3:
        raise ValueError(f"EEG 数据应为 3 维，实际形状: {EEG.shape}")

    print(f"  原始形状: {EEG.shape}")

    # ── 自动形状标准化（统一转为 (n_trials, n_channels, n_timepoints)）──
    dim0, dim1, dim2 = EEG.shape

    if dim0 == config.n_channels:
        # (channels, trials, times) → (trials, channels, times)
        EEG = np.transpose(EEG, (1, 0, 2))
        print("  检测到 (channels, trials, times) 格式，已转置 → (trials, channels, times)")

    elif dim2 == config.n_channels:
        # (trials, times, channels) → (trials, channels, times)
        EEG = np.transpose(EEG, (0, 2, 1))
        print("  检测到 (trials, times, channels) 格式，已转置 → (trials, channels, times)")

    elif dim1 == config.n_channels:
        print("  已经是标准格式 (n_trials, n_channels, n_timepoints)")
    else:
        raise ValueError(
            f"无法自动识别通道维度！\n"
            f"当前形状: {EEG.shape}，期望通道数: {config.n_channels}\n"
            f"请检查 mat 文件结构或手动调整 config.n_channels"
        )

    # 最终校验
    if EEG.shape[1] != config.n_channels:
        raise ValueError(f"标准化后通道数仍不匹配：{EEG.shape[1]} ≠ {config.n_channels}")

    # ── 提取标签 ─────────────────────────────────────────────────────
    if config.label_key not in data:
        raise KeyError(f"未找到标签键: {config.label_key}")

    y = np.array(data[config.label_key]).flatten().astype(np.int32)

    unique_y = np.unique(y)
    if not np.array_equal(unique_y, [0, 1]):
        print(f"  警告：标签不是严格的 0/1，原始值: {unique_y}")

    # ── 构建返回字典 ─────────────────────────────────────────────────
    data_dict = {
        'X': EEG,
        'y': y,
        'sfreq': config.sfreq,
        'n_channels': config.n_channels,
        'ch_names': config.ch_names,
    }

    # ── 尝试提取被试索引 ─────────────────────────────────────────────
    subject_key_candidates = ['subindex', 'subject_index', 'sub_idx', 'subject']
    subject_array = None
    for key in subject_key_candidates:
        if key in data:
            subject_array = np.array(data[key]).flatten().astype(np.int32)
            print(f"√ 从 mat 文件读取到被试索引，键名：'{key}'，唯一值数：{len(np.unique(subject_array))}")
            break

    if subject_array is not None:
        if len(subject_array) != len(y):
            print(f"⚠️ 被试索引长度 {len(subject_array)} 与样本数 {len(y)} 不匹配！将忽略此栏位")
        else:
            # 统一转为 0-based
            if subject_array.min() == 1:
                subject_array -= 1
                print("  已将 subindex 从 1-based 转为 0-based")
            data_dict['subject'] = subject_array
    else:
        print("⚠️ 未在 mat 文件中找到任何被试索引相关键，subject 栏位将为 None")

    # ── 保存缓存 ─────────────────────────────────────────────────────
    if use_cache:
        print(f"  保存缓存 → {cache_path.name}")
        joblib.dump(data_dict, cache_path, compress=3)

    elapsed = time.time() - start_load
    print(f"  原始读取 + 处理耗时: {elapsed:.2f}s")
    print(f"  最终形状: {EEG.shape} (n_trials, n_channels, n_timepoints)")
    print(f"{'═' * 60}")

    # 打印最终摘要
    counts = np.bincount(y, minlength=2)
    total = len(y)
    print(f"    Alert (0) :   {counts[0]:>6}  ({counts[0]/total:.1%})")
    print(f"    Drowsy(1) :   {counts[1]:>6}  ({counts[1]/total:.1%})")
    if 'subject' in data_dict:
        print(f"    被试数量:     {len(np.unique(data_dict['subject']))}")

    print(f"总耗时（含缓存检查）: {time.time() - start_total:.2f}s\n")

    return data_dict