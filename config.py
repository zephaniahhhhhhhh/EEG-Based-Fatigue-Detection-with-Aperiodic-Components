# -*- coding: utf-8 -*-
"""
config.py
数据集配置文件 - 集中管理不同数据集的参数
这段 config.py 的作用本质上是：把不同 EEG 数据集的所有“参数、结构、元信息”集中统一管理，让主程序不用关心数据细节，直接调用配置即可运行。

具体来说，它实现了以下功能：

首先，它定义了一个通用基类 DatasetConfig，负责所有数据集共享的基础功能，比如：自动创建结果保存目录（results_xxx）和图像目录（figures），以及提供一个 get_info() 方法，用来打印当前数据集的关键信息（数据路径、通道数、通道名、对称通道、采样率、类别等），相当于一个“数据集说明书”。

然后，它针对不同数据集（SAD 和 SEED）分别定义了子类：

SADConfig
指定数据文件 SAD.mat
设置采样率 128Hz、30个通道及其名称（标准10-20系统）
定义左右对称通道对（如 Fp1-Fp2、C3-C4 等），并自动转换为索引 pair_idx
定义分类任务（Alert vs Fatigue）
指定数据字段名（EEGsample）和标签字段（substate）
指定参考通道（Cz），用于后续可能的连接性或信息计算

SEEDConfig
指定数据文件 SEED.mat
设置17个通道（偏后脑区）
同样定义对称通道对
分类为 Alert vs Drowsy
支持被试编号（subindex），说明可以做跨被试实验

自动选择中间通道作为参考通道

这些子类的作用是：把“每个数据集的结构差异”封装起来，让后续代码统一处理。
此外，还提供了一个 CustomConfig，用于你以后接入新数据集时，只需要传入通道名、类别等信息，就可以快速复用整个框架。
接着，这个文件还定义了一个标准频段列表 STANDARD_BANDS（如 1–20Hz、1–40Hz 等），方便后续undefined谱或1/f分析统一使用不同频段配置。
最后，通过 get_config(dataset_name) 这个函数，实现了配置工厂模式：只需要传入 'SAD' 或 'SEED'，就能自动返回对应的配置对象，主程序无需写 if-else 判断。
"""

from pathlib import Path
import numpy as np


class DatasetConfig:
    """数据集配置基类"""

    def __init__(self, name):
        self.name = name
        self.output_dir = Path(f"results_{name}")
        self.output_dir.mkdir(exist_ok=True)
        self.figure_dir = self.output_dir / "figures"
        self.figure_dir.mkdir(exist_ok=True)

    def get_info(self):
        """打印配置信息"""
        print(f"\n{'=' * 60}")
        print(f"📋 数据集配置: {self.name}")
        print(f"{'=' * 60}")
        print(f"  数据文件: {self.data_path}")
        print(f"  通道数: {self.n_channels}")
        print(f"  通道名: {self.ch_names[:5]}... (共{len(self.ch_names)}个)")
        print(f"  对称通道对: {self.n_pairs}对")
        print(f"  采样率: {self.sfreq} Hz")
        print(f"  类别数: {self.n_classes}")
        print(f"  类别名称: {self.class_names}")
        print(f"{'=' * 60}\n")


class SADConfig(DatasetConfig):
    """SAD数据集配置"""

    def __init__(self):
        super().__init__("SAD")

        # 数据路径
        self.data_path = 'SAD.mat'

        # 数据集参数
        self.sfreq = 128
        self.n_channels = 30

        # 通道信息
        self.ch_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz',
            'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz',
            'CP4', 'TP8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2'
        ]

        # 对称通道对（用于后续特征计算）
        self.symmetric_pairs = [
            ('Fp1', 'Fp2'), ('F7', 'F8'), ('F3', 'F4'), ('FT7', 'FT8'),
            ('FC3', 'FC4'), ('T3', 'T4'), ('C3', 'C4'), ('TP7', 'TP8'),
            ('CP3', 'CP4'), ('T5', 'T6'), ('P3', 'P4'), ('O1', 'O2')
        ]
        self.n_pairs = len(self.symmetric_pairs)
        self.pair_idx = [(self.ch_names.index(a), self.ch_names.index(b))
                         for a, b in self.symmetric_pairs]

        # 分类信息
        self.n_classes = 2
        self.class_names = ['Alert', 'Drowsy']
        self.need_label_mapping = False
        self.label_key = 'substate'
        self.data_key = 'EEGsample'
        self.has_subject_index = False
        # 参考通道（用于全局MI计算）
        self.ref_channel_idx = 14  # Cz
        self.ref_channel_name = 'Cz'


class SEEDConfig(DatasetConfig):
    """SEED数据集配置"""
    def __init__(self):
        super().__init__("SEED")

        # 数据路径
        self.data_path = 'SEED.mat'
        # 数据集参数
        self.sfreq = 128
        self.n_channels = 17
        # 通道信息
        self.ch_names = [
            'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2',
            'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2'
        ]

        # 对称通道对（用于后续特征计算）
        self.symmetric_pairs = [
            ('FT7', 'FT8'), ('T7', 'T8'), ('TP7', 'TP8'),
            ('CP1', 'CP2'), ('P1', 'P2'), ('PO3', 'PO4'), ('O1', 'O2')
        ]
        self.n_pairs = len(self.symmetric_pairs)
        self.pair_idx = [(self.ch_names.index(a), self.ch_names.index(b))
                         for a, b in self.symmetric_pairs]

        # 分类信息（SEED是2分类）
        self.n_classes = 2
        self.class_names = ['Alert', 'Drowsy']

        # 标签处理（SEED需要重映射）
        self.need_label_mapping = False
        self.label_key = 'substate'

        # 其他数据键
        self.data_key = 'EEGsample'
        self.has_subject_index = True
        self.subject_key = 'subindex'

        # 参考通道
        self.ref_channel_idx = self.n_channels // 2  # 中间通道
        self.ref_channel_name = self.ch_names[self.ref_channel_idx]


# 频段配置（通用）
STANDARD_BANDS = [
    ("1-20Hz", 1, 20),
    ("1-40Hz", 1, 40),
    ("5-40Hz", 5, 40),
    ("5-20Hz", 5, 20),
    ("20-40Hz", 20, 40),
]


def get_config(dataset_name):
    """根据名称获取配置"""
    configs = {
        'SAD': SADConfig,
        'SEED': SEEDConfig,
    }

    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")

    return configs[dataset_name]()