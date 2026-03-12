# Raman-Analyzer - 拉曼光谱分析工具集

## 项目概述

Raman-Analyzer 是一个用于拉曼 (Raman) 光谱分析的 MATLAB 工具集，采用与 PL-Analyzer 相同的模块化架构。项目支持单条光谱绘制、功率依赖性分析、拉曼位移范围指定等功能。

## 项目结构

```
Raman-Analyzer/
│
├── core/                      # 核心功能模块
│   ├── data/                  # 📊 数据导入和处理
│   │   ├── import_raman_data.m          (单个文件导入)
│   │   └── load_raman_power_data.m      (双文件加载)
│   │
│   └── visualization/         # 📈 绘图和可视化
│       ├── plot_single_raman.m          (单条谱线) ⭐ 最基础
│       ├── plot_raman_stack.m           (堆叠图)
│       └── plot_raman_contour.m         (等高线图)
│
├── examples/                  # 📚 使用示例
│   ├── basic_plotting.m               (基础绘图)
│   └── power_dependence.m             (功率依赖)
│
└── README.md                  # 项目说明
```

## 核心特性

✅ **基础单谱线绘制** - `plot_single_raman()`
✅ **多谱线堆叠绘制** - `plot_raman_stack()`
✅ **等高线热图** - `plot_raman_contour()`
✅ **功率依赖分析** - 自动拟合 I ∝ P^n
✅ **灵活的拉曼范围指定** - 专注于特定峰（D、G、2D等）

## 数据结构

所有函数使用统一的 `RamanData` 结构：

```matlab
RamanData.data           % (N×M) 矩阵
                         % 行: 序列数 (功率/温度)
                         % 列: 拉曼位移点数
                         
RamanData.axisscale{1,1} % (N×1) 第一轴参数 (通常为功率 mW)
RamanData.axisscale{2,1} % (M×1) 拉曼位移 (cm⁻¹)
```

## 快速开始

### 最简单：绘制单条拉曼光谱

```matlab
ramanShift = 0:3000;  % cm⁻¹
spectrum = randn(size(ramanShift));

[fig, ax] = plot_single_raman(spectrum, ramanShift, ...
    'Title', 'My Raman Spectrum');
```

### 中级：加载数据并创建图表

```matlab
% 导入 .mat 数据
RamanData = import_raman_data('raman.mat');

% 等高线图
plot_raman_contour(RamanData, 'Title', 'Raman Map');

% 堆叠图
plot_raman_stack(RamanData, ...
    'NormalizeMethod', 'max', ...
    'Spacing', 0.15);
```

### 高级：功率依赖性分析

```matlab
% 加载功率扫描数据
RamanData = load_raman_power_data('raman.mat', 'power.mat');

% 整体浏览
plot_raman_contour(RamanData);

% 细节对比
plot_raman_stack(RamanData, ...
    'RamanRange', [1200 1700], ...  % D-G 区域
    'NormalizeMethod', 'peak_range');
```

## 各模块详解

### 📊 数据模块

| 函数 | 功能 |
|------|------|
| `import_raman_data` | 导入单个 .mat 文件 |
| `load_raman_power_data` | 加载 Raman + 功率数据 |

### 📈 绘图模块

| 函数 | 用途 | 数据类型 |
|------|------|---------|
| `plot_single_raman` | 单条光谱 | 1D 向量 |
| `plot_raman_stack` | 多条堆叠 | 2D 矩阵 + 参数轴 |
| `plot_raman_contour` | 等高线图 | 2D 矩阵 + 参数轴 |

## 常见参数

### plot_raman_stack 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `Indices` | 要绘制的索引 | 所有 |
| `RamanRange` | 拉曼范围 [min max] cm⁻¹ | 全部 |
| `NormalizeMethod` | 归一化: none/max/peak_range/area | 'max' |
| `PeakRange` | 峰值范围 [min max] cm⁻¹ | [200 3000] |
| `Spacing` | 曲线间距 | 0.1 |
| `Colormap` | 颜色: power/index/jet/hot | 'power' |

### 拉曼光谱常见峰位

| 峰名 | 位置 (cm⁻¹) | 说明 |
|------|-----------|------|
| D | ~1350 | Disorder (无定形碳) |
| G | ~1580 | Graphite (石墨) |
| 2D | ~2700 | 二阶 D 峰 |
| D' | ~1620 | D 峰的激活态 |

## 工作流程

### 工作流1：基础绘图（5分钟）
```
数据加载 → plot_single_raman() → 单条光谱展示
```

### 工作流2：功率扫描分析（15分钟）
```
加载数据 → plot_raman_contour() → 整体浏览
      → plot_raman_stack() → 细节对比
      → 拟合 I ∝ P^n → 参数提取
```

## 参考示例

- **basic_plotting.m** - 从零开始，3个基础图表
- **power_dependence.m** - 功率依赖性完整分析

## 与 PL-Analyzer 的差异

| 项目 | PL-Analyzer | Raman-Analyzer |
|------|------------|-----------------|
| X轴单位 | 波长 (nm) | 拉曼位移 (cm⁻¹) |
| 标准归一化 | peak_range | max |
| 典型范围 | 400-1000 nm | 0-3000 cm⁻¹ |
| 峰位 | 单个峰 | 多个峰 (D, G, 2D) |

## 快速开始

运行 `QUICKSTART.m` 了解基本概念和用法。

---

**版本**: 1.0 (基于 PL-Analyzer v2.0 架构)
**更新日期**: 2026-01-24
**MATLAB 版本**: R2018b 及以上
