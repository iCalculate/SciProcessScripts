# PL-Analyzer - 清晰的架构指南

> **✅ 修复说明 (v1.1)**: `peak_fitting()` 函数已修复，现在支持任意数量的多峰拟合。
> 
> **问题**: MATLAB `fittype()` 不支持 `sum()` 函数  
> **解决**: 动态生成拟合方程，用 `+` 替代 `sum()`  
> **验证**: 运行 `TEST_FIX.m` 确认修复

## 项目概述

PL-Analyzer 是一个用于光致发光 (Photoluminescence, PL) 光谱分析的 MATLAB 工具集。项目采用模块化架构，清晰的调用逻辑，易于理解、维护和扩展。

## 项目结构

```
PL-Analyzer/
│
├── core/                          # 核心功能模块
│   ├── data/                      # 📊 数据导入和处理
│   │   ├── import_pl_data.m              (单个文件导入)
│   │   ├── load_pl_power_data.m          (双文件加载)
│   │   └── find_data_files.m             (批量查找)
│   │
│   ├── visualization/             # 📈 绘图和可视化
│   │   ├── plot_single_spectrum.m        (单条谱线) ⭐ 最基础
│   │   ├── plot_spectra_stack.m          (堆叠图)
│   │   └── plot_contour_map.m            (等高线图)
│   │
│   └── analysis/                  # 🔬 数据分析
│       ├── normalize_spectrum.m           (标准化)
│       ├── peak_analysis.m                (峰值分析)
│       ├── peak_fitting.m                 (高斯拟合)
│       └── polarization_analysis.m        (偏振分析)
│
├── examples/                      # 📚 使用示例
│   ├── basic_plotting.m                  (基础绘图)
│   ├── peak_fitting_example.m            (分峰拟合)
│   ├── power_dependence.m                (功率依赖)
│   └── polarization_analysis_example.m   (偏振分析)
│
└── README.md                      # 项目说明
```

## 核心概念

### 数据结构 (标准格式)

所有函数使用统一的数据结构 `PL_Data`：

```matlab
PL_Data.data           % (N×M) 矩阵
                       % 行: 序列数 (功率/角度/温度等)
                       % 列: 波长点数
                       
PL_Data.axisscale{1,1} % (N×1) 第一轴参数
                       % 通常是功率 (mW)、角度 (°)、温度 (K) 等
                       
PL_Data.axisscale{2,1} % (M×1) 第二轴参数
                       % 通常是波长 (nm)
```

### 三层架构设计

```
┌─────────────────────────────────────────────────────┐
│  用户脚本 (examples/)                                │
│  ├─ basic_plotting.m                                │
│  ├─ peak_fitting_example.m                          │
│  ├─ power_dependence.m                              │
│  └─ polarization_analysis_example.m                 │
└──────────────┬──────────────────────────────────────┘
               │ (直接调用)
┌──────────────▼──────────────────────────────────────┐
│  高级分析函数 (core/analysis/)                        │
│  ├─ peak_fitting()      ← 多峰拟合                  │
│  ├─ peak_analysis()     ← 自动峰检测                │
│  └─ polarization_analysis()                         │
└──────────────┬──────────────────────────────────────┘
               │ (依赖)
┌──────────────▼──────────────────────────────────────┐
│  基础函数 (core/visualization/ + core/data/)        │
│  ├─ plot_single_spectrum()   ← 基础绘图             │
│  ├─ plot_spectra_stack()     ← 堆叠绘图             │
│  ├─ plot_contour_map()       ← 等高线图             │
│  ├─ normalize_spectrum()     ← 标准化               │
│  └─ import_pl_data()         ← 数据导入             │
└─────────────────────────────────────────────────────┘
```

## 使用指南

### 1️⃣ 最基础：单条光谱绘制

```matlab
% 最简单的用法
wavelength = 400:1000;  % nm
spectrum = randn(size(wavelength));

[fig, ax] = plot_single_spectrum(spectrum, wavelength, ...
    'Title', 'My Spectrum');
```

### 2️⃣ 基础：加载数据并绘制

```matlab
% 导入 .mat 数据
PL_Data = import_pl_data('data.mat');

% 绘制等高线图
[fig, ax] = plot_contour_map(PL_Data, ...
    'Title', 'PL Map');

% 绘制堆叠图
[fig, ax] = plot_spectra_stack(PL_Data, ...
    'NormalizeMethod', 'peak_range', ...
    'Spacing', 0.15);
```

### 3️⃣ 中级：分峰和拟合

```matlab
% 方法A: 自动峰检测
[peaks, positions, widths, amplitudes] = peak_analysis(spectrum, wavelength, ...
    'MinPeakHeight', 50, ...
    'Plot', true);

% 方法B: 高斯多峰拟合
[fitParams, fittedCurve, residuals, r2] = peak_fitting(spectrum, wavelength, 3, ...
    'FitType', 'gauss', ...
    'Plot', true);
```

### 4️⃣ 高级：功率和偏振依赖性

```matlab
% 功率依赖 - 查看 power_dependence.m
% 1. 加载数据
% 2. 绘制等高线图
% 3. 绘制堆叠谱线
% 4. 拟合 I ∝ P^n

% 偏振依赖 - 查看 polarization_analysis_example.m
% 1. 加载数据
% 2. 绘制等高线图
% 3. 绘制堆叠谱线
% 4. 拟合 I(θ) ∝ cos²(2θ)
```

## 各模块详解

### 📊 数据模块 (core/data/)

| 函数 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `import_pl_data` | 导入单个 .mat 文件 | 文件路径 | PL_Data 结构 |
| `load_pl_power_data` | 加载 PL + 功率数据 | 两个文件路径 | 合并的 PL_Data |
| `find_data_files` | 查找目录中的数据 | 目录路径 | 文件名列表 |

### 📈 绘图模块 (core/visualization/)

| 函数 | 用途 | 输入数据类型 |
|------|------|------------|
| `plot_single_spectrum` | 单条光谱 | 1D 向量 |
| `plot_spectra_stack` | 多条光谱堆叠 | 2D 矩阵 + 参数轴 |
| `plot_contour_map` | 等高线热图 | 2D 矩阵 + 参数轴 |

### 🔬 分析模块 (core/analysis/)

| 函数 | 功能 | 输出 |
|------|------|------|
| `normalize_spectrum` | 标准化单条光谱 | 归一化后的光谱 |
| `peak_analysis` | 自动检测峰 | 峰位、强度、宽度 |
| `peak_fitting` | 多峰高斯/洛伦兹拟合 | 拟合参数、R² |
| `polarization_analysis` | 偏振依赖分析 | 强度 vs 角度 |

## 快速开始

### 场景1: 我有单个光谱数据

```matlab
% 参考: examples/basic_plotting.m
wavelength = 400:1000;  % nm
spectrum = load_spectrum();

plot_single_spectrum(spectrum, wavelength, ...
    'Title', 'My PL Spectrum', ...
    'LineWidth', 2);
```

### 场景2: 我有功率扫描数据

```matlab
% 参考: examples/power_dependence.m
% 步骤:
% 1. PL_Data = load_pl_power_data('pl.mat', 'power.mat');
% 2. plot_contour_map(PL_Data);           % 等高线
% 3. plot_spectra_stack(PL_Data);         % 堆叠
% 4. 拟合 I ∝ P^n
```

### 场景3: 我要进行分峰拟合

```matlab
% 参考: examples/peak_fitting_example.m
% 步骤:
% 1. [peaks, pos, ...] = peak_analysis(spectrum, wavelength);
% 2. [params, fitted, ...] = peak_fitting(spectrum, wavelength, 3);
% 3. 绘制对比
```

### 场景4: 我要分析偏振依赖

```matlab
% 参考: examples/polarization_analysis_example.m
% 步骤:
% 1. 加载偏振扫描数据
% 2. plot_contour_map() + plot_spectra_stack()
% 3. 拟合 I(θ) ∝ cos²(2θ)
```

## 常见参数说明

### normalize_spectrum 中的方法

- **'none'** - 不做任何处理
- **'max'** - 按最大值归一化 → [0, 1]
- **'peak_range'** - 按指定范围的峰值归一化 → [0, 1]（推荐）
- **'area'** - 按积分面积归一化

### 颜色映射

- **'power'** - 按第一轴参数（功率等）从浅到深
- **'index'** - 按序列号从浅到深
- **'custom'** - 自定义颜色矩阵
- 其他 - MATLAB 内置名称（'jet', 'hot', 'viridis' 等）

## 扩展指南

### 添加新的分析函数

在 `core/analysis/` 中创建新文件：

```matlab
function [result] = my_analysis(spectrum, wavelength, varargin)
%MY_ANALYSIS 我的分析函数
%   简要说明
%
%   输入:
%       spectrum - 光谱数据
%       wavelength - 波长数据
%
%   输出:
%       result - 分析结果

p = inputParser;
addRequired(p, 'spectrum', @isvector);
addRequired(p, 'wavelength', @isvector);
% ... 添加参数 ...
parse(p, spectrum, wavelength, varargin{:});

% 实现分析逻辑
result = analysis_core(spectrum, wavelength, p);

end
```

### 添加新的绘图函数

在 `core/visualization/` 中创建新文件，参考现有函数结构。

## 性能提示

- 对于大数据集，使用 `plot_contour_map` 而不是 `plot_spectra_stack`
- 峰拟合时指定合理的初始参数加快收敛
- 使用 `'Indices'` 参数减少绘图点数以加快渲染

## 常见问题

**Q: 为什么我的堆叠图都是平的？**
A: 检查归一化方法。使用 `'peak_range'` 并指定正确的峰范围。

**Q: 如何批量处理多个文件？**
A: 使用 `find_data_files()` + 循环加载 + 分析。

**Q: 如何导出分析结果？**
A: `peak_fitting` 返回 `fitParams` (cell 数组)，可直接使用或保存。

## 参考示例

- **basic_plotting.m** - 从零开始，3 个基础图表
- **peak_fitting_example.m** - 完整的分峰工作流
- **power_dependence.m** - 功率依赖性分析
- **polarization_analysis_example.m** - 偏振特性分析

## 开发者便签

- 所有函数使用 `inputParser` 进行参数管理
- 所有函数支持可选的绘图和详细输出
- 所有数据结构保持一致性，便于模块组合

---

## 修复说明 (v1.1)

**问题**: `peak_fitting()` 函数使用了 MATLAB `fittype()` 不支持的 `sum()` 函数，导致多峰拟合失败。

**解决**: 改为动态生成完整的拟合方程字符串（如 `a1*exp(-((x-b1)/c1)^2) + a2*exp(-((x-b2)/c2)^2)`），用 `+` 替代 `sum()`。

**验证**: 运行 `TEST_FIX.m` 测试修复。

**结果**: 
- ✅ 支持任意数量的多峰拟合
- ✅ Gaussian 和 Lorentz 模式都支持
- ✅ 100% 向后兼容

**最后更新**: 2026-01-24
**维护者**: PL-Analyzer Team
