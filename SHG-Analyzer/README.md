# SHG-Analyzer - 二次谐波(SHG)光谱分析工具集

## 项目概述

**SHG-Analyzer** 是一个用于二次谐波 (Second Harmonic Generation, SHG) 光谱分析的 MATLAB 工具集，采用与 PL-Analyzer 和 Raman-Analyzer 相同的**模块化架构**。

该工具集提供了从数据导入、光谱绘制、到物理分析的完整工作流程，特别针对功率依赖性分析进行了优化。

### ✨ 核心特性

- ✅ **基础单谱线绘制** - `plot_single_shg()`
- ✅ **多谱线堆叠绘制** - `plot_shg_stack()`
- ✅ **二维等高线热图** - `plot_shg_contour()`
- ✅ **功率依赖性拟合** - `shg_power_fitting()` (重点功能)
- ✅ **灵活的波长范围指定** - 聚焦特定的SHG峰
- ✅ **多种归一化方法** - max/peak_range/area
- ✅ **对数坐标轴支持** - 适合宽动态范围的数据

---

## 项目结构

```
SHG-Analyzer/
│
├── core/                      # 核心功能模块
│   ├── data/                  # 📊 数据导入和处理
│   │   ├── import_shg_data.m          (单个文件导入)
│   │   └── load_shg_power_data.m      (双文件加载 + 功率标定)
│   │
│   ├── visualization/         # 📈 绘图和可视化
│   │   ├── plot_single_shg.m          (单条谱线) ⭐ 最基础
│   │   ├── plot_shg_stack.m           (堆叠图)
│   │   └── plot_shg_contour.m         (等高线图)
│   │
│   └── analysis/              # 🔬 数据分析
│       ├── normalize_shg_spectrum.m   (归一化处理)
│       └── shg_power_fitting.m        (功率拟合分析)
│
├── examples/                  # 📚 使用示例
│   ├── basic_plotting.m               (基础绘图示例)
│   └── power_dependence.m             (功率依赖性完整分析)
│
├── QUICKSTART.m               # 🚀 5分钟快速开始
└── README.md                  # 项目说明 (本文件)
```

---

## 快速开始

### 最简单的用法 - 绘制单条SHG光谱

```matlab
wavelength = 400:5:900;  % nm
spectrum = randn(1, 101);

[fig, ax] = plot_single_shg(spectrum, wavelength, ...
    'Title', 'My SHG Spectrum', ...
    'LineColor', 'r', ...
    'LineWidth', 2.5);
```

### 中级用法 - 功率扫描分析

```matlab
% 组织数据结构
powers = 0.1:0.1:2.0;
wavelength = 400:5:900;
data = randn(length(powers), length(wavelength));

SHGData.data = data;
SHGData.axisscale{1,1} = powers';
SHGData.axisscale{2,1} = wavelength';

% 绘制等高线图
plot_shg_contour(SHGData, 'Colormap', 'hot', 'Title', 'SHG Map');

% 绘制堆叠图
plot_shg_stack(SHGData, ...
    'NormalizeMethod', 'max', ...
    'Spacing', 0.15);
```

### 高级用法 - 功率依赖性拟合

```matlab
% 提取特定波长的数据
[~, idx] = min(abs(wavelength - 650));  % 650 nm
intensity = SHGData.data(:, idx);

% 进行幂律拟合: I ∝ P^n
[fitresult, gof] = shg_power_fitting(powers, intensity, ...
    'ModelType', 'power', ...
    'Plot', true);

% 结果: I = a × P^n
% 对于SHG: n ≈ 2.0
fprintf('模型: %s\n', fitresult.model);
fprintf('拟合优度: R² = %.6f\n', gof.rsquare);
```

---

## 数据结构

所有绘图和分析函数使用统一的 `SHGData` 结构：

```matlab
SHGData.data           % (N×M) 矩阵
                       % 行: 序列数 (功率/温度)
                       % 列: 波长点数
                       
SHGData.axisscale{1,1} % (N×1) 第一轴参数 (通常为功率 mW)
SHGData.axisscale{2,1} % (M×1) 波长 (nm)
```

---

## 各模块详解

### 📊 数据模块

| 函数 | 功能 | 返回值 |
|------|------|--------|
| `import_shg_data(filepath)` | 从单个.mat文件导入 | SHGData结构体 |
| `load_shg_power_data(shg_file, power_file)` | 加载功率标定数据 | SHGData结构体 |

### 📈 绘图模块

| 函数 | 用途 | 数据类型 | 输出 |
|------|------|---------|------|
| `plot_single_shg()` | 单条光谱 | 1D向量 | [fig, ax] |
| `plot_shg_stack()` | 多条堆叠 | 2D矩阵 + 参数轴 | [fig, ax] |
| `plot_shg_contour()` | 等高线图 | 2D矩阵 + 参数轴 | [fig, ax] |

### 🔬 分析模块

| 函数 | 功能 | 模型 |
|------|------|------|
| `normalize_shg_spectrum()` | 光谱归一化 | max/peak_range/area |
| `shg_power_fitting()` | 功率拟合 | I = a×P^n (幂律) |

---

## 常见参数

### plot_shg_stack 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `Indices` | 要绘制的索引 | 所有 |
| `WavelengthRange` | 波长范围 [min max] nm | 全部 |
| `NormalizeMethod` | 归一化: none/max/peak_range/area | 'max' |
| `PeakRange` | 峰值范围 [min max] nm | [300 800] |
| `Spacing` | 曲线间距 | 0.1 |
| `Colormap` | 颜色: power/index/jet/hot/cool | 'power' |

### plot_shg_contour 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `ContourLevels` | 等高线数 | 30 |
| `Colormap` | 颜色映射 | 'hot' |
| `ColorRange` | 强度范围 [min max] | 自动 |
| `ShowContourLines` | 显示等高线 | false |
| `ParamLabel` | 参数轴标签 | 'Parameter (a.u.)' |

### shg_power_fitting 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `ModelType` | 模型类型 | 'power' |
| `Plot` | 是否绘制拟合结果 | false |
| `InitialGuess` | 初始参数猜测 | 自动 |

---

## 工作流程

### 工作流1: 快速浏览 (~5分钟)
```
数据准备 → plot_single_shg() 或 plot_shg_stack() → 导出图像
```

### 工作流2: 整体分析 (~15分钟)
```
加载数据 → plot_shg_contour() → 整体观察
      → plot_shg_stack() → 细节对比
      → 调整参数 → 导出结果
```

### 工作流3: 深度分析 (~30分钟)
```
加载功率标定数据 → 多波长归一化
              → shg_power_fitting() → 幂律拟合
              → 提取n值 → 物理分析
              → 生成报告
```

---

## 参考示例

- **QUICKSTART.m** - 5分钟快速入门（推荐首先查看！）
- **basic_plotting.m** - 基础绘图示例（3个实用案例）
- **power_dependence.m** - 功率依赖性完整分析（包括拟合统计）

---

## 常见问题

### Q: 如何绘制特定波长范围内的数据?

```matlab
plot_shg_stack(SHGData, ...
    'WavelengthRange', [600 700]);  % 只显示600-700 nm
```

### Q: 如何调整图表的外观?

```matlab
plot_shg_stack(SHGData, ...
    'NormalizeMethod', 'peak_range', ...
    'Spacing', 0.2, ...
    'Colormap', 'jet', ...
    'FontSize', 14, ...
    'LabelFontSize', 16);
```

### Q: 如何保存图像?

```matlab
[fig, ax] = plot_single_shg(spectrum, wavelength);
savefig(fig, 'my_shg_spectrum.fig');
print(fig, 'my_shg_spectrum.png', '-dpng', '-r300');
```

### Q: SHG功率拟合中的幂指数应该是多少?

对于真正的二次谐波过程，幂指数应该接近 **2.0**。实际测量值可能因以下原因略有偏差：
- 非线性吸收
- 相位匹配条件
- 样品缺陷
- 实验噪声

R² > 0.99 表示良好的拟合质量。

---

## 与其他Analyzer的对比

| 项目 | PL-Analyzer | Raman-Analyzer | **SHG-Analyzer** |
|------|------------|-----------------|-----------------|
| X轴单位 | 波长 (nm) | 拉曼位移 (cm⁻¹) | **波长 (nm)** |
| 标准范围 | 400-1000 nm | 0-3000 cm⁻¹ | **400-900 nm** |
| 主要应用 | 荧光发射 | 拉曼散射 | **非线性光学** |
| 关键分析 | 峰值拟合 | 频率分析 | **功率依赖性** |
| 理论关系 | 线性 | 线性 | **I ∝ P^2 (非线性)** |

---

## 版本信息

- **版本**: 1.0
- **创建日期**: 2026-01-24
- **基于**: PL-Analyzer v2.0 架构
- **MATLAB 版本要求**: R2018b 及以上
- **依赖库**: MATLAB Image Processing Toolbox (可选)

---

## 许可证

该项目遵循与 PL-Analyzer 和 Raman-Analyzer 相同的许可协议。

---

## 联系与反馈

如有问题或建议，请参考以下文件：
- `QUICKSTART.m` - 快速入门
- `examples/basic_plotting.m` - 基础示例
- 各函数的内嵌帮助文档

---

**推荐下一步**:
1. 运行 `QUICKSTART.m` 了解项目概况
2. 查看 `examples/basic_plotting.m` 学习基本用法
3. 运行 `examples/power_dependence.m` 进行功率拟合分析
4. 参考文档编写自己的分析脚本

**祝你使用愉快！🎉**
