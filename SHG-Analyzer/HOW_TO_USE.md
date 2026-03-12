# 如何使用 SHG-Analyzer

## 📌 概览

SHG-Analyzer 是一个**独立的脚本包**，基于 PL-Analyzer 和 Raman-Analyzer 的架构，专门为**二次谐波(SHG)光谱**的绘制和分析而设计。

现在你已经拥有了一套**完整的、可直接使用的工具集**。

---

## 🚀 立即开始 (3步)

### 第1步：了解项目 (5分钟)
在 MATLAB 中打开并运行：
```matlab
QUICKSTART
```

这个脚本会：
- 给你一个项目的快速概览
- 展示4个实用示例
- 介绍核心模块
- 说明使用模式

### 第2步：学习基础用法 (20分钟)
查看并运行：
```matlab
cd examples
basic_plotting
```

这个示例演示了：
- 如何绘制单条SHG光谱
- 如何处理功率扫描数据
- 如何使用等高线图和堆叠图
- 不同的归一化方法

### 第3步：尝试功率分析 (20分钟)
查看并运行：
```matlab
power_dependence
```

这个高级示例演示了：
- 完整的功率扫描数据分析
- SHG功率拟合 (I ∝ P^n)
- 多波长分析
- 拟合质量评估

---

## 💾 使用自己的数据

### 方案A：直接使用绘图函数

```matlab
% 加载你的SHG光谱数据
load('my_shg_data.mat');  % 假设变量名为 'spectrum' 和 'wavelength'

% 绘制单条光谱
plot_single_shg(spectrum, wavelength, ...
    'Title', 'My SHG Spectrum', ...
    'LineColor', 'r');

% 保存图像
savefig(gcf, 'my_spectrum.fig');
print(gcf, 'my_spectrum.png', '-dpng', '-r300');
```

### 方案B：功率扫描数据分析

```matlab
% 加载功率扫描数据
load('my_shg_power_scan.mat');  
% 假设:
%   data 是 (N×M) 矩阵
%   powers 是 (N×1) 功率向量 (mW)
%   wavelength 是 (M×1) 波长向量 (nm)

% 组织成标准格式
SHGData.data = data;
SHGData.axisscale{1,1} = powers;
SHGData.axisscale{2,1} = wavelength;

% 绘制等高线热图
figure('Position', [100 100 900 600]);
plot_shg_contour(SHGData, 'Colormap', 'hot', 'Title', 'My Data');

% 绘制堆叠图
figure('Position', [1050 100 900 600]);
plot_shg_stack(SHGData, ...
    'NormalizeMethod', 'max', ...
    'Spacing', 0.15);
```

### 方案C：功率依赖性分析

```matlab
% 使用上面的 SHGData

% 提取特定波长的数据
target_wavelength = 650;  % nm
[~, idx] = min(abs(wavelength - target_wavelength));
intensity = data(:, idx);

% 进行幂律拟合
[fitresult, gof] = shg_power_fitting(powers, intensity, ...
    'ModelType', 'power', ...
    'Plot', true);

% 查看结果
fprintf('\n=== SHG Power Fitting Results ===\n');
fprintf('Model: %s\n', fitresult.model);
fprintf('Power exponent: %.4f (expected ~2.0)\n', fitresult.exponent);
fprintf('Fitting quality (R²): %.6f\n', gof.rsquare);
fprintf('RMSE: %.6f\n', gof.rmse);
```

---

## 📚 核心函数速查

### 绘图函数

#### 1️⃣ `plot_single_shg()` - 单条光谱
**用途**: 绘制单条SHG光谱

```matlab
[fig, ax] = plot_single_shg(spectrum, wavelength, ...
    'Title', 'My Spectrum', ...
    'LineColor', 'r', ...
    'LineWidth', 2.5, ...
    'LogScale', false);
```

**适用场景**: 快速查看单条光谱

---

#### 2️⃣ `plot_shg_stack()` - 堆叠图
**用途**: 多条光谱堆叠可视化 (最常用)

```matlab
plot_shg_stack(SHGData, ...
    'Indices', 1:3:20, ...  % 选择行索引
    'WavelengthRange', [600 700], ...  % 波长范围
    'NormalizeMethod', 'peak_range', ...  % 归一化
    'Spacing', 0.15, ...  % 曲线间距
    'Colormap', 'power');  % 颜色
```

**适用场景**: 观察光谱随参数的演变

---

#### 3️⃣ `plot_shg_contour()` - 等高线图
**用途**: 二维热图展示

```matlab
plot_shg_contour(SHGData, ...
    'ContourLevels', 50, ...  % 等高线数
    'Colormap', 'hot', ...
    'ColorRange', [], ...  % 自动范围
    'ShowContourLines', false);  % 显示等高线
```

**适用场景**: 整体观察数据分布

---

### 数据处理函数

#### 4️⃣ `import_shg_data()` - 导入数据
```matlab
SHGData = import_shg_data('my_data.mat');
```

#### 5️⃣ `load_shg_power_data()` - 加载功率数据
```matlab
SHGData = load_shg_power_data('shg.mat', 'power.mat');
```

#### 6️⃣ `normalize_shg_spectrum()` - 归一化
```matlab
spectrum_norm = normalize_shg_spectrum(spectrum, 'max');
% 支持: 'max', 'peak_range', 'area', 'none'
```

---

### 分析函数

#### 7️⃣ `shg_power_fitting()` - 功率拟合 ⭐ 重点
**用途**: 分析SHG强度随功率变化的规律

```matlab
[fitresult, gof] = shg_power_fitting(power, intensity, ...
    'ModelType', 'power', ...
    'Plot', true);

% 输出:
% fitresult.model = 'I = 0.5 × P^2.01'
% fitresult.exponent = 2.01
% gof.rsquare = 0.9987
% gof.rmse = 0.0123
```

**关键点**:
- 幂律模型: I = a × P^n
- 对于SHG: n ≈ 2.0
- R² > 0.99: 优秀拟合

---

## 🎯 典型工作流程

### 工作流1: 快速绘图 (5分钟)

```matlab
% 加载单条光谱
load('single_spectrum.mat', 'spectrum', 'wavelength');

% 一行代码绘制
plot_single_shg(spectrum, wavelength, 'Title', 'My SHG');

% 保存
print(gcf, 'result.png', '-dpng', '-r300');
```

---

### 工作流2: 数据浏览 (15分钟)

```matlab
% 加载功率扫描数据
load('power_scan.mat');  % data, powers, wavelength

% 组织数据
SHGData.data = data;
SHGData.axisscale{1,1} = powers;
SHGData.axisscale{2,1} = wavelength;

% 创建两个图表
figure('Position', [100 100 1000 600]);

% 热图
subplot(1,2,1);
plot_shg_contour(SHGData, 'Colormap', 'hot');

% 堆叠
subplot(1,2,2);
plot_shg_stack(SHGData, 'NormalizeMethod', 'max');

% 保存
savefig(gcf, 'overview.fig');
```

---

### 工作流3: 深度分析 (30分钟)

```matlab
% 加载功率标定数据
load('power_scan.mat');  % data, powers, wavelength

% 组织数据
SHGData.data = data;
SHGData.axisscale{1,1} = powers;
SHGData.axisscale{2,1} = wavelength;

% 步骤1: 整体浏览
figure;
plot_shg_contour(SHGData, 'Title', 'SHG Map');

% 步骤2: 选择分析波长
target_wl = 650;  % nm
[~, idx] = min(abs(wavelength - target_wl));
intensity_at_target = data(:, idx);

% 步骤3: 功率拟合
[fitresult, gof] = shg_power_fitting(powers, intensity_at_target, 'Plot', true);

% 步骤4: 提取参数
fprintf('幂指数: %.4f\n', fitresult.exponent);
fprintf('拟合优度: R² = %.6f\n', gof.rsquare);

% 步骤5: 多波长分析
wavelengths_to_test = 600:20:700;
exponents = [];

for wl = wavelengths_to_test
    [~, wl_idx] = min(abs(wavelength - wl));
    intensity = data(:, wl_idx);
    [fr, ~] = shg_power_fitting(powers, intensity);
    exponents = [exponents; fr.exponent];
end

% 绘制结果
figure;
plot(wavelengths_to_test, exponents, 'o-', 'LineWidth', 2);
xlabel('Wavelength (nm)');
ylabel('Power Exponent');
title('SHG Exponent vs Wavelength');
```

---

## 📖 文档导航

| 文档 | 内容 | 阅读时间 |
|------|------|---------|
| **QUICKSTART.m** | 快速入门 | 5分钟 |
| **README.md** | 详细说明 | 15分钟 |
| **USAGE_GUIDE.txt** | 使用指南 | 10分钟 |
| **examples/basic_plotting.m** | 基础示例 | 20分钟 |
| **examples/power_dependence.m** | 深度示例 | 30分钟 |

**推荐阅读顺序**:
1. 本文件 (HOW_TO_USE.md)
2. QUICKSTART.m
3. examples/basic_plotting.m
4. README.md (参考)

---

## 🔧 常见任务

### 任务1: 绘制单条光谱

```matlab
wavelength = 400:900;
spectrum = randn(1, 501);
plot_single_shg(spectrum, wavelength, 'Title', 'My Spectrum');
```

### 任务2: 比较两条光谱

```matlab
figure;
plot(wavelength, spectrum1, 'r', 'LineWidth', 2); hold on;
plot(wavelength, spectrum2, 'b', 'LineWidth', 2);
xlabel('Wavelength (nm)');
ylabel('Intensity (a.u.)');
legend('Spectrum 1', 'Spectrum 2');
```

### 任务3: 绘制功率扫描

```matlab
plot_shg_contour(SHGData);  % 热图
figure;
plot_shg_stack(SHGData);    % 堆叠图
```

### 任务4: 提取幂指数

```matlab
[~, idx] = min(abs(wavelength - 650));
intensity = data(:, idx);
[fitresult, gof] = shg_power_fitting(powers, intensity);
exponent = fitresult.exponent;  % 应接近 2.0
```

### 任务5: 批量处理多个文件

```matlab
files = dir('*.mat');
results = [];

for i = 1:length(files)
    data = load(files(i).name);
    [fitresult, gof] = shg_power_fitting(data.powers, data.intensity);
    results(i) = fitresult.exponent;
end

plot(results, 'o-');
```

---

## 💡 提示和技巧

### 💡 技巧1: 快速调整图表外观

```matlab
% 改变线宽
plot_shg_stack(SHGData, ...
    'FontSize', 14, ...
    'LabelFontSize', 16);
```

### 💡 技巧2: 聚焦特定范围

```matlab
% 只显示650nm附近
plot_shg_stack(SHGData, ...
    'WavelengthRange', [600 700], ...
    'NormalizeMethod', 'peak_range', ...
    'PeakRange', [625 675]);
```

### 💡 技巧3: 归一化处理

```matlab
% 4种归一化方法
plot_shg_stack(SHGData, 'NormalizeMethod', 'max');       % 最大值
plot_shg_stack(SHGData, 'NormalizeMethod', 'peak_range'); % 峰值范围
plot_shg_stack(SHGData, 'NormalizeMethod', 'area');      % 面积
plot_shg_stack(SHGData, 'NormalizeMethod', 'none');      % 无
```

### 💡 技巧4: 保存高质量图像

```matlab
% PNG格式 (300 DPI)
print(gcf, 'figure.png', '-dpng', '-r300');

% EPS格式 (出版级)
print(gcf, 'figure.eps', '-depsc', '-r300');

% PDF格式
print(gcf, 'figure.pdf', '-dpdf', '-r300');
```

### 💡 技巧5: 批量图表创建

```matlab
for i = 1:10
    figure;
    plot_single_shg(data(i,:), wavelength, ...
        'Title', sprintf('Spectrum %d', i));
    savefig(gcf, sprintf('spectrum_%d.fig', i));
    close;
end
```

---

## ⚠️ 常见问题解决

### Q: 数据形状错误怎么办?

**问题**: 函数报错说矩阵维度不匹配

**解决**:
```matlab
% 确保数据格式正确
[m, n] = size(data);  % m: 序列数, n: 波长点数

SHGData.data = data;           % (m×n)
SHGData.axisscale{1,1} = powers;  % (m×1)
SHGData.axisscale{2,1} = wavelength;  % (n×1)
```

---

### Q: 拟合失败怎么办?

**问题**: shg_power_fitting 返回 NaN

**解决**:
```matlab
% 检查数据
power_test = 0.1:0.1:2.0;
intensity_test = randn(1, 20) + power_test.^2;

% 确保有足够数据点 (至少5个)
% 确保强度和功率都是正数
% 使用 'Plot', true 查看拟合过程
[fit, gof] = shg_power_fitting(power_test, intensity_test, 'Plot', true);
```

---

### Q: 图表显示不完整怎么办?

**问题**: 某些标签或图例显示不全

**解决**:
```matlab
% 调整图窗大小
figure('Position', [100 100 1000 700]);

% 或使用 tiledlayout
t = tiledlayout(2, 2);
ax1 = nexttile;
plot_shg_stack(SHGData);  % 会使用当前坐标轴
```

---

## 🎓 进阶使用

### 自定义绘图函数

```matlab
function plot_my_shg(SHGData)
    figure('Position', [100 100 1200 400]);
    
    subplot(1, 3, 1);
    plot_shg_contour(SHGData);
    
    subplot(1, 3, 2);
    plot_shg_stack(SHGData);
    
    subplot(1, 3, 3);
    % 自己添加分析...
end

% 使用
plot_my_shg(SHGData);
```

### 集成到自己的分析流程

```matlab
% 你的数据处理流程
my_raw_data = load_raw_spectra('folder/');
my_processed_data = preprocess(my_raw_data);

% 转换为 SHGData 格式
SHGData.data = my_processed_data.spectra;
SHGData.axisscale{1,1} = my_processed_data.params;
SHGData.axisscale{2,1} = my_processed_data.wavelength;

% 使用 SHG-Analyzer
plot_shg_contour(SHGData);
```

---

## 🚀 下一步

### 立即体验:
1. 运行 `QUICKSTART.m`
2. 查看 `examples/basic_plotting.m`
3. 用你的数据试试

### 深入学习:
1. 查阅 `README.md`
2. 研究 `examples/power_dependence.m`
3. 修改代码进行实验

### 寻求帮助:
1. 查看函数帮助: `help plot_shg_stack`
2. 参考示例代码
3. 检查数据格式

---

## 📞 快速参考

| 需求 | 使用函数 | 参数 |
|------|---------|------|
| 绘制单条光谱 | `plot_single_shg()` | wavelength, spectrum |
| 绘制多条堆叠 | `plot_shg_stack()` | SHGData, NormalizeMethod |
| 绘制热图 | `plot_shg_contour()` | SHGData, Colormap |
| 功率拟合 | `shg_power_fitting()` | power, intensity |
| 归一化处理 | `normalize_shg_spectrum()` | spectrum, method |

---

**祝你使用愉快！** 🎉

有任何问题，参考文档或示例代码。
