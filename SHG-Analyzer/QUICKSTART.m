%% SHG-Analyzer 快速开始指南
% 本脚本演示SHG-Analyzer的基本使用方法

clear; clc; close all;
addpath(genpath('.'));

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║         SHG-Analyzer 快速开始指南 - Quick Start Guide          ║\n');
fprintf('║         Version 1.0 - 基于 PL-Analyzer 架构改进               ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% 快速使用演示
fprintf('📌 五分钟快速入门\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

%% 场景1: 绘制单条SHG光谱
fprintf('【场景1】绘制单条SHG光谱 - 最简单的用法\n');
fprintf('───────────────────────────────────────\n');

% 创建示例数据
wavelength = 400:5:900;
spectrum = 500 * exp(-((wavelength - 650).^2) / 10000) + randn(size(wavelength)) * 20;
spectrum = max(spectrum, 0);

% 绘制
[fig1, ax1] = plot_single_shg(spectrum, wavelength, ...
    'Title', 'My First SHG Spectrum', ...
    'LineColor', 'r', ...
    'LineWidth', 2.5);

fprintf('✓ 成功绘制单条SHG光谱！\n');
fprintf('  代码只需3行:\n');
fprintf('    wavelength = 400:5:900;\n');
fprintf('    spectrum = randn(1, 101);\n');
fprintf('    plot_single_shg(spectrum, wavelength, ''Title'', ''Sample'');\n\n');

%% 场景2: 功率扫描数据分析
fprintf('【场景2】功率扫描数据分析 - 最常见的用法\n');
fprintf('───────────────────────────────────────\n');

% 创建模拟功率扫描数据
powers = 0.05:0.05:1.0;
wavelength = 400:5:900;

spectra = zeros(length(powers), length(wavelength));
for i = 1:length(powers)
    powerFactor = powers(i)^2;
    spectra(i, :) = powerFactor * 500 * exp(-((wavelength - 650).^2) / 10000) + ...
                    randn(1, length(wavelength)) * 20;
end
spectra = max(spectra, 0);

% 组织数据结构
SHGData.data = spectra;
SHGData.axisscale{1,1} = powers';
SHGData.axisscale{2,1} = wavelength';

fprintf('✓ 创建了功率扫描数据集 (%d条谱线)\n', length(powers));

% 等高线图
figure('Position', [100 100 1000 400]);
subplot(1, 2, 1);
plot_shg_contour(SHGData, 'Colormap', 'hot', 'Title', 'Heatmap View');
fprintf('✓ 绘制等高线热图\n');

% 堆叠图
subplot(1, 2, 2);
ax = gca;
indices = 1:3:length(powers);
data_norm = normalize_shg_spectrum(SHGData.data, 'max');
hold(ax, 'on');
colors = jet(length(indices));
for i = 1:length(indices)
    idx = indices(i);
    y_offset = (i - 1) * 0.15;
    plot(ax, wavelength, data_norm(idx, :) + y_offset, ...
        'Color', colors(i, :), 'LineWidth', 2);
end
hold(ax, 'off');
xlabel(ax, 'Wavelength (nm)');
ylabel(ax, 'Normalized Intensity');
title(ax, 'Stack View');
grid(ax, 'on');
fprintf('✓ 绘制堆叠图\n\n');

%% 场景3: 功率依赖性分析
fprintf('【场景3】功率依赖性分析 - 最实用的分析\n');
fprintf('───────────────────────────────────────\n');

% 提取中心波长处的数据
[~, center_idx] = min(abs(wavelength - 650));
center_intensity = SHGData.data(:, center_idx);

% 进行幂律拟合
[fitresult, gof] = shg_power_fitting(powers, center_intensity);

fprintf('✓ 完成功率依赖性拟合\n\n');

%% 核心模块介绍
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('🎯 SHG-Analyzer 核心模块\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('📊 绘图模块 (visualization/)\n');
fprintf('  ├─ plot_single_shg()      - 单条光谱绘制 ⭐ 最基础\n');
fprintf('  ├─ plot_shg_stack()       - 堆叠图绘制\n');
fprintf('  └─ plot_shg_contour()     - 等高线热图\n\n');

fprintf('📂 数据模块 (data/)\n');
fprintf('  ├─ import_shg_data()      - 导入单个.mat文件\n');
fprintf('  └─ load_shg_power_data()  - 加载功率扫描数据\n\n');

fprintf('🔬 分析模块 (analysis/)\n');
fprintf('  ├─ normalize_shg_spectrum() - 光谱归一化\n');
fprintf('  └─ shg_power_fitting()      - 功率拟合分析\n\n');

%% 常用参数速查表
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('📋 常用参数速查表\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('plot_shg_stack 常用参数:\n');
fprintf('  NormalizeMethod  - 归一化方法: ''max''/''peak_range''/''area''\n');
fprintf('  Spacing          - 曲线间距，范围 0.1-0.3\n');
fprintf('  Colormap         - 颜色: ''power''/''jet''/''hot''/''cool''\n');
fprintf('  Indices          - 要绘制的行索引，默认全部\n');
fprintf('  WavelengthRange  - 波长范围 [min max] nm\n\n');

fprintf('plot_shg_contour 常用参数:\n');
fprintf('  Colormap         - 颜色: ''hot''/''jet''/''cool''/''parula''\n');
fprintf('  ContourLevels    - 等高线数，通常 20-50\n');
fprintf('  ColorRange       - 强度范围 [min max]，默认自动\n');
fprintf('  ShowContourLines - 是否显示等高线，默认 false\n\n');

%% 三种典型工作流程
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('🔄 三种典型工作流程\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('⚡ 工作流1: 快速浏览 (~5分钟)\n');
fprintf('   目标: 快速查看单条或多条SHG光谱\n');
fprintf('   步骤:\n');
fprintf('     1. plot_single_shg() 或 plot_shg_stack()\n');
fprintf('     2. 调整参数\n');
fprintf('     3. 导出图像\n\n');

fprintf('📊 工作流2: 整体分析 (~15分钟)\n');
fprintf('   目标: 分析功率扫描数据的整体特性\n');
fprintf('   步骤:\n');
fprintf('     1. import_shg_data() 加载数据\n');
fprintf('     2. plot_shg_contour() 查看整体分布\n');
fprintf('     3. plot_shg_stack() 查看细节变化\n\n');

fprintf('🔬 工作流3: 深度分析 (~30分钟)\n');
fprintf('   目标: 深入研究功率依赖性和谱线特性\n');
fprintf('   步骤:\n');
fprintf('     1. load_shg_power_data() 加载功率标定数据\n');
fprintf('     2. 多波长归一化处理\n');
fprintf('     3. shg_power_fitting() 进行幂律拟合\n');
fprintf('     4. 分析和统计结果\n\n');

%% 数据格式说明
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('📐 标准数据结构 (SHGData)\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('SHGData 是一个结构体，包含以下字段:\n\n');
fprintf('  SHGData.data           % (N×M) 矩阵\n');
fprintf('    ├─ 行数 (N)    : 序列数 (如功率点数)\n');
fprintf('    └─ 列数 (M)    : 波长点数\n\n');

fprintf('  SHGData.axisscale{1,1} % (N×1) 第一轴参数\n');
fprintf('    └─ 通常为功率 (mW) 或其他扫描参数\n\n');

fprintf('  SHGData.axisscale{2,1} % (M×1) 波长\n');
fprintf('    └─ 波长 (nm)，从短到长排列\n\n');

fprintf('示例:\n');
fprintf('  powers = 0.1:0.1:2.0;              %% 20个功率点\n');
fprintf('  wavelength = 400:5:900;            %% 101个波长点\n');
fprintf('  data = randn(20, 101);             %% 20×101 矩阵\n');
fprintf('  \n');
fprintf('  SHGData.data = data;\n');
fprintf('  SHGData.axisscale{1,1} = powers'';\n');
fprintf('  SHGData.axisscale{2,1} = wavelength'';\n\n');

%% 查看详细示例
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('📚 更多示例和高级用法\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('查看以下文件获得详细的示例:\n\n');
fprintf('  examples/basic_plotting.m       - 基础绘图示例 (推荐首先查看)\n');
fprintf('  examples/power_dependence.m     - 功率依赖性完整分析\n\n');

fprintf('✅ SHG-Analyzer 快速开始完成!\n');
fprintf('   现在你可以:\n');
fprintf('   • 运行 basic_plotting.m 学习基础用法\n');
fprintf('   • 运行 power_dependence.m 学习高级分析\n');
fprintf('   • 参考本文件的代码编写自己的脚本\n\n');

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('有问题? 查看各函数的帮助文档:\n');
fprintf('  help plot_single_shg\n');
fprintf('  help plot_shg_stack\n');
fprintf('  help plot_shg_contour\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');
