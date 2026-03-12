%% 拉曼基础绘图示例
% 演示如何绘制单条拉曼光谱和拉曼堆叠图

clear; clc; close all;
addpath(genpath('.'));

fprintf('=== 拉曼光谱基础绘图示例 ===\n\n');

%% 示例1: 绘制单条拉曼光谱
fprintf('Step 1: 单条拉曼光谱绘制\n');
fprintf('-------------------------------------\n');

ramanShift = 0:3000;  % cm⁻¹
% 模拟拉曼光谱 (D峰 @1350cm⁻¹, G峰 @1580cm⁻¹)
spectrum1 = 500 * exp(-((ramanShift - 1350).^2) / 20000) + ...  % D peak
           1000 * exp(-((ramanShift - 1580).^2) / 30000) + ...  % G peak
           randn(size(ramanShift)) * 30;

[fig1, ax1] = plot_single_raman(spectrum1, ramanShift, ...
    'Title', 'Single Raman Spectrum', ...
    'LineColor', 'b', ...
    'LineWidth', 2.5);

fprintf('✓ 绘制了第一个拉曼光谱！\n\n');

%% 示例2: 创建功率依赖的拉曼数据
fprintf('Step 2: 创建多条拉曼光谱数据集\n');
fprintf('-------------------------------------\n');

% 模拟功率扫描的拉曼数据
powers = 0.1:0.1:2.0;
ramanShift = 0:3000;

% 创建数据矩阵
spectra = zeros(length(powers), length(ramanShift));
for i = 1:length(powers)
    % 强度随功率增加 (I ∝ P^n)
    powerFactor = powers(i)^1.2;
    spectra(i, :) = powerFactor * 500 * exp(-((ramanShift - 1350).^2) / 20000) + ...
                    powerFactor * 1000 * exp(-((ramanShift - 1580).^2) / 30000) + ...
                    randn(1, length(ramanShift)) * 30;
end

% 按照标准格式组织数据
RamanData.data = spectra;
RamanData.axisscale{1,1} = powers';         % 功率轴
RamanData.axisscale{2,1} = ramanShift';     % 拉曼位移轴

fprintf('✓ 创建了 %d 条拉曼光谱数据集\n\n', length(powers));

%% 示例3: 可视化整个数据集
fprintf('Step 3: 可视化整个数据集\n');
fprintf('-------------------------------------\n');

% 3.1 等高线图 - 整体浏览
figure('Position', [100 100 900 600]);
[fig2, ax2] = plot_raman_contour(RamanData, ...
    'ColorRange', [0.5 1.0], ...
    'ContourLevels', 50, ...
    'Colormap', 'hot', ...
    'Title', 'Raman Intensity Map (Power Scan)');

fprintf('  ├─ ✓ 绘制等高线图 (可视化整体分布)\n');

% 3.2 堆叠图 - 细节对比
figure('Position', [1050 100 900 600]);
indices = 1:3:length(powers);  % 每隔3条
[fig3, ax3] = plot_raman_stack(RamanData, ...
    'Indices', indices, ...
    'NormalizeMethod', 'max', ...
    'Spacing', 0.2, ...
    'Colormap', 'power', ...
    'Title', 'Raman Spectra Evolution with Power');

fprintf('  └─ ✓ 绘制堆叠图 (观察谱线演变)\n\n');

%% 示例4: 高级用法 - 指定拉曼范围
fprintf('Step 4: 高级用法 - 指定拉曼范围\n');
fprintf('-------------------------------------\n');

figure('Position', [100 700 900 600]);
[fig4, ax4] = plot_raman_stack(RamanData, ...
    'Indices', 1:4:length(powers), ...
    'RamanRange', [1200 1700], ...  % 只显示D和G峰区域
    'NormalizeMethod', 'peak_range', ...
    'PeakRange', [1300 1600], ...
    'Spacing', 0.15, ...
    'Colormap', 'jet', ...
    'Title', 'Raman Spectra - D and G Band Region');

fprintf('✓ 高级绘图示例完成！\n\n');

%% 示例5: 工作流总结
fprintf('=====================================\n');
fprintf('📌 常见使用模式\n');
fprintf('=====================================\n\n');

fprintf('【模式1】绘制单条拉曼光谱\n');
fprintf('  RamanData = import_raman_data("raman.mat");\n');
fprintf('  plot_single_raman(RamanData.data(1,:), RamanData.axisscale{2,1});\n\n');

fprintf('【模式2】功率扫描分析\n');
fprintf('  RamanData = load_raman_power_data("raman.mat", "power.mat");\n');
fprintf('  plot_raman_contour(RamanData);        %% 等高线\n');
fprintf('  plot_raman_stack(RamanData);          %% 堆叠\n\n');

fprintf('【模式3】指定范围绘图\n');
fprintf('  plot_raman_stack(RamanData, ...\n');
fprintf('      ''RamanRange'', [1200 1700], ...\n');
fprintf('      ''NormalizeMethod'', ''peak_range'');\n\n');

fprintf('✅ 拉曼基础绘图示例完成！\n');
