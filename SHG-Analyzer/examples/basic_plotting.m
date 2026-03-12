%% SHG基础绘图示例
% 演示如何绘制单条SHG光谱和SHG堆叠图

clear; clc; close all;
addpath(genpath('.'));

fprintf('=== SHG光谱基础绘图示例 ===\n\n');

%% 示例1: 绘制单条SHG光谱
fprintf('Step 1: 单条SHG光谱绘制\n');
fprintf('-------------------------------------\n');

wavelength = 400:5:900;  % nm
% 模拟SHG光谱 (中心在650 nm)
spectrum1 = 800 * exp(-((wavelength - 650).^2) / 10000) + ...
           randn(size(wavelength)) * 20;

% 确保强度非负
spectrum1 = max(spectrum1, 0);

[fig1, ax1] = plot_single_shg(spectrum1, wavelength, ...
    'Title', 'Single SHG Spectrum (650 nm)', ...
    'LineColor', 'r', ...
    'LineWidth', 2.5);

fprintf('✓ 绘制了第一个SHG光谱！\n\n');

%% 示例2: 创建功率依赖的SHG数据
fprintf('Step 2: 创建多条SHG光谱数据集(功率扫描)\n');
fprintf('-------------------------------------\n');

% 模拟功率扫描的SHG数据
powers = 0.05:0.05:1.0;  % mW
wavelength = 400:5:900;

% 创建数据矩阵
spectra = zeros(length(powers), length(wavelength));
for i = 1:length(powers)
    % SHG强度随功率变化: I ∝ P^2
    powerFactor = powers(i)^2;
    spectra(i, :) = powerFactor * 800 * exp(-((wavelength - 650).^2) / 10000) + ...
                    randn(1, length(wavelength)) * 20;
end

% 确保强度非负
spectra = max(spectra, 0);

% 按照标准格式组织数据
SHGData.data = spectra;
SHGData.axisscale{1,1} = powers';         % 功率轴
SHGData.axisscale{2,1} = wavelength';     % 波长轴

fprintf('✓ 创建了 %d 条SHG光谱数据集\n\n', length(powers));

%% 示例3: 可视化整个数据集
fprintf('Step 3: 可视化整个数据集\n');
fprintf('-------------------------------------\n');

% 3.1 等高线图 - 整体浏览
figure('Position', [100 100 900 600]);
[fig2, ax2] = plot_shg_contour(SHGData, ...
    'ContourLevels', 50, ...
    'Colormap', 'hot', ...
    'Title', 'SHG Intensity Map (Power Scan)', ...
    'ParamLabel', 'Power (mW)');

fprintf('  ├─ ✓ 绘制等高线图 (可视化整体分布)\n');

% 3.2 堆叠图 - 细节对比
figure('Position', [1050 100 900 600]);
indices = 1:2:length(powers);  % 每隔2条
[fig3, ax3] = plot_shg_stack(SHGData, ...
    'Indices', indices, ...
    'NormalizeMethod', 'max', ...
    'Spacing', 0.15, ...
    'Colormap', 'power', ...
    'Title', 'SHG Spectra Evolution with Power');

fprintf('  └─ ✓ 绘制堆叠图 (观察谱线演变)\n\n');

%% 示例4: 不同的归一化方法
fprintf('Step 4: 比较不同的归一化方法\n');
fprintf('-------------------------------------\n');

figure('Position', [100 700 1200 400]);

norm_methods = {'max', 'peak_range', 'area'};
for i = 1:length(norm_methods)
    ax = subplot(1, 3, i);
    
    data_norm = normalize_shg_spectrum(SHGData.data, norm_methods{i}, ...
        'Wavelength', wavelength', ...
        'PeakRange', [600 700]);
    
    SHGData_temp = SHGData;
    SHGData_temp.data = data_norm;
    
    indices = 1:3:length(powers);
    
    hold(ax, 'on');
    for idx = indices
        plot(ax, wavelength, data_norm(idx, :), 'LineWidth', 2);
    end
    hold(ax, 'off');
    
    xlabel(ax, 'Wavelength (nm)', 'FontSize', 11);
    ylabel(ax, 'Normalized SHG Intensity', 'FontSize', 11);
    title(ax, sprintf('Normalization: %s', norm_methods{i}), 'FontSize', 12);
    grid(ax, 'on');
    set(ax, 'GridAlpha', 0.3);
end

fprintf('✓ 比较了不同的归一化方法！\n\n');

%% 示例5: 功率依赖性拟合
fprintf('Step 5: 功率依赖性分析\n');
fprintf('-------------------------------------\n');

% 选择中心波长处的强度进行拟合
[~, center_idx] = min(abs(wavelength - 650));
center_intensity = SHGData.data(:, center_idx);

% 进行幂律拟合
[fitresult, gof] = shg_power_fitting(powers, center_intensity, 'Plot', true);

fprintf('✓ 功率依赖性拟合完成！\n\n');

%% 示例6: 工作流总结
fprintf('=====================================\n');
fprintf('📌 常见使用模式\n');
fprintf('=====================================\n\n');

fprintf('【模式1】绘制单条SHG光谱\n');
fprintf('  plot_single_shg(spectrum, wavelength, ''Title'', ''My SHG'');\n\n');

fprintf('【模式2】功率扫描分析\n');
fprintf('  SHGData = import_shg_data("shg.mat");\n');
fprintf('  plot_shg_contour(SHGData);        %% 等高线\n');
fprintf('  plot_shg_stack(SHGData);          %% 堆叠\n\n');

fprintf('【模式3】功率依赖性拟合\n');
fprintf('  [fitresult, gof] = shg_power_fitting(power, intensity);\n\n');

fprintf('【模式4】指定波长范围绘图\n');
fprintf('  plot_shg_stack(SHGData, ...\n');
fprintf('      ''WavelengthRange'', [600 700], ...\n');
fprintf('      ''NormalizeMethod'', ''peak_range'');\n\n');

fprintf('✅ SHG基础绘图示例完成！\n');
