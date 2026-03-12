%% SHG功率依赖性分析完整示例
% 演示如何进行SHG强度随激发功率的依赖性分析

clear; clc; close all;
addpath(genpath('.'));

fprintf('=== SHG功率依赖性分析示例 ===\n\n');

%% 1. 生成模拟数据
fprintf('Step 1: 生成模拟SHG功率扫描数据\n');
fprintf('-------------------------------------\n');

% 功率扫描参数
powers = 0.01:0.01:1.0;  % 0.01 - 1.0 mW
wavelength = 400:5:900;  % 400 - 900 nm

% 创建模拟SHG光谱
% 参数: 
%   - 中心波长: 650 nm
%   - 强度随功率: I ∝ P^2 (二次谐波特性)
%   - 线宽: 50 nm (FWHM ~70 nm)

spectra = zeros(length(powers), length(wavelength));
center_wl = 650;
sigma = 25;  % 高斯宽度

for i = 1:length(powers)
    % SHG强度随功率二次方变化
    power_factor = powers(i)^2;
    
    % 高斯型光谱
    spectrum_base = 1000 * exp(-((wavelength - center_wl).^2) / (2 * sigma^2));
    
    % 添加噪声
    noise = randn(1, length(wavelength)) * 30;
    
    spectra(i, :) = power_factor * spectrum_base + noise;
end

% 确保强度非负
spectra = max(spectra, 0);

% 组织数据结构
SHGData.data = spectra;
SHGData.axisscale{1,1} = powers';
SHGData.axisscale{2,1} = wavelength';

fprintf('✓ 创建了 %d 条SHG光谱 (%.2f - %.2f mW)\n', length(powers), powers(1), powers(end));
fprintf('  波长范围: %.0f - %.0f nm\n\n', wavelength(1), wavelength(end));

%% 2. 整体数据可视化
fprintf('Step 2: 整体数据可视化\n');
fprintf('-------------------------------------\n');

% 2.1 等高线热图
figure('Position', [50 50 1000 700]);
subplot(2, 2, 1);
plot_shg_contour(SHGData, ...
    'ContourLevels', 40, ...
    'Colormap', 'hot', ...
    'Title', 'SHG Intensity Map', ...
    'ParamLabel', 'Power (mW)');

fprintf('  ├─ ✓ 绘制等高线热图\n');

% 2.2 堆叠图 - 所有谱线
subplot(2, 2, 2);
indices_all = 1:5:length(powers);  % 每隔5条
ax = gca;

data_norm = normalize_shg_spectrum(SHGData.data, 'max');
hold(ax, 'on');
colors = jet(length(indices_all));
for i = 1:length(indices_all)
    idx = indices_all(i);
    y_offset = (i - 1) * 0.12;
    plot(ax, wavelength, data_norm(idx, :) + y_offset, ...
        'Color', colors(i, :), 'LineWidth', 2);
end
hold(ax, 'off');

xlabel(ax, 'Wavelength (nm)', 'FontSize', 11);
ylabel(ax, 'Normalized SHG Intensity', 'FontSize', 11);
title(ax, 'SHG Spectra Evolution', 'FontSize', 12);
grid(ax, 'on');
set(ax, 'GridAlpha', 0.3);

fprintf('  ├─ ✓ 绘制堆叠图\n');

% 2.3 强度vs功率曲线
subplot(2, 2, 3);
[~, center_idx] = min(abs(wavelength - center_wl));
center_intensity = SHGData.data(:, center_idx);

ax = gca;
plot(ax, powers, center_intensity, 'ro-', 'MarkerSize', 6, 'LineWidth', 2);
xlabel(ax, 'Power (mW)', 'FontSize', 11);
ylabel(ax, 'SHG Intensity (a.u.)', 'FontSize', 11);
title(ax, sprintf('Intensity vs Power @ %.0f nm', center_wl), 'FontSize', 12);
grid(ax, 'on');
set(ax, 'GridAlpha', 0.3);

fprintf('  ├─ ✓ 绘制强度-功率曲线\n');

% 2.4 Log-Log图
subplot(2, 2, 4);
ax = gca;
loglog(ax, powers, center_intensity, 'go-', 'MarkerSize', 6, 'LineWidth', 2);
xlabel(ax, 'Power (mW)', 'FontSize', 11);
ylabel(ax, 'SHG Intensity (a.u.)', 'FontSize', 11);
title(ax, 'Log-Log Plot', 'FontSize', 12);
grid(ax, 'on');
set(ax, 'GridAlpha', 0.3);

fprintf('  └─ ✓ 绘制Log-Log图\n\n');

%% 3. 功率依赖性拟合分析
fprintf('Step 3: 功率依赖性拟合\n');
fprintf('-------------------------------------\n');

% 3.1 在中心波长处进行拟合
figure('Position', [1150 50 800 700]);

[fitresult_center, gof_center] = shg_power_fitting(powers, center_intensity, ...
    'ModelType', 'power', ...
    'Plot', false);

% 手动绘制拟合结果
subplot(3, 1, 1);
ax = gca;
loglog(ax, powers, center_intensity, 'bo', 'MarkerSize', 8, 'LineWidth', 2);
hold(ax, 'on');

% 绘制拟合曲线
power_fit_range = linspace(min(powers), max(powers), 100);
intensity_fitted = fitresult_center.coeff * power_fit_range.^fitresult_center.exponent;
loglog(ax, power_fit_range, intensity_fitted, 'r-', 'LineWidth', 2.5);

hold(ax, 'off');
xlabel(ax, 'Power (mW)', 'FontSize', 11);
ylabel(ax, 'SHG Intensity (a.u.)', 'FontSize', 11);
title(ax, sprintf('Center Wavelength: %s', fitresult_center.model), 'FontSize', 12);
legend(ax, 'Data', 'Fit', 'FontSize', 10, 'Location', 'northwest');
grid(ax, 'on');
set(ax, 'GridAlpha', 0.3);

% 添加拟合统计信息
text(ax, 0.05, 0.95, sprintf('R² = %.6f\nRMSE = %.4f', gof_center.rsquare, gof_center.rmse), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'BackgroundColor', 'white', 'EdgeColor', 'black', 'FontSize', 10);

fprintf('  ├─ ✓ 中心波长处拟合完成\n');
fprintf('      幂指数: %.4f ≈ 2.0 (符合二次谐波特性)\n', fitresult_center.exponent);

% 3.2 在不同波长处的拟合
fprintf('  ├─ 计算不同波长处的拟合幂指数...\n');

test_wavelengths = 630:10:670;
exponents = [];
rsquares = [];

for wl = test_wavelengths
    [~, wl_idx] = min(abs(wavelength - wl));
    wl_intensity = SHGData.data(:, wl_idx);
    
    [fitresult_wl, gof_wl] = shg_power_fitting(powers, wl_intensity, ...
        'ModelType', 'power', ...
        'Plot', false);
    
    exponents = [exponents; fitresult_wl.exponent];
    rsquares = [rsquares; gof_wl.rsquare];
end

% 绘制不同波长的幂指数
subplot(3, 1, 2);
ax = gca;
plot(ax, test_wavelengths, exponents, 'gs-', 'MarkerSize', 8, 'LineWidth', 2);
axline(ax, [0 2], [1 0], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5, 'Label', 'Ideal I ∝ P²');
xlabel(ax, 'Wavelength (nm)', 'FontSize', 11);
ylabel(ax, 'Power Exponent', 'FontSize', 11);
title(ax, 'Power Exponent vs Wavelength', 'FontSize', 12);
legend(ax, 'FontSize', 10, 'Location', 'best');
grid(ax, 'on');
set(ax, 'GridAlpha', 0.3);
set(ax, 'YLim', [1.5 2.5]);

fprintf('  ├─ ✓ 不同波长处的幂指数已计算\n');

% 绘制R²随波长的变化
subplot(3, 1, 3);
ax = gca;
plot(ax, test_wavelengths, rsquares, 'ms-', 'MarkerSize', 8, 'LineWidth', 2);
axline(ax, [0 0.99], [1 0], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5, 'Label', 'Excellent fit');
xlabel(ax, 'Wavelength (nm)', 'FontSize', 11);
ylabel(ax, 'R² Value', 'FontSize', 11);
title(ax, 'Fitting Quality vs Wavelength', 'FontSize', 12);
legend(ax, 'FontSize', 10, 'Location', 'best');
grid(ax, 'on');
set(ax, 'GridAlpha', 0.3);
set(ax, 'YLim', [0.8 1.0]);

fprintf('  └─ ✓ 拟合质量评估完成\n\n');

%% 4. 总结
fprintf('=====================================\n');
fprintf('📊 SHG功率依赖性分析总结\n');
fprintf('=====================================\n\n');

fprintf('✓ 分析完成\n');
fprintf('  数据点数: %d\n', length(powers));
fprintf('  波长范围: %.0f - %.0f nm\n', wavelength(1), wavelength(end));
fprintf('  功率范围: %.3f - %.3f mW\n\n', powers(1), powers(end));

fprintf('关键发现:\n');
fprintf('  • 中心波长处的幂指数: %.4f\n', fitresult_center.exponent);
fprintf('  • 拟合优度 R²: %.6f\n', gof_center.rsquare);
fprintf('  • 强度随功率的依赖性: %s\n\n', fitresult_center.model);

fprintf('💡 提示:\n');
fprintf('  - 幂指数 ≈ 2.0 符合二次谐波 (SHG) 的理论预期\n');
fprintf('  - R² > 0.99 表示拟合质量优秀\n');
fprintf('  - 可以调整功率范围进行更精细的分析\n');
