%% 偏振相关性分析示例
% 演示 PL 随偏振角度变化的分析流程

clear; clc; close all;
addpath(genpath('.'));

fprintf('=== 偏振相关性分析示例 ===\n\n');

%% 创建模拟偏振扫描数据
fprintf('Step 1: 创建偏振依赖的数据\n');
fprintf('-------------------------------------------\n');

angles = 0:5:180;  % 偏振角度 (度)
numAngles = length(angles);
wavelength = 400:1:1000;
numWavelengths = length(wavelength);

% 创建偏振依赖的光谱: I(θ) = I0·cos²(2θ)
spectra = zeros(numAngles, numWavelengths);
for i = 1:numAngles
    % 各向异性: 强度随偏振角度变化
    polarizationFactor = cos(2 * deg2rad(angles(i)))^2;
    peakAmplitude = 500 * (0.5 + 0.5 * polarizationFactor);  % 0.5 到 1.0 之间
    
    spectra(i, :) = peakAmplitude * exp(-((wavelength - 650).^2) / 3000);
    spectra(i, :) = spectra(i, :) + randn(1, numWavelengths) * 20;
end

% 构造标准数据结构
PolarPL.data = spectra;
PolarPL.axisscale{1,1} = angles';
PolarPL.axisscale{2,1} = wavelength';

fprintf('数据大小: %d × %d\n', size(PolarPL.data));
fprintf('角度范围: %.0f - %.0f°\n', min(angles), max(angles));
fprintf('波长范围: %.0f - %.0f nm\n\n', min(wavelength), max(wavelength));

%% 分析1: 绘制等高线图
fprintf('Step 2: 等高线图分析\n');
fprintf('-------------------------------------------\n');

figure('Position', [100 100 900 600]);
[fig1, ax1] = plot_contour_map(PolarPL, ...
    'ColorRange', [0.4 1.0], ...
    'ContourLevels', 40, ...
    'Colormap', 'viridis', ...
    'ShowEnergyAxis', false, ...
    'Title', 'PL Intensity Map vs Polarization Angle');

%% 分析2: 堆叠谱线
fprintf('\nStep 3: 谱线随角度的变化\n');
fprintf('-------------------------------------------\n');

figure('Position', [1050 100 900 600]);
indices = 1:6:numAngles;  % 每隔6条绘制
[fig2, ax2] = plot_spectra_stack(PolarPL, ...
    'Indices', indices, ...
    'NormalizeMethod', 'peak_range', ...
    'PeakRange', [630 670], ...
    'Spacing', 0.15, ...
    'Colormap', 'index', ...
    'Title', 'PL Spectra Evolution with Polarization Angle');

%% 分析3: 各向异性分析
fprintf('\nStep 4: 偏振各向异性分析\n');
fprintf('-------------------------------------------\n');

% 计算不同角度下的积分强度
wavelengthRange = [630 670];
wavelengthMask = wavelength >= wavelengthRange(1) & wavelength <= wavelengthRange(2);
intensities = sum(spectra(:, wavelengthMask), 2);

% 拟合为 I = I0·cos²(2θ) + bg
fitOpts = fitoptions('Method', 'NonlinearLeastSquares', 'Display', 'off');
fitOpts.StartPoint = [1, 0.5];
fitOpts.Lower = [0, 0];
fitOpts.Upper = [2000, 1000];

fitType = fittype('I0*cos(2*pi*x/180)^2 + bg', 'independent', 'x', 'dependent', 'y');
try
    fo = fit(angles', intensities, fitType, fitOpts);
    fprintf('拟合结果: I(θ) = I₀·cos²(2θ) + bg\n');
    fprintf('  I₀ = %.2f\n', fo.I0);
    fprintf('  bg = %.2f\n', fo.bg);
    
    % 计算各向异性系数
    anisotropy = (max(intensities) - min(intensities)) / (max(intensities) + min(intensities));
    fprintf('  各向异性系数 ρ = %.3f\n', anisotropy);
    
    % 绘制偏振角度依赖性
    figure('Color', 'w');
    plot(angles, intensities, 'bo', 'MarkerSize', 8, 'LineWidth', 2, ...
        'DisplayName', 'Data'); hold on;
    
    anglesFit = linspace(0, 180, 1000);
    intensitiesFit = fo.I0 * cos(2*pi*anglesFit/180).^2 + fo.bg;
    plot(anglesFit, intensitiesFit, 'r-', 'LineWidth', 2.5, ...
        'DisplayName', 'Fit: I₀·cos²(2θ)');
    
    set(gca, 'FontName', 'Arial', 'FontSize', 12);
    xlabel('Polarization Angle (°)', 'FontSize', 14);
    ylabel('Integrated Intensity (a.u.)', 'FontSize', 14);
    title('Polarization-Dependent PL Intensity', 'FontSize', 14);
    legend('FontSize', 12);
    grid on;
    xlim([0 180]);
    
catch
    fprintf('偏振拟合失败\n');
end

%% 分析4: 偏振度分析
fprintf('\nStep 5: 偏振特性汇总\n');
fprintf('-------------------------------------------\n');

fprintf('\n=== 偏振各向异性特性 ===\n');
fprintf('最大强度: %.2f (角度: %.0f°)\n', max(intensities), angles(find(intensities == max(intensities), 1)));
fprintf('最小强度: %.2f (角度: %.0f°)\n', min(intensities), angles(find(intensities == min(intensities), 1)));
fprintf('各向异性系数: %.3f\n', anisotropy);
fprintf('(ρ值越接近1，表示各向异性越强)\n');

fprintf('\n✓ 偏振分析示例完成！\n');
