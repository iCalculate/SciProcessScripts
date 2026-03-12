%% 功率依赖性分析示例
% 演示 PL 随激光功率变化的分析流程

clear; clc; close all;
addpath(genpath('.'));

fprintf('=== 功率依赖性分析示例 ===\n\n');

%% 创建模拟功率扫描数据
fprintf('Step 1: 加载或创建数据\n');
fprintf('-------------------------------------------\n');

powers = 0.05:0.05:2.0;  % mW
numPowers = length(powers);
wavelength = 400:1:1000;
numWavelengths = length(wavelength);

% 创建功率依赖的光谱
spectra = zeros(numPowers, numWavelengths);
for i = 1:numPowers
    % 峰强度与功率的关系: I ∝ P^n (n ≈ 1.5 for excitonic recombination)
    peakAmplitude = 200 * powers(i)^1.5;
    spectra(i, :) = peakAmplitude * exp(-((wavelength - 650).^2) / 3000);
    spectra(i, :) = spectra(i, :) + randn(1, numWavelengths) * (peakAmplitude * 0.05);
end

% 构造标准数据结构
PowSerPL.data = spectra;
PowSerPL.axisscale{1,1} = powers';
PowSerPL.axisscale{2,1} = wavelength';

fprintf('数据大小: %d × %d\n', size(PowSerPL.data));
fprintf('功率范围: %.2f - %.2f mW\n', min(powers), max(powers));
fprintf('波长范围: %.0f - %.0f nm\n\n', min(wavelength), max(wavelength));

%% 分析1: 绘制等高线图
fprintf('Step 2: 等高线图分析\n');
fprintf('-------------------------------------------\n');

figure('Position', [100 100 900 600]);
[fig1, ax1] = plot_contour_map(PowSerPL, ...
    'ColorRange', [0.5 1.0], ...
    'ContourLevels', 50, ...
    'Colormap', 'hot', ...
    'ShowEnergyAxis', false, ...
    'Title', 'PL Intensity Map vs Laser Power');

%% 分析2: 绘制堆叠谱线
fprintf('\nStep 3: 谱线演变分析\n');
fprintf('-------------------------------------------\n');

figure('Position', [1050 100 900 600]);
indices = 1:4:numPowers;  % 每隔4条绘制
[fig2, ax2] = plot_spectra_stack(PowSerPL, ...
    'Indices', indices, ...
    'NormalizeMethod', 'peak_range', ...
    'PeakRange', [630 670], ...
    'Spacing', 0.2, ...
    'Colormap', 'power', ...
    'Title', 'PL Spectra Evolution with Power');

%% 分析3: 功率依赖性定量分析
fprintf('\nStep 4: 功率依赖性分析\n');
fprintf('-------------------------------------------\n');

% 计算每条光谱在特定波长范围内的积分强度
wavelengthRange = [630 670];
wavelengthMask = wavelength >= wavelengthRange(1) & wavelength <= wavelengthRange(2);
intensities = sum(spectra(:, wavelengthMask), 2);

% 拟合 I ∝ P^n
fitOpts = fitoptions('Method', 'NonlinearLeastSquares', 'Display', 'off');
fitOpts.StartPoint = [1, 1.5];
fitOpts.Lower = [0, 0.1];
fitOpts.Upper = [1000, 3];

fitType = fittype('a*x^n', 'independent', 'x', 'dependent', 'y');
try
    fo = fit(powers', intensities, fitType, fitOpts);
    nValue = fo.n;
    aValue = fo.a;
    fprintf('拟合结果: I = a·P^n\n');
    fprintf('  a = %.2f\n', aValue);
    fprintf('  n = %.3f\n', nValue);
    
    % 绘制功率依赖性
    figure('Color', 'w');
    plot(powers, intensities, 'bo', 'MarkerSize', 8, 'LineWidth', 2, ...
        'DisplayName', 'Data'); hold on;
    
    powersFit = linspace(min(powers), max(powers), 100);
    intensitiesFit = aValue * powersFit.^nValue;
    plot(powersFit, intensitiesFit, 'r-', 'LineWidth', 2.5, ...
        'DisplayName', sprintf('Fit: I = %.2f·P^{%.3f}', aValue, nValue));
    
    set(gca, 'FontName', 'Arial', 'FontSize', 12);
    xlabel('Laser Power (mW)', 'FontSize', 14);
    ylabel('Integrated Intensity (a.u.)', 'FontSize', 14);
    title('Power-Dependent PL Intensity', 'FontSize', 14);
    legend('FontSize', 12);
    grid on;
    set(gca, 'XScale', 'log', 'YScale', 'log');
    
catch
    fprintf('功率拟合失败\n');
end

fprintf('\n✓ 功率依赖性分析示例完成！\n');
