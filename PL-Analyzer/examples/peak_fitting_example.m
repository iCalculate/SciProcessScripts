%% 光谱分峰和拟合示例
% 演示如何进行峰值分析和高斯拟合

clear; clc; close all;
addpath(genpath('.'));

fprintf('=== 光谱分峰和拟合示例 ===\n\n');

% 创建模拟数据: 包含多个峰
wavelength = 400:0.5:800;
spectrum = zeros(size(wavelength));

% 添加三个高斯峰
peaks_center = [500, 650, 750];
peaks_amplitude = [300, 800, 400];
peaks_sigma = [20, 25, 30];

for i = 1:length(peaks_center)
    spectrum = spectrum + peaks_amplitude(i) * exp(-((wavelength - peaks_center(i)).^2) / (2 * peaks_sigma(i)^2));
end

% 加入噪声和背景
spectrum = spectrum + randn(size(spectrum)) * 20 + 50;

%% 方法1: 峰值分析
fprintf('Step 1: 峰值分析\n');
fprintf('-------------------------------------------\n');

[peaks, positions, widths, amplitudes] = peak_analysis(spectrum, wavelength, ...
    'MinPeakHeight', 50, ...
    'MinPeakDistance', 30, ...
    'Plot', true);

%% 方法2: 高斯拟合
fprintf('\nStep 2: 多峰高斯拟合\n');
fprintf('-------------------------------------------\n');

numPeaks = 3;  % 已知有3个峰
[fitParams, fittedCurve, residuals, rSquare] = peak_fitting(spectrum, wavelength, numPeaks, ...
    'FitType', 'gauss', ...
    'Plot', true, ...
    'Verbose', true);

%% 方法3: 比较原始数据与拟合结果
fprintf('\nStep 3: 结果对比\n');
fprintf('-------------------------------------------\n');

figure('Color', 'w');
plot(wavelength, spectrum, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Raw Data'); hold on;
plot(wavelength, fittedCurve, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Fitted Curve');
plot(wavelength, residuals + min(spectrum), 'g--', 'LineWidth', 1, 'DisplayName', 'Residuals');

% 标记峰位
for i = 1:length(fitParams)
    plot(fitParams{i}.center, fitParams{i}.amplitude, 'ko', ...
        'MarkerSize', 8, 'MarkerFaceColor', 'r');
end

set(gca, 'FontName', 'Arial', 'FontSize', 12);
xlabel('Wavelength (nm)', 'FontSize', 14);
ylabel('Intensity (a.u.)', 'FontSize', 14);
title(sprintf('Spectral Peak Analysis and Fitting (R² = %.4f)', rSquare), 'FontSize', 14);
legend('Location', 'best', 'FontSize', 12);
grid on;
xlim([400 800]);

fprintf('\n✓ 分峰拟合示例完成！\n');
