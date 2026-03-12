%% 基础绘图示例
% 演示如何绘制单条PL光谱和基本的堆叠图

clear; clc; close all;

% 添加路径
addpath(genpath('.'));

%% 示例1: 绘制单条光谱
fprintf('=== 示例1: 单条PL光谱绘制 ===\n\n');

% 创建模拟数据
wavelength = 400:1:1000;  % nm
spectrum1 = 1000 * exp(-((wavelength - 650).^2) / 5000);  % 高斯峰 @650nm
spectrum1 = spectrum1 + randn(size(spectrum1)) * 10;  % 加入噪声

% 绘制
[fig1, ax1] = plot_single_spectrum(spectrum1, wavelength, ...
    'Title', 'Single PL Spectrum', ...
    'LineColor', 'b', ...
    'LineWidth', 2.5);

%% 示例2: 创建模拟数据
fprintf('示例2: 多条光谱堆叠图\n\n');

% 模拟功率扫描数据
powers = 0.1:0.1:2.0;  % mW
numPowers = length(powers);
numWavelengths = 401;  % 400-800 nm
wavelength = linspace(400, 800, numWavelengths);

% 创建光谱数据 (功率依赖)
spectra = zeros(numPowers, numWavelengths);
for i = 1:numPowers
    % 峰强度随功率增加
    peakAmplitude = 500 * sqrt(powers(i));
    spectra(i, :) = peakAmplitude * exp(-((wavelength - 650).^2) / 5000);
    % 加入噪声
    spectra(i, :) = spectra(i, :) + randn(1, numWavelengths) * (peakAmplitude * 0.05);
end

% 构造标准数据结构
PowSerPL.data = spectra;
PowSerPL.axisscale{1,1} = powers';
PowSerPL.axisscale{2,1} = wavelength';

% 绘制堆叠图
[fig2, ax2] = plot_spectra_stack(PowSerPL, ...
    'Indices', 1:2:numPowers, ...  % 每隔一条绘制
    'NormalizeMethod', 'peak_range', ...
    'PeakRange', [630 670], ...
    'Spacing', 0.15, ...
    'Colormap', 'power', ...
    'Title', 'PL Spectra Evolution with Power', ...
    'LineWidth', 2);

%% 示例3: 等高线图
fprintf('示例3: 等高线图\n\n');

[fig3, ax3] = plot_contour_map(PowSerPL, ...
    'ColorRange', [0.5 1.0], ...
    'ContourLevels', 50, ...
    'Colormap', 'hot', ...
    'ShowEnergyAxis', true, ...
    'Title', 'PL Intensity Map vs Power');

fprintf('\n✓ 所有基础绘图示例完成！\n');
