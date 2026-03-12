function [fitParams, fittedCurve, residuals, rSquare] = peak_fitting(spectrum, wavelength, numPeaks, varargin)
%PEAK_FITTING 用高斯或洛伦兹函数拟合光谱峰
%   [fitParams, fittedCurve, residuals, rSquare] = peak_fitting(spectrum, wavelength, numPeaks, Name, Value)
%
%   输入:
%       spectrum - 光谱强度数据
%       wavelength - 波长数据
%       numPeaks - 要拟合的峰数
%
%   可选参数:
%       'FitType' - 'gauss' (默认) 或 'lorentz'
%       'Plot' - 是否绘制拟合结果，默认 false
%       'Verbose' - 详细输出，默认 true
%
%   输出:
%       fitParams - 拟合参数 (cell数组)
%           fitParams{i}.center - 中心波长 (nm)
%           fitParams{i}.amplitude - 幅度
%           fitParams{i}.sigma - 标准差
%           fitParams{i}.background - 背景
%       fittedCurve - 拟合后的曲线
%       residuals - 残差
%       rSquare - 决定系数 R²
%
%   示例:
%       [params, fitted, res, r2] = peak_fitting(spectrum, wavelength, 2, 'Plot', true);

p = inputParser;
addRequired(p, 'spectrum', @isvector);
addRequired(p, 'wavelength', @isvector);
addRequired(p, 'numPeaks', @(x) isnumeric(x) && x > 0);
addParameter(p, 'FitType', 'gauss', @(x) ismember(x, {'gauss', 'lorentz'}));
addParameter(p, 'Plot', false, @islogical);
addParameter(p, 'Verbose', true, @islogical);
parse(p, spectrum, wavelength, numPeaks, varargin{:});

spectrum = spectrum(:);
wavelength = wavelength(:);

% 基础拟合选项
fitOpts = fitoptions('Method', 'NonlinearLeastSquares', 'Display', 'off');
fitOpts.MaxIter = 10000;
fitOpts.TolFun = 1e-6;

% 初始参数估计
[~, locs] = findpeaks(spectrum, 'SortStr', 'descend', 'NPeaks', p.Results.numPeaks);
locs = sort(locs);
peakCenters = wavelength(locs);
peakAmplitudes = spectrum(locs);

% 背景估计
background = mean([spectrum(1:5); spectrum(end-4:end)]);

% 构建拟合模型 - 使用匿名函数支持任意峰数
% 生成动态的拟合方程
numPeaks = p.Results.numPeaks;
fitType_str = '';

if strcmp(p.Results.FitType, 'gauss')
    % 高斯函数: a*exp(-((x-b)/c)^2)
    for i = 1:numPeaks
        if i == 1
            fitType_str = sprintf('a%d*exp(-((x-b%d)/c%d)^2)', i, i, i);
        else
            fitType_str = sprintf('%s + a%d*exp(-((x-b%d)/c%d)^2)', fitType_str, i, i, i);
        end
    end
else
    % 洛伦兹函数: a/(1+((x-b)/c)^2)
    for i = 1:numPeaks
        if i == 1
            fitType_str = sprintf('a%d/(1+((x-b%d)/c%d)^2)', i, i, i);
        else
            fitType_str = sprintf('%s + a%d/(1+((x-b%d)/c%d)^2)', fitType_str, i, i, i);
        end
    end
end

% 创建 fittype 对象
fitType = fittype(fitType_str, 'independent', 'x', 'dependent', 'y');

% 生成初始参数向量
startPointVec = [];
for i = 1:numPeaks
    startPointVec = [startPointVec; peakAmplitudes(i); peakCenters(i); (max(wavelength) - min(wavelength)) / 20];
end

% 设置初始参数
fitOpts.StartPoint = startPointVec;

% 进行拟合
try
    fo = fit(wavelength, spectrum, fitType, fitOpts);
    fittedCurve = fo(wavelength);
    residuals = spectrum - fittedCurve;
    
    % 计算 R²
    ssRes = sum(residuals.^2);
    ssTot = sum((spectrum - mean(spectrum)).^2);
    rSquare = 1 - ssRes / ssTot;
    
    % 提取参数
    coeffs = coeffvalues(fo);
    numPeaksToExtract = min(p.Results.numPeaks, length(coeffs) / 3);
    fitParams = cell(numPeaksToExtract, 1);
    for i = 1:numPeaksToExtract
        fitParams{i}.center = coeffs(i*3-1);
        fitParams{i}.amplitude = coeffs(i*3-2);
        fitParams{i}.sigma = abs(coeffs(i*3));
        fitParams{i}.background = background;
    end
    
catch ME
    warning('拟合失败: %s', ME.message);
    fittedCurve = spectrum;
    residuals = zeros(size(spectrum));
    rSquare = 0;
    fitParams = {};
end

% 输出信息
if p.Results.Verbose
    fprintf('\n=== Peak Fitting Results ===\n');
    fprintf('Fit Type: %s\n', p.Results.FitType);
    fprintf('Number of Peaks: %d\n', p.Results.numPeaks);
    fprintf('R² = %.4f\n\n', rSquare);
    
    for i = 1:length(fitParams)
        fprintf('Peak %d:\n', i);
        fprintf('  Center: %.2f nm\n', fitParams{i}.center);
        fprintf('  Amplitude: %.2f\n', fitParams{i}.amplitude);
        fprintf('  FWHM: %.2f nm\n', 2.355 * fitParams{i}.sigma);
    end
end

% 绘图
if p.Results.Plot
    figure('Color', 'w');
    plot(wavelength, spectrum, 'o-', 'LineWidth', 1.5, 'DisplayName', 'Data'); hold on;
    plot(wavelength, fittedCurve, 'r-', 'LineWidth', 2, 'DisplayName', 'Fit');
    
    for i = 1:length(fitParams)
        plot(fitParams{i}.center, fitParams{i}.amplitude, 'g*', 'MarkerSize', 12);
    end
    
    set(gca, 'FontName', 'Arial', 'FontSize', 12);
    xlabel('Wavelength (nm)', 'FontSize', 14);
    ylabel('Intensity (a.u.)', 'FontSize', 14);
    title(sprintf('Peak Fitting (R² = %.4f)', rSquare), 'FontSize', 14);
    grid on;
    legend('Location', 'best');
end

end
