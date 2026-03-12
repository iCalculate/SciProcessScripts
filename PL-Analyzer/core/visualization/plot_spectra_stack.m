function [fig, ax] = plot_spectra_stack(PowSerPL, varargin)
%PLOT_SPECTRA_STACK 绘制PL谱线堆叠图（随功率/角度等变化）
%   [fig, ax] = plot_spectra_stack(PowSerPL, Name, Value)
%
%   输入:
%       PowSerPL - PL数据结构
%           .data - 光谱矩阵
%           .axisscale{1,1} - 参数轴 (功率/角度等)
%           .axisscale{2,1} - 波长轴
%
%   可选参数:
%       'Indices' - 要绘制的索引列表，默认绘制所有
%       'WavelengthRange' - 波长范围 [min max]
%       'NormalizeMethod' - 'none'/'max'/'peak_range'/'area'，默认 'peak_range'
%       'PeakRange' - 峰值范围 [min max]
%       'Spacing' - 曲线间距，默认 0.1
%       'Colormap' - 颜色映射方式
%       'Title' - 图形标题
%
%   输出:
%       fig - 图形句柄
%       ax - 坐标轴句柄

p = inputParser;
addRequired(p, 'PowSerPL', @isstruct);
addParameter(p, 'Indices', [], @(x) isnumeric(x) || islogical(x));
addParameter(p, 'WavelengthRange', [], @(x) isempty(x) || (isnumeric(x) && length(x)==2));
addParameter(p, 'NormalizeMethod', 'peak_range', ...
    @(x) ismember(x, {'none', 'max', 'peak_range', 'area'}));
addParameter(p, 'PeakRange', [650 700], @(x) isnumeric(x) && length(x)==2);
addParameter(p, 'Spacing', 0.1, @(x) isnumeric(x) && x >= 0);
addParameter(p, 'Colormap', 'power', @(x) ischar(x) || isstring(x) || ismatrix(x));
addParameter(p, 'Title', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'FontSize', 12, @(x) isnumeric(x) && x > 0);
addParameter(p, 'LabelFontSize', 14, @(x) isnumeric(x) && x > 0);
addParameter(p, 'LineWidth', 1.5, @(x) isnumeric(x) && x > 0);
parse(p, PowSerPL, varargin{:});

% 数据提取
data = double(PowSerPL.data);
paramAxis = PowSerPL.axisscale{1,1};
wavelengthAxis = PowSerPL.axisscale{2,1};

% 处理索引
if isempty(p.Results.Indices)
    indices = 1:size(data, 1);
elseif islogical(p.Results.Indices)
    indices = find(p.Results.Indices);
else
    indices = p.Results.Indices(:);
end

% 波长范围
if isempty(p.Results.WavelengthRange)
    wavelengthRange = [min(wavelengthAxis), max(wavelengthAxis)];
else
    wavelengthRange = p.Results.WavelengthRange;
end
wavelengthMask = wavelengthAxis >= wavelengthRange(1) & wavelengthAxis <= wavelengthRange(2);

% 数据归一化
numSpectra = length(indices);
normalizedData = zeros(numSpectra, sum(wavelengthMask));

for i = 1:numSpectra
    idx = indices(i);
    spectrum = data(idx, wavelengthMask);
    normalizedData(i, :) = normalize_spectrum_core(spectrum, ...
        wavelengthAxis(wavelengthMask), p.Results.NormalizeMethod, p.Results.PeakRange);
end

% 颜色设置
colors = get_colors_core(paramAxis(indices), p.Results.Colormap, numSpectra);

% 绘图
fig = figure('Color', 'w');
ax = axes('Parent', fig);
hold(ax, 'on');

wavelengthPlot = wavelengthAxis(wavelengthMask);

for i = 1:numSpectra
    yOffset = (i - 1) * p.Results.Spacing;
    plotData = normalizedData(i, :) + yOffset;
    
    plot(ax, wavelengthPlot, plotData, ...
        'Color', colors(i, :), 'LineWidth', p.Results.LineWidth);
end

hold(ax, 'off');

% 轴设置
set(ax, 'FontName', 'Arial', 'FontSize', p.Results.FontSize, 'LineWidth', 1, 'Box', 'on');
xlabel(ax, 'Wavelength (nm)', 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);
ylabel(ax, 'Normalized Intensity (a.u.)', 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);

if ~isempty(p.Results.Title)
    title(ax, p.Results.Title, 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);
end

xlim(ax, wavelengthRange);
maxOffset = (numSpectra - 1) * p.Results.Spacing;
ylim(ax, [-0.1, 1 + maxOffset + 0.1]);
set(ax, 'YTickLabel', []);

end

%% 辅助函数

function normalizedSpectrum = normalize_spectrum_core(spectrum, wavelengthAxis, method, peakRange)
switch method
    case 'none'
        normalizedSpectrum = spectrum;
    case 'max'
        maxVal = max(spectrum);
        if maxVal > 0
            normalizedSpectrum = spectrum / maxVal;
        else
            normalizedSpectrum = spectrum;
        end
    case 'peak_range'
        peakMask = wavelengthAxis >= peakRange(1) & wavelengthAxis <= peakRange(2);
        if any(peakMask)
            peakVal = max(spectrum(peakMask));
        else
            peakVal = max(spectrum);
        end
        
        if peakVal > 0
            normalizedSpectrum = spectrum / peakVal;
        else
            normalizedSpectrum = spectrum;
        end
        
        minVal = min(normalizedSpectrum);
        normalizedSpectrum = normalizedSpectrum - minVal;
        maxVal = max(normalizedSpectrum);
        
        if maxVal > 0
            normalizedSpectrum = normalizedSpectrum / maxVal;
        end
    case 'area'
        area = trapz(wavelengthAxis, spectrum);
        if area > 0
            normalizedSpectrum = spectrum / area;
        else
            normalizedSpectrum = spectrum;
        end
end
end

function colors = get_colors_core(paramValues, colormapMode, numSpectra)
if ischar(colormapMode) || isstring(colormapMode)
    if strcmp(colormapMode, 'power')
        paramNorm = (paramValues - min(paramValues)) / max(max(paramValues) - min(paramValues), eps);
        colors = [zeros(numSpectra, 1), zeros(numSpectra, 1), 1 - paramNorm(:)];
    elseif strcmp(colormapMode, 'orange')
        % 橙色到黑色的渐变 (Orange to Black)
        paramNorm = (paramValues - min(paramValues)) / max(max(paramValues) - min(paramValues), eps);
        colors = [(1 - paramNorm(:)) * 1.0, (1 - paramNorm(:)) * 0.5, zeros(numSpectra, 1)];
    elseif strcmp(colormapMode, 'index')
        idxNorm = (1:numSpectra) / numSpectra;
        colors = [zeros(numSpectra, 1), zeros(numSpectra, 1), 1 - idxNorm'];
    else
        try
            cmap = colormap(colormapMode);
            if size(cmap, 1) < numSpectra
                colors = interp1(1:size(cmap, 1), cmap, ...
                               linspace(1, size(cmap, 1), numSpectra), 'linear');
            else
                colors = cmap(1:numSpectra, :);
            end
        catch
            idxNorm = (1:numSpectra) / numSpectra;
            colors = [zeros(numSpectra, 1), zeros(numSpectra, 1), 1 - idxNorm'];
        end
    end
else
    colors = colormapMode;
end
end
