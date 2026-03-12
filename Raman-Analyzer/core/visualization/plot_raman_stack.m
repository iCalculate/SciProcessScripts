function [fig, ax] = plot_raman_stack(RamanData, varargin)
%PLOT_RAMAN_STACK 绘制拉曼谱线堆叠图（随功率/温度等变化）
%   [fig, ax] = plot_raman_stack(RamanData, Name, Value)
%
%   输入:
%       RamanData - 拉曼数据结构
%           .data - 光谱矩阵
%           .axisscale{1,1} - 参数轴 (功率/温度等)
%           .axisscale{2,1} - 拉曼位移轴
%
%   可选参数:
%       'Indices' - 要绘制的索引列表，默认绘制所有
%       'RamanRange' - 拉曼位移范围 [min max]
%       'NormalizeMethod' - 'none'/'max'/'peak_range'/'area'，默认 'max'
%       'PeakRange' - 峰值范围 [min max]
%       'Spacing' - 曲线间距，默认 0.1
%       'Colormap' - 颜色映射方式
%       'Title' - 图形标题
%
%   输出:
%       fig - 图形句柄
%       ax - 坐标轴句柄

p = inputParser;
addRequired(p, 'RamanData', @isstruct);
addParameter(p, 'Indices', [], @(x) isnumeric(x) || islogical(x));
addParameter(p, 'RamanRange', [], @(x) isempty(x) || (isnumeric(x) && length(x)==2));
addParameter(p, 'NormalizeMethod', 'max', ...
    @(x) ismember(x, {'none', 'max', 'peak_range', 'area'}));
addParameter(p, 'PeakRange', [200 3000], @(x) isnumeric(x) && length(x)==2);
addParameter(p, 'Spacing', 0.1, @(x) isnumeric(x) && x >= 0);
addParameter(p, 'Colormap', 'power', @(x) ischar(x) || isstring(x) || ismatrix(x));
addParameter(p, 'Title', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'FontSize', 12, @(x) isnumeric(x) && x > 0);
addParameter(p, 'LabelFontSize', 14, @(x) isnumeric(x) && x > 0);
addParameter(p, 'LineWidth', 1.5, @(x) isnumeric(x) && x > 0);
parse(p, RamanData, varargin{:});

% 数据提取
data = double(RamanData.data);
paramAxis = RamanData.axisscale{1,1};
ramanAxis = RamanData.axisscale{2,1};

% 处理索引
if isempty(p.Results.Indices)
    indices = 1:size(data, 1);
elseif islogical(p.Results.Indices)
    indices = find(p.Results.Indices);
else
    indices = p.Results.Indices(:);
end

% 拉曼位移范围
if isempty(p.Results.RamanRange)
    ramanRange = [min(ramanAxis), max(ramanAxis)];
else
    ramanRange = p.Results.RamanRange;
end
ramanMask = ramanAxis >= ramanRange(1) & ramanAxis <= ramanRange(2);

% 数据归一化
numSpectra = length(indices);
normalizedData = zeros(numSpectra, sum(ramanMask));

for i = 1:numSpectra
    idx = indices(i);
    spectrum = data(idx, ramanMask);
    normalizedData(i, :) = normalize_raman_spectrum(spectrum, ...
        ramanAxis(ramanMask), p.Results.NormalizeMethod, p.Results.PeakRange);
end

% 颜色设置
colors = get_colors_raman(paramAxis(indices), p.Results.Colormap, numSpectra);

% 绘图
fig = figure('Color', 'w');
ax = axes('Parent', fig);
hold(ax, 'on');

ramanPlot = ramanAxis(ramanMask);

for i = 1:numSpectra
    yOffset = (i - 1) * p.Results.Spacing;
    plotData = normalizedData(i, :) + yOffset;
    
    plot(ax, ramanPlot, plotData, ...
        'Color', colors(i, :), 'LineWidth', p.Results.LineWidth);
end

hold(ax, 'off');

% 轴设置
set(ax, 'FontName', 'Arial', 'FontSize', p.Results.FontSize, 'LineWidth', 1, 'Box', 'on');
xlabel(ax, 'Raman Shift (cm⁻¹)', 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);
ylabel(ax, 'Normalized Intensity (a.u.)', 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);

if ~isempty(p.Results.Title)
    title(ax, p.Results.Title, 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);
end

xlim(ax, ramanRange);
maxOffset = (numSpectra - 1) * p.Results.Spacing;
ylim(ax, [-0.1, 1 + maxOffset + 0.1]);
set(ax, 'YTickLabel', []);

end

%% 辅助函数

function normalizedSpectrum = normalize_raman_spectrum(spectrum, ramanAxis, method, peakRange)
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
        peakMask = ramanAxis >= peakRange(1) & ramanAxis <= peakRange(2);
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
        area = trapz(ramanAxis, spectrum);
        if area > 0
            normalizedSpectrum = spectrum / area;
        else
            normalizedSpectrum = spectrum;
        end
end
end

function colors = get_colors_raman(paramValues, colormapMode, numSpectra)
if ischar(colormapMode) || isstring(colormapMode)
    if strcmp(colormapMode, 'power')
        paramNorm = (paramValues - min(paramValues)) / max(max(paramValues) - min(paramValues), eps);
        colors = [zeros(numSpectra, 1), zeros(numSpectra, 1), 1 - paramNorm(:)];
    elseif strcmp(colormapMode, 'orange')
        % 橙色到黑色渐变: 从 [1, 0.5, 0] (橙色) 到 [0, 0, 0] (黑色)
        paramNorm = (paramValues - min(paramValues)) / max(max(paramValues) - min(paramValues), eps);
        colors = [(1 - paramNorm(:)), 0.5 * (1 - paramNorm(:)), zeros(numSpectra, 1)];
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
