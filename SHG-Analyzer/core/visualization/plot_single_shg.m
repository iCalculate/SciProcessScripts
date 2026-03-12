function [fig, ax] = plot_single_shg(spectrum, wavelength, varargin)
%PLOT_SINGLE_SHG 绘制单条二次谐波(SHG)光谱曲线
%   [fig, ax] = plot_single_shg(spectrum, wavelength, Name, Value)
%
%   这是最基础的绘图函数，用于绘制单条二次谐波(SHG)光谱
%
%   输入:
%       spectrum - SHG强度数据向量
%       wavelength - 波长数据向量 (nm)
%
%   可选参数:
%       'Title' - 图形标题
%       'LineColor' - 线条颜色，默认 'r'
%       'LineWidth' - 线宽，默认 2
%       'FontSize' - 字体大小，默认 12
%       'LabelFontSize' - 标签字体大小，默认 14
%       'ShowGrid' - 是否显示网格，默认 true
%       'LogScale' - 是否使用对数坐标轴，默认 false
%
%   输出:
%       fig - 图形句柄
%       ax - 坐标轴句柄
%
%   示例:
%       wavelength = 400:1000;  % nm
%       spectrum = randn(1, 601);
%       [fig, ax] = plot_single_shg(spectrum, wavelength, 'Title', 'SHG Spectrum');

p = inputParser;
addRequired(p, 'spectrum', @isvector);
addRequired(p, 'wavelength', @isvector);
addParameter(p, 'Title', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'LineColor', 'r', @(x) ischar(x) || isstring(x) || isvector(x));
addParameter(p, 'LineWidth', 2, @(x) isnumeric(x) && x > 0);
addParameter(p, 'FontSize', 12, @(x) isnumeric(x) && x > 0);
addParameter(p, 'LabelFontSize', 14, @(x) isnumeric(x) && x > 0);
addParameter(p, 'ShowGrid', true, @islogical);
addParameter(p, 'LogScale', false, @islogical);
parse(p, spectrum, wavelength, varargin{:});

% 创建图形
fig = figure('Color', 'w');
ax = axes('Parent', fig);

% 绘制曲线
plot(ax, wavelength(:), spectrum(:), ...
    'Color', p.Results.LineColor, ...
    'LineWidth', p.Results.LineWidth);

% 格式化
set(ax, 'FontName', 'Arial', 'FontSize', p.Results.FontSize, ...
    'LineWidth', 1.5, 'Box', 'on');

xlabel(ax, 'Wavelength (nm)', 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);
ylabel(ax, 'SHG Intensity (a.u.)', 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);

if ~isempty(p.Results.Title)
    title(ax, p.Results.Title, 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);
end

if p.Results.ShowGrid
    grid(ax, 'on');
    set(ax, 'GridAlpha', 0.3);
end

% 应用对数坐标轴
if p.Results.LogScale
    set(ax, 'YScale', 'log');
end

end
