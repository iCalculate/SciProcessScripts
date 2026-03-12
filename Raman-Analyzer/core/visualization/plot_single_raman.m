function [fig, ax] = plot_single_raman(spectrum, ramanShift, varargin)
%PLOT_SINGLE_RAMAN 绘制单条拉曼光谱曲线
%   [fig, ax] = plot_single_raman(spectrum, ramanShift, Name, Value)
%
%   这是最基础的绘图函数，用于绘制单条拉曼光谱
%
%   输入:
%       spectrum - 拉曼强度数据向量
%       ramanShift - 拉曼位移数据向量 (cm⁻¹)
%
%   可选参数:
%       'Title' - 图形标题
%       'LineColor' - 线条颜色，默认 'b'
%       'LineWidth' - 线宽，默认 2
%       'FontSize' - 字体大小，默认 12
%       'LabelFontSize' - 标签字体大小，默认 14
%       'ShowGrid' - 是否显示网格，默认 true
%
%   输出:
%       fig - 图形句柄
%       ax - 坐标轴句柄
%
%   示例:
%       ramanShift = 0:3000;  % cm⁻¹
%       spectrum = randn(1, 3001);
%       [fig, ax] = plot_single_raman(spectrum, ramanShift, 'Title', 'Sample Raman');

p = inputParser;
addRequired(p, 'spectrum', @isvector);
addRequired(p, 'ramanShift', @isvector);
addParameter(p, 'Title', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'LineColor', 'b', @(x) ischar(x) || isstring(x) || isvector(x));
addParameter(p, 'LineWidth', 2, @(x) isnumeric(x) && x > 0);
addParameter(p, 'FontSize', 12, @(x) isnumeric(x) && x > 0);
addParameter(p, 'LabelFontSize', 14, @(x) isnumeric(x) && x > 0);
addParameter(p, 'ShowGrid', true, @islogical);
parse(p, spectrum, ramanShift, varargin{:});

% 创建图形
fig = figure('Color', 'w');
ax = axes('Parent', fig);

% 绘制曲线
plot(ax, ramanShift(:), spectrum(:), ...
    'Color', p.Results.LineColor, ...
    'LineWidth', p.Results.LineWidth);

% 格式化
set(ax, 'FontName', 'Arial', 'FontSize', p.Results.FontSize, ...
    'LineWidth', 1, 'Box', 'on');

xlabel(ax, 'Raman Shift (cm⁻¹)', 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);
ylabel(ax, 'Raman Intensity (a.u.)', 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);

if ~isempty(p.Results.Title)
    title(ax, p.Results.Title, 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);
end

if p.Results.ShowGrid
    grid(ax, 'on');
    set(ax, 'GridAlpha', 0.3);
end

end
