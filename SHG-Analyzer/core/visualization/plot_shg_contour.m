function [fig, ax] = plot_shg_contour(SHGData, varargin)
%PLOT_SHG_CONTOUR 绘制SHG二维等高线热图
%   [fig, ax] = plot_shg_contour(SHGData, Name, Value)
%
%   用于绘制二次谐波(SHG)强度的二维等高线热图，展示参数(如功率)
%   与波长的依赖关系
%
%   输入:
%       SHGData - 包含以下字段的结构体:
%           .data - (N×M) 矩阵，行为序列数(功率/温度等)，列为波长点数
%           .axisscale{1,1} - (N×1) 第一轴参数向量
%           .axisscale{2,1} - (M×1) 波长向量 (nm)
%
%   可选参数:
%       'WavelengthRange' - 波长范围 [min max] nm，默认全部
%       'ColorRange' - 强度范围 [min max]，默认自动
%       'ContourLevels' - 等高线层数，默认 30
%       'Colormap' - 颜色映射，默认 'hot'
%       'ShowContourLines' - 是否显示等高线，默认 false
%       'Title' - 图形标题
%       'FontSize' - 字体大小，默认 12
%       'LabelFontSize' - 标签字体大小，默认 14
%       'ParamLabel' - 第一轴标签，默认 'Parameter'
%
%   输出:
%       fig - 图形句柄
%       ax - 坐标轴句柄
%
%   示例:
%       powers = 0.1:0.1:2.0;
%       wavelength = 400:800;
%       data = randn(length(powers), length(wavelength));
%       SHGData.data = data;
%       SHGData.axisscale{1,1} = powers';
%       SHGData.axisscale{2,1} = wavelength';
%       plot_shg_contour(SHGData, 'Colormap', 'hot', 'Title', 'SHG Map');

p = inputParser;
addRequired(p, 'SHGData', @isstruct);
addParameter(p, 'WavelengthRange', [], @(x) isempty(x) || (isvector(x) && length(x) == 2));
addParameter(p, 'ColorRange', [], @(x) isempty(x) || (isvector(x) && length(x) == 2));
addParameter(p, 'ContourLevels', 30, @(x) isnumeric(x) && x > 0);
addParameter(p, 'Colormap', 'hot', @(x) ischar(x) || isstring(x));
addParameter(p, 'ShowContourLines', false, @islogical);
addParameter(p, 'Title', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'FontSize', 12, @(x) isnumeric(x) && x > 0);
addParameter(p, 'LabelFontSize', 14, @(x) isnumeric(x) && x > 0);
addParameter(p, 'ParamLabel', 'Parameter (a.u.)', @(x) ischar(x) || isstring(x));
parse(p, SHGData, varargin{:});

% 数据提取
data = SHGData.data;
wavelength = SHGData.axisscale{2,1}(:);
paramAxis = SHGData.axisscale{1,1}(:);

% 波长范围筛选
if isempty(p.Results.WavelengthRange)
    wl_idx = 1:length(wavelength);
else
    wl_idx = wavelength >= p.Results.WavelengthRange(1) & wavelength <= p.Results.WavelengthRange(2);
end

wavelength_plot = wavelength(wl_idx);
data_plot = data(:, wl_idx);

% 创建图形
fig = figure('Color', 'w', 'Position', [100 100 1000 600]);
ax = axes('Parent', fig);

% 绘制等高线图
if isempty(p.Results.ColorRange)
    [C, h] = contourf(ax, wavelength_plot, paramAxis, data_plot, p.Results.ContourLevels, 'LineStyle', 'none');
else
    levels = linspace(p.Results.ColorRange(1), p.Results.ColorRange(2), p.Results.ContourLevels);
    [C, h] = contourf(ax, wavelength_plot, paramAxis, data_plot, levels, 'LineStyle', 'none');
end

% 添加等高线
if p.Results.ShowContourLines
    contour(ax, wavelength_plot, paramAxis, data_plot, 15, 'black', 'LineWidth', 0.5);
end

% 应用颜色映射
colormap(ax, p.Results.Colormap);
cb = colorbar(ax);
ylabel(cb, 'Intensity (a.u.)', 'FontSize', p.Results.FontSize);

% 格式化
set(ax, 'FontName', 'Arial', 'FontSize', p.Results.FontSize, ...
    'LineWidth', 1.5, 'Box', 'on');

xlabel(ax, 'Wavelength (nm)', 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);
ylabel(ax, p.Results.ParamLabel, 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);

if ~isempty(p.Results.Title)
    title(ax, p.Results.Title, 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);
end

end
