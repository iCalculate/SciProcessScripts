function [fig, ax] = plot_shg_stack(SHGData, varargin)
%PLOT_SHG_STACK 绘制多条SHG光谱堆叠图
%   [fig, ax] = plot_shg_stack(SHGData, Name, Value)
%
%   用于绘制多条二次谐波(SHG)光谱的堆叠图，便于观察谱线的演变
%
%   输入:
%       SHGData - 包含以下字段的结构体:
%           .data - (N×M) 矩阵，行为序列数(功率/温度等)，列为波长点数
%           .axisscale{1,1} - (N×1) 第一轴参数向量(如功率 mW)
%           .axisscale{2,1} - (M×1) 波长向量 (nm)
%
%   可选参数:
%       'Indices' - 要绘制的行索引，默认为全部
%       'WavelengthRange' - 波长范围 [min max] nm，默认全部
%       'NormalizeMethod' - 归一化方法: 'none'/'max'/'peak_range'/'area'，默认 'max'
%       'PeakRange' - 峰值范围 [min max] nm，默认 [300 800]
%       'Spacing' - 曲线间距(相对强度)，默认 0.1
%       'Colormap' - 颜色模式: 'power'/'index'/'jet'/'hot'/'cool'，默认 'power'
%       'Title' - 图形标题
%       'FontSize' - 字体大小，默认 12
%       'LabelFontSize' - 标签字体大小，默认 14
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
%       plot_shg_stack(SHGData, 'NormalizeMethod', 'max', 'Spacing', 0.15);

p = inputParser;
addRequired(p, 'SHGData', @isstruct);
addParameter(p, 'Indices', [], @(x) isempty(x) || isvector(x));
addParameter(p, 'WavelengthRange', [], @(x) isempty(x) || (isvector(x) && length(x) == 2));
addParameter(p, 'NormalizeMethod', 'max', @(x) ismember(x, {'none', 'max', 'peak_range', 'area'}));
addParameter(p, 'PeakRange', [300 800], @(x) isvector(x) && length(x) == 2);
addParameter(p, 'Spacing', 0.1, @(x) isnumeric(x) && x > 0);
addParameter(p, 'Colormap', 'power', @(x) ismember(x, {'power', 'index', 'jet', 'hot', 'cool'}));
addParameter(p, 'Title', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'FontSize', 12, @(x) isnumeric(x) && x > 0);
addParameter(p, 'LabelFontSize', 14, @(x) isnumeric(x) && x > 0);
parse(p, SHGData, varargin{:});

% 数据提取
data = SHGData.data;
wavelength = SHGData.axisscale{2,1}(:);
paramAxis = SHGData.axisscale{1,1}(:);

% 确定要绘制的行索引
if isempty(p.Results.Indices)
    indices = 1:size(data, 1);
else
    indices = p.Results.Indices(:);
end

% 波长范围筛选
if isempty(p.Results.WavelengthRange)
    wl_idx = 1:length(wavelength);
else
    wl_idx = wavelength >= p.Results.WavelengthRange(1) & wavelength <= p.Results.WavelengthRange(2);
end

wavelength_plot = wavelength(wl_idx);
data_plot = data(indices, wl_idx);

% 归一化处理
data_norm = data_plot;
switch p.Results.NormalizeMethod
    case 'max'
        for i = 1:size(data_norm, 1)
            max_val = max(data_norm(i, :));
            if max_val > 0
                data_norm(i, :) = data_norm(i, :) / max_val;
            end
        end
    case 'peak_range'
        peak_idx = wavelength_plot >= p.Results.PeakRange(1) & wavelength_plot <= p.Results.PeakRange(2);
        for i = 1:size(data_norm, 1)
            max_val = max(data_norm(i, peak_idx));
            if max_val > 0
                data_norm(i, :) = data_norm(i, :) / max_val;
            end
        end
    case 'area'
        for i = 1:size(data_norm, 1)
            area_val = trapz(wavelength_plot, data_norm(i, :));
            if area_val > 0
                data_norm(i, :) = data_norm(i, :) / area_val;
            end
        end
end

% 创建颜色映射
n_curves = length(indices);
switch p.Results.Colormap
    case 'power'
        colors = parula(n_curves);
    case 'index'
        colors = jet(n_curves);
    case 'jet'
        colors = jet(n_curves);
    case 'hot'
        colors = hot(n_curves);
    case 'cool'
        colors = cool(n_curves);
end

% 创建图形
fig = figure('Color', 'w');
ax = axes('Parent', fig);
hold(ax, 'on');

% 绘制堆叠曲线
for i = 1:n_curves
    y_offset = (i - 1) * p.Results.Spacing;
    plot(ax, wavelength_plot, data_norm(i, :) + y_offset, ...
        'Color', colors(i, :), ...
        'LineWidth', 2);
end

hold(ax, 'off');

% 格式化
set(ax, 'FontName', 'Arial', 'FontSize', p.Results.FontSize, ...
    'LineWidth', 1.5, 'Box', 'on');

xlabel(ax, 'Wavelength (nm)', 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);
ylabel(ax, 'SHG Intensity (a.u.)', 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);

% 添加参数轴标签
param_vals = paramAxis(indices);
legend_labels = arrayfun(@(x) sprintf('%.2f', x), param_vals, 'UniformOutput', false);
legend(ax, legend_labels, 'Location', 'best', 'FontSize', p.Results.FontSize - 2);

if ~isempty(p.Results.Title)
    title(ax, p.Results.Title, 'FontName', 'Arial', 'FontSize', p.Results.LabelFontSize);
end

grid(ax, 'on');
set(ax, 'GridAlpha', 0.3);

end
