function [fig, ax] = plot_raman_contour(RamanData, varargin)
%PLOT_RAMAN_CONTOUR 绘制拉曼数据等高线图
%   [fig, ax] = plot_raman_contour(RamanData, Name, Value)
%
%   输入:
%       RamanData - 拉曼数据结构
%
%   可选参数:
%       'ColorRange' - 颜色范围 [min max]
%       'ContourLevels' - 等高线层数
%       'Colormap' - 颜色映射
%       'Title' - 图形标题
%
%   输出:
%       fig - 图形句柄
%       ax - 坐标轴句柄

p = inputParser;
addRequired(p, 'RamanData', @isstruct);
addParameter(p, 'ColorRange', [0.5 1.0], @(x) isnumeric(x) && length(x)==2);
addParameter(p, 'ContourLevels', 100, @(x) isnumeric(x) && x > 0);
addParameter(p, 'Colormap', 'jet', @(x) ischar(x) || isstring(x));
addParameter(p, 'Title', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'FontSize', 12, @(x) isnumeric(x) && x > 0);
addParameter(p, 'LabelFontSize', 14, @(x) isnumeric(x) && x > 0);
parse(p, RamanData, varargin{:});

% 数据提取与归一化
Z = double(RamanData.data);
x = RamanData.axisscale{1,1};      % 参数 (mW等)
y = RamanData.axisscale{2,1};      % 拉曼位移 (cm⁻¹)

rowMax = max(Z, [], 2);
rowMax(rowMax==0) = 1;
Znorm = Z ./ rowMax;

% 创建图形
fig = figure('Color','w');

% 检查X轴是否单调
x_sorted = issorted(x) || issorted(x, 'descend');
if x_sorted
    [~,h] = contourf(x, y, Znorm', p.Results.ContourLevels);
    set(h, 'LineStyle', 'none');
else
    imagesc(x, y, Znorm');
    axis xy;
end

colormap(p.Results.Colormap);
caxis(p.Results.ColorRange);

% 轴设置
ax = gca;
set(ax,'FontName','Arial','FontSize',p.Results.FontSize,'LineWidth',1,'Layer','top','Box','on', ...
    'Toolbar', [], 'UIContextMenu', []);

xlabel('Parameter (mW or K)', 'FontName','Arial','FontSize',p.Results.LabelFontSize);
ylabel('Raman Shift (cm⁻¹)','FontName','Arial','FontSize',p.Results.LabelFontSize);

if ~isempty(p.Results.Title)
    title(p.Results.Title, 'FontName','Arial','FontSize',p.Results.LabelFontSize);
end

% 添加颜色条
c = colorbar;
c.Label.String = 'Normalized Intensity';
c.Label.FontSize = p.Results.LabelFontSize;

end
