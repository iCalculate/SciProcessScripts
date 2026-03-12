function [fig, ax] = plot_contour_map(PowSerPL, varargin)
%PLOT_CONTOUR_MAP 绘制PL数据等高线图
%   [fig, ax] = plot_contour_map(PowSerPL, Name, Value)
%
%   输入:
%       PowSerPL - PL数据结构
%
%   可选参数:
%       'ColorRange' - 颜色范围 [min max]
%       'ContourLevels' - 等高线层数
%       'Colormap' - 颜色映射
%       'ShowEnergyAxis' - 是否显示能量轴
%       'Title' - 图形标题
%
%   输出:
%       fig - 图形句柄
%       ax - 坐标轴句柄

p = inputParser;
addRequired(p, 'PowSerPL', @isstruct);
addParameter(p, 'ColorRange', [0.8 1.1], @(x) isnumeric(x) && length(x)==2);
addParameter(p, 'ContourLevels', 100, @(x) isnumeric(x) && x > 0);
addParameter(p, 'Colormap', 'jet', @(x) ischar(x) || isstring(x));
addParameter(p, 'ShowEnergyAxis', true, @islogical);
addParameter(p, 'Title', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'FontSize', 12, @(x) isnumeric(x) && x > 0);
addParameter(p, 'LabelFontSize', 14, @(x) isnumeric(x) && x > 0);
parse(p, PowSerPL, varargin{:});

% 数据提取与归一化
Z = double(PowSerPL.data);
x = PowSerPL.axisscale{1,1};
y = PowSerPL.axisscale{2,1};

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

xlabel('Parameter (mW or deg)', 'FontName','Arial','FontSize',p.Results.LabelFontSize);
ylabel('\lambda (nm)','FontName','Arial','FontSize',p.Results.LabelFontSize);

if ~isempty(p.Results.Title)
    title(p.Results.Title, 'FontName','Arial','FontSize',p.Results.LabelFontSize);
end

% 能量轴（可选）
if p.Results.ShowEnergyAxis
    add_energy_axis(ax, y, p.Results.LabelFontSize);
end

yyaxis left

end

function add_energy_axis(ax, y, labelFontSize)
try
    yyaxis left
    yl_vis = get(gca,'YLim');
    yyaxis right
    set(gca,'YColor','k','LineWidth',1,'FontName','Arial', 'Toolbar', [], 'UIContextMenu', []);

    ypos = y(y>0);
    if isempty(ypos)
        yyaxis left
        return;
    end
    
    lam_data_min = min(ypos);
    lam_data_max = max(ypos);

    yl1 = max(min(yl_vis), lam_data_min);
    yl2 = min(max(yl_vis), lam_data_max);
    yl = sort([yl1, yl2]);

    if ~(isfinite(yl(1)) && isfinite(yl(2))) || yl(2) <= yl(1)
        yl = [lam_data_min, lam_data_max];
    end

    eps_lam = 1e-3;
    yl(1) = max(yl(1), eps_lam);

    Emin = 1240/yl(2);
    Emax = 1240/yl(1);
    if ~isfinite(Emin) || ~isfinite(Emax) || Emax <= Emin
        yl = [lam_data_min, max(lam_data_max, lam_data_min+eps_lam)];
        Emin = 1240/yl(2);
        Emax = 1240/yl(1);
    end

    nTicks = 6;
    E_ticks = linspace(Emin, Emax, nTicks);
    lambda_ticks = 1240 ./ E_ticks;
    mask = isfinite(lambda_ticks) & lambda_ticks >= yl(1) & lambda_ticks <= yl(2);
    lambda_ticks = lambda_ticks(mask);
    E_labels = E_ticks(mask);

    [lambda_ticks, idx] = sort(lambda_ticks, 'ascend');
    E_labels = E_labels(idx);

    set(gca, 'YLim', yl, 'YTick', lambda_ticks, ...
        'YTickLabel', arrayfun(@(v) sprintf('%.2f', v), E_labels, 'UniformOutput', false));
    ylabel('E_{ph} (eV)', 'FontName','Arial','FontSize',labelFontSize);
    
catch
    yyaxis left
end
end
