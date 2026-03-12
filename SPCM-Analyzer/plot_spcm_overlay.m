%% SPCM数据可视化脚本 - 半透明热图叠加于扫描图像
% 此脚本读取SPCM扫描数据并将其作为半透明热图叠加在扫描区域图像上
% 
% 输入文件：
%   - mapping.csv: SPCM扫描数据 (x_mm, y_mm, I)
%   - 图片1.png: 扫描区域照片
%
% 输出：叠加图像和高分辨率热图

clear; close all; clc;

%% 参数设置
script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, 'mapping.csv');
image_file = fullfile(script_dir, '图片1.png');

% 热图参数
transparency_alpha = 0.6;        % 透明度 (0-1)
figure_dpi = 150;                % 输出图分辨率
colormap_style = 'hot';          % 颜色方案: 'hot', 'jet', 'parula', 'turbo' 等
gaussian_sigma_heatmap = 2.0;    % 热图高斯模糊标准差（0表示不模糊，范围推荐0-5）
gaussian_sigma_image = 3.0;      % 背景图像高斯模糊标准差（0表示不模糊，范围推荐0-3）
hue_shift = 20;                  % 色相旋转 (度数, -180到180)

%% 第一步：读取CSV数据
data = readtable(csv_file);
x = data.x_mm;
y = data.y_mm;
intensity = data.I;

fprintf('数据统计信息：\n');
fprintf('  扫描范围: X = [%.1f, %.1f] mm, Y = [%.1f, %.1f] mm\n', ...
    min(x), max(x), min(y), max(y));
fprintf('  数据点数: %d\n', length(x));
fprintf('  强度范围: [%.2f, %.2f]\n', min(intensity), max(intensity));

%% 第二步：将离散数据插值到网格
% 创建高分辨率网格以获得光滑的热图
[xi, yi] = meshgrid(linspace(min(x), max(x), 200), ...
                     linspace(min(y), max(y), 200));

% 使用散点插值方法
zi = griddata(x, y, intensity, xi, yi, 'cubic');

% 应用高斯模糊（如果启用）
if gaussian_sigma_heatmap > 0
    zi = imgaussfilt(zi, gaussian_sigma_heatmap);
    fprintf('已应用热图高斯模糊，标准差 = %.2f\n', gaussian_sigma_heatmap);
end

%% 第三步：加载扫描区域图像
if isfile(image_file)
    img = imread(image_file);
    fprintf('已加载扫描区域图像: %s\n', image_file);
    fprintf('  图像尺寸: %d × %d × %d\n', size(img, 1), size(img, 2), size(img, 3));
else
    error('找不到图像文件: %s', image_file);
end

% 转换为double并归一化 (0-1)
if isa(img, 'uint8')
    img = double(img) / 255;
elseif isa(img, 'uint16')
    img = double(img) / 65535;
end

% 应用高斯模糊到图像（可选）
if gaussian_sigma_image > 0
    for c = 1:size(img, 3)
        img(:,:,c) = imgaussfilt(img(:,:,c), gaussian_sigma_image);
    end
    fprintf('已应用背景图像高斯模糊，标准差 = %.2f\n', gaussian_sigma_image);
end

% 应用色相旋转（如果启用）
if hue_shift ~= 0 && size(img, 3) == 3
    img_hsv = rgb2hsv(img);
    img_hsv(:,:,1) = mod(img_hsv(:,:,1) + hue_shift/360, 1);
    img = hsv2rgb(img_hsv);
    fprintf('已应用色相旋转，旋转角度 = %.1f°\n', hue_shift);
end

img_height = size(img, 1);
img_width = size(img, 2);

%% 第四步：创建主图 - 热图叠加于图像上
fig1 = figure('Name', 'SPCM数据热图叠加', 'NumberTitle', 'off', ...
              'Position', [100, 100, 1000, 900]);

% 显示扫描区域图像
imagesc([min(x), max(x)], [min(y), max(y)], img);
axis image xy;
hold on;

% 创建热图层
h = pcolor(xi, yi, zi);
set(h, 'EdgeColor', 'none', 'FaceAlpha', transparency_alpha);

% 设置颜色方案
colormap(colormap_style);
cbar = colorbar('eastoutside');
ylabel(cbar, 'SPCM强度 (counts)', 'FontSize', 11, 'FontWeight', 'bold');
caxis([min(intensity), max(intensity)]);

% 标签和标题
xlabel('X 位置 (mm)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Y 位置 (mm)', 'FontSize', 12, 'FontWeight', 'bold');
title('SPCM扫描数据热图 - 半透明叠加', 'FontSize', 14, 'FontWeight', 'bold');

% 网格和外观
grid on;
set(gca, 'FontSize', 10);
set(gca, 'GridAlpha', 0.3);

% 添加数据点标记（可选，取消注释以显示）
% scatter(x, y, 20, intensity, 'filled', 'MarkerEdgeColor', 'k', 'MarkerEdgeAlpha', 0.3);

hold off;

%% 第五步：创建副图 - 纯热图（不含背景图像）
fig2 = figure('Name', 'SPCM热图（纯热图）', 'NumberTitle', 'off', ...
              'Position', [1150, 100, 900, 800]);

pcolor(xi, yi, zi);
set(gca, 'XDir', 'normal', 'YDir', 'normal');
shading interp;
axis image;

colormap(colormap_style);
cbar2 = colorbar;
ylabel(cbar2, 'SPCM强度 (counts)', 'FontSize', 11, 'FontWeight', 'bold');
caxis([min(intensity), max(intensity)]);

xlabel('X 位置 (mm)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Y 位置 (mm)', 'FontSize', 12, 'FontWeight', 'bold');
title('SPCM扫描数据热图', 'FontSize', 14, 'FontWeight', 'bold');

grid on;
set(gca, 'FontSize', 10);
set(gca, 'GridAlpha', 0.2);

hold on;
scatter(x, y, 30, intensity, 'filled', 'MarkerEdgeColor', 'none', ...
        'MarkerFaceAlpha', 0.5);
hold off;

%% 第六步：创建副图 - 扫描数据的轮廓图
fig3 = figure('Name', 'SPCM数据轮廓图', 'NumberTitle', 'off', ...
              'Position', [100, 1050, 900, 800]);

% 绘制等高线
[C, h_contour] = contourf(xi, yi, zi, 15, 'LineColor', 'none');
shading interp;
hold on;
h_contour_lines = contour(xi, yi, zi, 10, 'LineColor', 'black', 'LineWidth', 0.5);
set(h_contour_lines, 'LineAlpha', 0.3);  % 设置透明度

colormap(colormap_style);
cbar3 = colorbar;
ylabel(cbar3, 'SPCM强度 (counts)', 'FontSize', 11, 'FontWeight', 'bold');

xlabel('X 位置 (mm)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Y 位置 (mm)', 'FontSize', 12, 'FontWeight', 'bold');
title('SPCM扫描数据等高线图', 'FontSize', 14, 'FontWeight', 'bold');

scatter(x, y, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.4);

axis image;
grid on;
set(gca, 'FontSize', 10);
hold off;

%% 第七步：保存高分辨率输出图像
output_dir = script_dir;

% 保存叠加图
fig1_file = fullfile(output_dir, 'SPCM_overlay_result.png');
exportgraphics(fig1, fig1_file, 'Resolution', figure_dpi);
fprintf('已保存叠加图: %s\n', fig1_file);

% 保存纯热图
fig2_file = fullfile(output_dir, 'SPCM_heatmap_only.png');
exportgraphics(fig2, fig2_file, 'Resolution', figure_dpi);
fprintf('已保存纯热图: %s\n', fig2_file);

% 保存等高线图
fig3_file = fullfile(output_dir, 'SPCM_contour_map.png');
exportgraphics(fig3, fig3_file, 'Resolution', figure_dpi);
fprintf('已保存等高线图: %s\n', fig3_file);

%% 完成
fprintf('\n✓ 绘图完成！已生成3张图像文件。\n');
fprintf('  - SPCM_overlay_result.png: 热图叠加于扫描图像\n');
fprintf('  - SPCM_heatmap_only.png: 纯热图视图\n');
fprintf('  - SPCM_contour_map.png: 等高线图视图\n');
