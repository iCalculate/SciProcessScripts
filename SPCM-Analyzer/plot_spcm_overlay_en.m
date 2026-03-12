%% SPCM Data Visualization Script - Heatmap Overlay on Scan Image
% This script reads SPCM scan data and overlays it as a semi-transparent heatmap
% on the scan region image
% 
% Input files:
%   - mapping.csv: SPCM scan data (x_mm, y_mm, I)
%   - image_1.png: Scan region photograph
%
% Output: Overlay image and high-resolution heatmaps

clear; close all; clc;

%% Parameter Settings
script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, 'mapping.csv');
image_file = fullfile(script_dir, '图片1.png');

% Heatmap parameters
transparency_alpha = 0.6;        % Transparency (0-1)
figure_dpi = 300;                % Output figure resolution
colormap_style = 'hot';          % Colormap style: 'hot', 'jet', 'parula', 'turbo', etc.
gaussian_sigma_heatmap = 2.0;    % Heatmap Gaussian blur sigma (0 = no blur, range 0-5)
gaussian_sigma_image = 3.0;      % Background image Gaussian blur sigma (0 = no blur, range 0-3)
hue_shift = 20;                  % Hue rotation (degrees, -180 to 180)

% Figure parameters
fig_width = 800;                 % Figure width (pixels)
fig_height = 700;                % Figure height (pixels)
font_size_label = 16;            % Axis label font size
font_size_tick = 14;             % Tick label font size
font_size_colorbar = 14;         % Colorbar label font size

%% Step 1: Read CSV Data
data = readtable(csv_file);
x = data.x_mm;
y = data.y_mm;
intensity = data.I;

fprintf('Data Statistics:\n');
fprintf('  Scan Range: X = [%.1f, %.1f] mm, Y = [%.1f, %.1f] mm\n', ...
    min(x), max(x), min(y), max(y));
fprintf('  Number of Data Points: %d\n', length(x));
fprintf('  Intensity Range: [%.2f, %.2f]\n', min(intensity), max(intensity));

%% Step 2: Interpolate Discrete Data to Grid
% Create high-resolution grid for smooth heatmap
[xi, yi] = meshgrid(linspace(min(x), max(x), 200), ...
                     linspace(min(y), max(y), 200));

% Use scattered data interpolation method
zi = griddata(x, y, intensity, xi, yi, 'cubic');

% Apply Gaussian blur if enabled
if gaussian_sigma_heatmap > 0
    zi = imgaussfilt(zi, gaussian_sigma_heatmap);
    fprintf('Applied heatmap Gaussian blur, sigma = %.2f\n', gaussian_sigma_heatmap);
end

%% Step 3: Load Scan Region Image
if isfile(image_file)
    img = imread(image_file);
    fprintf('Loaded scan region image: %s\n', image_file);
    fprintf('  Image size: %d × %d × %d\n', size(img, 1), size(img, 2), size(img, 3));
else
    error('Image file not found: %s', image_file);
end

% Convert to double and normalize to (0-1)
if isa(img, 'uint8')
    img = double(img) / 255;
elseif isa(img, 'uint16')
    img = double(img) / 65535;
end

% Apply Gaussian blur to background image if enabled
if gaussian_sigma_image > 0
    for c = 1:size(img, 3)
        img(:,:,c) = imgaussfilt(img(:,:,c), gaussian_sigma_image);
    end
    fprintf('Applied background image Gaussian blur, sigma = %.2f\n', gaussian_sigma_image);
end

% Apply hue rotation if enabled
if hue_shift ~= 0 && size(img, 3) == 3
    img_hsv = rgb2hsv(img);
    img_hsv(:,:,1) = mod(img_hsv(:,:,1) + hue_shift/360, 1);
    img = hsv2rgb(img_hsv);
    fprintf('Applied hue rotation, rotation angle = %.1f°\n', hue_shift);
end

img_height = size(img, 1);
img_width = size(img, 2);

%% Step 4: User Selection Menu
fprintf('\n=== Select Visualization Type ===\n');
choice = menu('Which visualization would you like to generate?', ...
    'Heatmap Overlay on Scan Image', ...
    'Pure Heatmap (without background)', ...
    'Contour Map');

output_dir = script_dir;

%% Step 5: Generate Selected Visualization

switch choice
    case 1
        % Heatmap Overlaid on Image
        fprintf('\nGenerating: Heatmap Overlay on Scan Image...\n');
        fig1 = figure('Name', 'SPCM Heatmap Overlay', 'NumberTitle', 'off', ...
                      'Position', [100, 100, fig_width, fig_height]);
        
        % Display scan region image
        imagesc([min(x), max(x)], [min(y), max(y)], img);
        axis image xy;
        hold on;
        
        % Create heatmap layer
        h = pcolor(xi, yi, zi);
        set(h, 'EdgeColor', 'none', 'FaceAlpha', transparency_alpha);
        
        % Set colormap
        colormap(colormap_style);
        cbar = colorbar('eastoutside');
        ylabel(cbar, 'Photocurrent (\muA)', 'FontSize', font_size_colorbar);
        caxis([min(intensity), max(intensity)]);
        
        % Labels
        xlabel('X Position (\mum)', 'FontSize', font_size_label);
        ylabel('Y Position (\mum)', 'FontSize', font_size_label);
        
        % Grid and appearance
        grid on;
        set(gca, 'FontSize', font_size_tick);
        set(gca, 'GridAlpha', 0.3);
        
        hold off;
        
        % Save
        fig_file = fullfile(output_dir, 'SPCM_overlay_result.png');
        exportgraphics(fig1, fig_file, 'Resolution', figure_dpi);
        fprintf('✓ Saved: %s\n', fig_file);
        
    case 2
        % Pure Heatmap
        fprintf('\nGenerating: Pure Heatmap (without background)...\n');
        fig2 = figure('Name', 'SPCM Heatmap (Pure Heatmap)', 'NumberTitle', 'off', ...
                      'Position', [100, 100, fig_width, fig_height]);
        
        pcolor(xi, yi, zi);
        set(gca, 'XDir', 'normal', 'YDir', 'normal');
        shading interp;
        axis image;
        
        colormap(colormap_style);
        cbar2 = colorbar;
        ylabel(cbar2, 'Photocurrent (\muA)', 'FontSize', font_size_colorbar);
        caxis([min(intensity), max(intensity)]);
        
        xlabel('X Position (\mum)', 'FontSize', font_size_label);
        ylabel('Y Position (\mum)', 'FontSize', font_size_label);
        
        grid on;
        set(gca, 'FontSize', font_size_tick);
        set(gca, 'GridAlpha', 0.2);
        
        % Save
        fig_file = fullfile(output_dir, 'SPCM_heatmap_only.png');
        exportgraphics(fig2, fig_file, 'Resolution', figure_dpi);
        fprintf('✓ Saved: %s\n', fig_file);
        
    case 3
        % Contour Map
        fprintf('\nGenerating: Contour Map...\n');
        fig3 = figure('Name', 'SPCM Contour Map', 'NumberTitle', 'off', ...
                      'Position', [100, 100, fig_width, fig_height]);
        
        % Plot contours
        [C, h_contour] = contourf(xi, yi, zi, 15, 'LineColor', 'none');
        shading interp;
        hold on;
        h_contour_lines = contour(xi, yi, zi, 10, 'LineColor', 'black', 'LineWidth', 0.5);
        set(h_contour_lines, 'LineAlpha', 0.3);  % Set transparency
        
        colormap(colormap_style);
        cbar3 = colorbar;
        ylabel(cbar3, 'Photocurrent (\muA)', 'FontSize', font_size_colorbar);
        
        xlabel('X Position (\mum)', 'FontSize', font_size_label);
        ylabel('Y Position (\mum)', 'FontSize', font_size_label);
        
        axis image;
        grid on;
        set(gca, 'FontSize', font_size_tick);
        hold off;
        
        % Save
        fig_file = fullfile(output_dir, 'SPCM_contour_map.png');
        exportgraphics(fig3, fig_file, 'Resolution', figure_dpi);
        fprintf('✓ Saved: %s\n', fig_file);
        
    otherwise
        fprintf('\nNo selection made. Exiting.\n');
        return;
end

%% Complete
fprintf('\n✓ Plotting complete!\n');
