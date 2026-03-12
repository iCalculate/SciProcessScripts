%% Raman-Analyzer 快速开始
%
% 这个脚本演示如何快速使用拉曼光谱分析工具集

clear; clc; close all;
addpath(genpath('.'));

fprintf('📚 Raman-Analyzer 快速开始\n');
fprintf('=====================================\n\n');

%% =========== 项目结构 ===========
fprintf('项目采用 PL-Analyzer 的模块化架构：\n\n');
fprintf('core/data/           → 数据导入\n');
fprintf('core/visualization/  → 绘图函数\n');
fprintf('examples/            → 工作流示例\n\n');

%% =========== 最基础: 单条谱线 ===========
fprintf('【用例1】绘制单条拉曼光谱\n');
fprintf('-------------------------------------\n');

ramanShift = 0:3000;
spectrum = 500 * exp(-((ramanShift - 1350).^2) / 20000) + ...
           1000 * exp(-((ramanShift - 1580).^2) / 30000);

[fig, ax] = plot_single_raman(spectrum, ramanShift, ...
    'Title', 'Single Raman Spectrum', ...
    'LineWidth', 2);

fprintf('✓ 完成！\n\n');

%% =========== 中级: 创建完整数据 ===========
fprintf('【用例2】创建和可视化数据集\n');
fprintf('-------------------------------------\n');

powers = 0.1:0.1:1.0;
ramanShift = 0:3000;

% 创建数据
spectra = zeros(length(powers), length(ramanShift));
for i = 1:length(powers)
    factor = powers(i)^1.2;
    spectra(i, :) = factor * 500 * exp(-((ramanShift - 1350).^2) / 20000) + ...
                    factor * 1000 * exp(-((ramanShift - 1580).^2) / 30000);
end

% 标准格式
RamanData.data = spectra;
RamanData.axisscale{1,1} = powers';
RamanData.axisscale{2,1} = ramanShift';

% 绘图
figure('Position', [100 100 900 600]);
plot_raman_contour(RamanData, 'Title', 'Raman Map');

fprintf('✓ 完成！\n\n');

%% =========== 高级: 堆叠分析 ===========
fprintf('【用例3】堆叠谱线和分析\n');
fprintf('-------------------------------------\n');

figure('Position', [1050 100 900 600]);
plot_raman_stack(RamanData, ...
    'NormalizeMethod', 'max', ...
    'Spacing', 0.15, ...
    'Colormap', 'power', ...
    'Title', 'Raman Evolution');

fprintf('✓ 完成！\n\n');

%% =========== 快速参考 ===========
fprintf('=====================================\n');
fprintf('📌 API 快速参考\n');
fprintf('=====================================\n\n');

fprintf('【数据导入】\n');
fprintf('  RamanData = import_raman_data(''raman.mat'');\n');
fprintf('  RamanData = load_raman_power_data(''raman.mat'', ''power.mat'');\n\n');

fprintf('【绘图函数】\n');
fprintf('  plot_single_raman(spectrum, ramanShift, ''Title'', ''...'');\n');
fprintf('  plot_raman_stack(RamanData, ''Spacing'', 0.15);\n');
fprintf('  plot_raman_contour(RamanData, ''Colormap'', ''hot'');\n\n');

fprintf('【可选参数】\n');
fprintf('  NormalizeMethod: ''none''/''max''/''peak_range''/''area''\n');
fprintf('  RamanRange: [min max]    - 指定拉曼位移范围\n');
fprintf('  Colormap: ''power''/''index''/''jet''/''hot'' 等\n\n');

fprintf('【常见工作流】\n');
fprintf('  参考 examples/ 文件夹:\n');
fprintf('  ├─ basic_plotting.m      (基础用法)\n');
fprintf('  └─ power_dependence.m    (功率分析)\n\n');

fprintf('✅ 快速开始完成！\n');
fprintf('\n下一步: 查看 examples/ 目录下的完整示例\n');
