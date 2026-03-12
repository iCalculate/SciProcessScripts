%% PL-Analyzer 快速开始指南
%
% 这个脚本演示如何快速使用 PL-Analyzer 工具集
% 
% 按照以下步骤了解整个工作流程
% ========================================================================

clear; clc; close all;
addpath(genpath('.'));

%% =========== 步骤0: 理解项目结构 ===========
% 项目分为三层:
% 
% 第1层 (core/data/)          → 数据导入
%   └─ import_pl_data()
%   └─ load_pl_power_data()
%   └─ find_data_files()
%
% 第2层 (core/visualization/) → 可视化
%   └─ plot_single_spectrum()   ⭐ 最基础
%   └─ plot_spectra_stack()
%   └─ plot_contour_map()
%
% 第3层 (core/analysis/)      → 数据分析
%   └─ normalize_spectrum()     (标准化)
%   └─ peak_analysis()          (峰检测)
%   └─ peak_fitting()           (多峰拟合)
%   └─ polarization_analysis()  (偏振分析)

fprintf('📚 PL-Analyzer 快速开始\n');
fprintf('=====================================\n\n');

%% =========== 步骤1: 最简单 - 绘制单条光谱 ===========
fprintf('Step 1: 绘制单条光谱 (最基础)\n');
fprintf('-------------------------------------\n');

% 打开文件对话框让用户选择光谱数据
[fileName, filePath] = uigetfile({'*.mat';'*.csv';'*.*'}, '请选择光谱数据文件');

if fileName == 0
    fprintf('用户取消了文件选择，脚本中止\n');
    return;
end

fullPath = fullfile(filePath, fileName);
fprintf('正在加载数据: %s\n', fileName);

% 加载数据
try
    data = load(fullPath);
    
    % 自动识别数据结构
    if isstruct(data)
        fieldNames = fieldnames(data);
        if ismember('spectrum', fieldNames) && ismember('wavelength', fieldNames)
            spectrum = data.spectrum;
            wavelength = data.wavelength;
        elseif ismember('data', fieldNames) && ismember('wavelength', fieldNames)
            spectrum = data.data;
            wavelength = data.wavelength;
        else
            spectrum = data.(fieldNames{1});
            if length(fieldNames) > 1
                wavelength = data.(fieldNames{2});
            else
                wavelength = 1:length(spectrum);
            end
        end
    else
        if size(data, 2) >= 2
            wavelength = data(:, 1)';
            spectrum = data(:, 2)';
        else
            spectrum = data(:)';
            wavelength = 1:length(spectrum);
        end
    end
    
    spectrum = spectrum(:)';
    wavelength = wavelength(:)';
    fprintf('✓ 数据加载成功 (点数: %d)\n\n', length(spectrum));
    
catch ME
    fprintf('✗ 加载文件失败: %s\n', ME.message);
    return;
end

% 调用最基础的函数
[fig1, ax1] = plot_single_spectrum(spectrum, wavelength, ...
    'Title', 'Single PL Spectrum', ...
    'LineColor', 'b', ...
    'LineWidth', 2);

fprintf('✓ 绘制了光谱！\n\n');

%% =========== 步骤2: 中级 - 创建完整的数据集 ===========
fprintf('Step 2: 可视化数据\n');
fprintf('-------------------------------------\n');

% 如果用户有多条光谱的数据文件，可以直接使用
% 否则使用已加载的单条谱线构建数据集
PL_Data.data = spectrum;  % 单条谱线
PL_Data.axisscale{1,1} = 1;              % 参数轴（示例）
PL_Data.axisscale{2,1} = wavelength';    % 波长轴

fprintf('✓ 数据已准备就绪\n\n');

%% =========== 步骤3: 数据分析 ===========
fprintf('Step 3: 数据分析\n');
fprintf('-------------------------------------\n');

% 对加载的光谱进行分析
testSpectrum = spectrum;
testWavelength = wavelength;

% 3.1 峰值分析
fprintf('  分析中...\n');
[peaks, positions, widths, amplitudes] = peak_analysis(testSpectrum, testWavelength, ...
    'MinPeakHeight', 0, ...
    'MinPeakDistance', 20, ...
    'Plot', false);

fprintf('  ├─ ✓ 峰值分析: 找到 %d 个峰\n', length(peaks));

% 3.2 峰拟合
if length(peaks) > 0
    numPeaksToFit = min(length(peaks), 3);  % 最多拟合3个峰
else
    numPeaksToFit = 1;
end

[fitParams, fittedCurve, residuals, r2] = peak_fitting(testSpectrum, testWavelength, numPeaksToFit, ...
    'FitType', 'gauss', ...
    'Plot', true, ...
    'Verbose', true);

fprintf('  └─ ✓ 多峰拟合: R² = %.4f\n\n', r2);

%% =========== 步骤4: 工作流总结 ===========
fprintf('Step 4: 工作流总结\n');
fprintf('-------------------------------------\n');

fprintf('常见用例:\n\n');

fprintf('【用例1】我有单个光谱文件\n');
fprintf('  → 使用 plot_single_spectrum()\n');
fprintf('  → 参考: examples/basic_plotting.m\n\n');

fprintf('【用例2】我要分析功率依赖关系\n');
fprintf('  1. 加载: load_pl_power_data()\n');
fprintf('  2. 绘图: plot_contour_map() + plot_spectra_stack()\n');
fprintf('  3. 拟合: 功率 vs 强度\n');
fprintf('  → 参考: examples/power_dependence.m\n\n');

fprintf('【用例3】我要进行分峰和拟合\n');
fprintf('  1. 自动检测: peak_analysis()\n');
fprintf('  2. 多峰拟合: peak_fitting()\n');
fprintf('  3. 绘制对比\n');
fprintf('  → 参考: examples/peak_fitting_example.m\n\n');

fprintf('【用例4】我要分析偏振各向异性\n');
fprintf('  1. 加载偏振扫描数据\n');
fprintf('  2. 绘图: plot_contour_map() + plot_spectra_stack()\n');
fprintf('  3. 拟合: I(θ) ∝ cos²(2θ)\n');
fprintf('  → 参考: examples/polarization_analysis_example.m\n\n');

%% =========== 关键概念 ===========
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('📌 关键概念\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

fprintf('【数据结构】\n');
fprintf('所有函数使用统一的 PL_Data 结构:\n');
fprintf('  PL_Data.data           - 光谱矩阵 (行: 序列, 列: 波长)\n');
fprintf('  PL_Data.axisscale{1,1} - 参数轴 (功率/角度/温度等)\n');
fprintf('  PL_Data.axisscale{2,1} - 波长轴 (nm)\n\n');

fprintf('【三层架构】\n');
fprintf('  Layer 1 (基础)    - 单个光谱处理\n');
fprintf('  Layer 2 (可视化)  - 多光谱绘图\n');
fprintf('  Layer 3 (分析)    - 复杂数据分析\n\n');

fprintf('【标准参数】\n');
fprintf('  NormalizeMethod: none/max/peak_range/area\n');
fprintf('  Colormap: power/index/jet/hot/viridis等\n');
fprintf('  Plot: true/false (绘图开关)\n\n');

%% =========== 下一步 ===========
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('🚀 后续建议\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

fprintf('1. 查看 examples/ 目录下的具体示例\n');
fprintf('   ✓ basic_plotting.m           (基础用法)\n');
fprintf('   ✓ peak_fitting_example.m     (分峰拟合)\n');
fprintf('   ✓ power_dependence.m         (功率分析)\n');
fprintf('   ✓ polarization_analysis_example.m (偏振分析)\n\n');

fprintf('2. 查阅 README.md 了解完整 API\n\n');

fprintf('3. 根据自己的需求，组合使用各模块\n\n');

fprintf('✅ 快速开始指南完成！\n\n');
