%% 多曲线 PL 数据绘制工具
% 
% 这个脚本允许用户：
% 1. 加载PL数据系列 (.mat 文件)
% 2. 交互式地选择要绘制的特定曲线
% 3. 支持多种绘图模式：
%    - 单条曲线绘制
%    - 选中曲线堆叠
%    - 所有曲线等高线图
%    - 多子图对比绘制
%
% 用法:
%   multi_curve_plotter
%
% 数据格式需求:
%   PL_Data.data           - 光谱矩阵 (行: 序列/功率, 列: 波长)
%   PL_Data.axisscale{1,1} - 参数轴 (功率/角度/温度等)
%   PL_Data.axisscale{2,1} - 波长轴 (nm)

clear; clc; close all;
addpath(genpath('.'));

%% ============================================================================
%  初始化和文件选择
% ============================================================================

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║         多曲线 PL 数据绘制工具 v1.0                            ║\n');
fprintf('║  智能加载数据 • 交互式选择 • 灵活绘图                           ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

% 文件选择
fprintf('【步骤 1】选择数据文件\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

[fileName, filePath] = uigetfile({'*.mat';'*.*'}, ...
    '请选择包含 PL 数据的 .mat 文件');

if fileName == 0
    fprintf('❌ 用户取消了文件选择，程序中止\n\n');
    return;
end

fullPath = fullfile(filePath, fileName);
fprintf('✓ 已选择: %s\n', fileName);

%% ============================================================================
%  数据加载和验证
% ============================================================================

fprintf('\n【步骤 2】加载数据\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

try
    % 加载数据
    data = load(fullPath);
    fieldNames = fieldnames(data);
    
    % 自动查找 PL_Data 结构或第一个结构体
    PL_Data = [];
    if ismember('PL_Data', fieldNames)
        PL_Data = data.PL_Data;
    elseif ismember('PowSerPL', fieldNames)
        PL_Data = data.PowSerPL;
    else
        % 使用第一个结构体
        for i = 1:length(fieldNames)
            if isstruct(data.(fieldNames{i}))
                PL_Data = data.(fieldNames{i});
                break;
            end
        end
    end
    
    % 验证数据结构
    if isempty(PL_Data) || ~isstruct(PL_Data)
        error('未找到有效的数据结构');
    end
    
    if ~isfield(PL_Data, 'data') || ~isfield(PL_Data, 'axisscale')
        error('数据结构缺少必要字段: data 或 axisscale');
    end
    
    spectra = PL_Data.data;
    paramAxis = PL_Data.axisscale{1,1}(:);
    wavelengthAxis = PL_Data.axisscale{2,1}(:);
    
    % 数据信息
    numCurves = size(spectra, 1);
    numWavelengths = size(spectra, 2);
    
    fprintf('✓ 数据加载成功\n');
    fprintf('  ├─ 曲线数量: %d\n', numCurves);
    fprintf('  ├─ 波长点数: %d\n', numWavelengths);
    fprintf('  ├─ 参数范围: %.4f - %.4f\n', min(paramAxis), max(paramAxis));
    fprintf('  └─ 波长范围: %.0f - %.0f nm\n\n', min(wavelengthAxis), max(wavelengthAxis));
    
catch ME
    fprintf('❌ 加载失败: %s\n\n', ME.message);
    return;
end

%% ============================================================================
%  曲线信息显示和选择
% ============================================================================

fprintf('【步骤 3】选择要绘制的曲线\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

% 显示所有曲线的参数值
fprintf('可用的曲线列表:\n');
fprintf('%5s | %15s\n', '编号', '参数值');
fprintf('─────┼─────────────────\n');

for i = 1:min(numCurves, 20)  % 显示前20条，避免过长
    fprintf('%5d | %15.6f\n', i, paramAxis(i));
end

if numCurves > 20
    fprintf(' ...  | ...\n');
    fprintf('%5d | %15.6f\n', numCurves, paramAxis(numCurves));
end

fprintf('\n');

% 交互式选择
fprintf('【选择模式】\n');
fprintf('  1 - 选择单条曲线\n');
fprintf('  2 - 选择多条曲线（手动输入编号）\n');
fprintf('  3 - 按范围选择曲线\n');
fprintf('  4 - 绘制所有曲线\n');
fprintf('  5 - 自定义间隔选择\n\n');

choice = input('请选择模式 (1-5，默认 4): ');
if isempty(choice)
    choice = 4;
end

selectedIndices = [];

switch choice
    case 1
        % 单条曲线
        idx = input(sprintf('请输入曲线编号 (1-%d): ', numCurves));
        if isnumeric(idx) && idx >= 1 && idx <= numCurves
            selectedIndices = idx;
        else
            fprintf('❌ 无效的编号\n');
            return;
        end
        
    case 2
        % 手动输入
        fprintf('请输入要绘制的曲线编号，用逗号或空格分隔（例：1 3 5 或 1,3,5）:\n');
        inputStr = input('> ', 's');
        inputStr = strrep(inputStr, ',', ' ');
        selectedIndices = str2num(inputStr);
        
        % 验证
        if isempty(selectedIndices) || any(selectedIndices < 1) || any(selectedIndices > numCurves)
            fprintf('❌ 输入无效\n');
            return;
        end
        selectedIndices = unique(selectedIndices);
        
    case 3
        % 范围选择
        startIdx = input('请输入起始编号: ');
        endIdx = input('请输入结束编号: ');
        
        if isnumeric(startIdx) && isnumeric(endIdx) && ...
           startIdx >= 1 && endIdx <= numCurves && startIdx <= endIdx
            selectedIndices = startIdx:endIdx;
        else
            fprintf('❌ 无效的范围\n');
            return;
        end
        
    case 4
        % 所有曲线
        selectedIndices = 1:numCurves;
        fprintf('已选择所有 %d 条曲线\n', numCurves);
        
    case 5
        % 间隔选择
        step = input('请输入选择间隔 (例：每隔5条选择一条，输入 5): ');
        if isnumeric(step) && step >= 1 && step <= numCurves
            selectedIndices = 1:step:numCurves;
        else
            fprintf('❌ 无效的间隔\n');
            return;
        end
        
    otherwise
        fprintf('❌ 无效的选择\n');
        return;
end

numSelected = length(selectedIndices);
fprintf('✓ 已选择 %d 条曲线\n\n', numSelected);

%% ============================================================================
%  绘图模式选择
% ============================================================================

fprintf('【步骤 4】选择绘图模式\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

fprintf('可用的绘图模式:\n');
fprintf('  A - 单条曲线绘制（仅当选择1条）\n');
fprintf('  B - 堆叠图（多条曲线）\n');
fprintf('  C - 等高线图（仅当选择>1条）\n');
fprintf('  D - 对比图（多个子图）\n');
fprintf('  E - 光谱演变分析\n\n');

plotMode = input('请选择绘图模式 (A-E，默认 B): ', 's');
if isempty(plotMode)
    plotMode = 'B';
end
plotMode = upper(plotMode);

%% ============================================================================
%  执行绘图
% ============================================================================

fprintf('\n【步骤 5】生成图表\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

switch plotMode
    case 'A'
        % 单条曲线
        if numSelected ~= 1
            fprintf('❌ 单条曲线模式需要恰好选择 1 条曲线\n');
            return;
        end
        
        idx = selectedIndices(1);
        spectrum = spectra(idx, :);
        paramValue = paramAxis(idx);
        
        fprintf('绘制单条曲线 #%d (参数值: %.6f)\n', idx, paramValue);
        
        [fig, ax] = plot_single_spectrum(spectrum, wavelengthAxis, ...
            'Title', sprintf('PL Spectrum #%d (Parameter = %.6f)', idx, paramValue), ...
            'LineColor', 'b', ...
            'LineWidth', 2.5);
        
        fprintf('✓ 图表已生成\n');
        
    case 'B'
        % 堆叠图
        if numSelected < 2
            fprintf('⚠ 建议选择多条曲线来展示堆叠效果\n');
        end
        
        fprintf('生成堆叠图（%d 条曲线）\n', numSelected);
        
        % 构造子数据结构
        subData.data = spectra(selectedIndices, :);
        subData.axisscale{1,1} = paramAxis(selectedIndices);
        subData.axisscale{2,1} = wavelengthAxis;
        
        % 获取归一化方法参数
        fprintf('\n【归一化选项】\n');
        fprintf('  1 - 无（none）\n');
        fprintf('  2 - 按最大值（max）\n');
        fprintf('  3 - 按峰范围（peak_range）- 推荐\n');
        fprintf('  4 - 按面积（area）\n\n');
        
        normChoice = input('选择归一化方法 (1-4，默认 3): ');
        if isempty(normChoice)
            normChoice = 3;
        end
        
        normMethods = {'none', 'max', 'peak_range', 'area'};
        if normChoice >= 1 && normChoice <= 4
            normMethod = normMethods{normChoice};
        else
            normMethod = 'peak_range';
        end
        
        % 如果选择峰范围，让用户输入
        peakRange = [min(wavelengthAxis) + (max(wavelengthAxis) - min(wavelengthAxis)) * 0.4, ...
                     min(wavelengthAxis) + (max(wavelengthAxis) - min(wavelengthAxis)) * 0.6];
        
        if strcmp(normMethod, 'peak_range')
            fprintf('\n输入峰范围 (按 Enter 使用默认 [%.0f %.0f]):\n', peakRange(1), peakRange(2));
            peakMin = input('最小波长 (nm): ');
            if ~isempty(peakMin)
                peakRange(1) = peakMin;
            end
            peakMax = input('最大波长 (nm): ');
            if ~isempty(peakMax)
                peakRange(2) = peakMax;
            end
        end
        
        [fig, ax] = plot_spectra_stack(subData, ...
            'Indices', 1:numSelected, ...
            'NormalizeMethod', normMethod, ...
            'PeakRange', peakRange, ...
            'Spacing', 0.15, ...
            'Colormap', 'orange', ...
            'Title', sprintf('PL Spectra Stack (%d curves)', numSelected), ...
            'LineWidth', 2);
        
        fprintf('✓ 堆叠图已生成\n');
        
    case 'C'
        % 等高线图
        if numSelected < 2
            fprintf('❌ 等高线图需要至少 2 条曲线\n');
            return;
        end
        
        fprintf('生成等高线图（%d 条曲线）\n', numSelected);
        
        % 构造子数据结构
        subData.data = spectra(selectedIndices, :);
        subData.axisscale{1,1} = paramAxis(selectedIndices);
        subData.axisscale{2,1} = wavelengthAxis;
        
        [fig, ax] = plot_contour_map(subData, ...
            'ColorRange', [0.5 1.0], ...
            'ContourLevels', 50, ...
            'Colormap', 'hot', ...
            'ShowEnergyAxis', false, ...
            'Title', sprintf('PL Intensity Map (%d curves)', numSelected));
        
        fprintf('✓ 等高线图已生成\n');
        
    case 'D'
        % 对比图（多个子图）
        fprintf('生成对比图（%d 条曲线，%d 子图）\n', numSelected, min(numSelected, 6));
        
        numPlots = min(numSelected, 6);
        nRows = ceil(sqrt(numPlots));
        nCols = ceil(numPlots / nRows);
        
        fig = figure('Position', [100 100 1200 800], 'Color', 'w');
        
        for i = 1:numPlots
            ax = subplot(nRows, nCols, i);
            
            idx = selectedIndices(i);
            spectrum = spectra(idx, :);
            paramValue = paramAxis(idx);
            
            plot(ax, wavelengthAxis, spectrum, 'b-', 'LineWidth', 1.5);
            set(ax, 'FontName', 'Arial', 'FontSize', 10);
            xlabel(ax, 'Wavelength (nm)', 'FontSize', 11);
            ylabel(ax, 'Intensity (a.u.)', 'FontSize', 11);
            title(ax, sprintf('#%d: Param = %.4f', idx, paramValue), 'FontSize', 11);
            grid(ax, 'on');
            box(ax, 'on');
        end
        
        fprintf('✓ 对比图已生成\n');
        
    case 'E'
        % 光谱演变分析
        fprintf('生成光谱演变分析（%d 条曲线）\n', numSelected);
        
        % 构造子数据结构
        subData.data = spectra(selectedIndices, :);
        subData.axisscale{1,1} = paramAxis(selectedIndices);
        subData.axisscale{2,1} = wavelengthAxis;
        
        fig = figure('Position', [100 100 1400 800], 'Color', 'w');
        
        % 子图1: 堆叠图
        ax1 = subplot(1, 2, 1);
        [fig_stack, ax_stack] = plot_spectra_stack(subData, ...
            'Indices', 1:numSelected, ...
            'NormalizeMethod', 'peak_range', ...
            'Spacing', 0.15, ...
            'Colormap', 'orange', ...
            'Title', 'Normalized Stack', ...
            'LineWidth', 2);
        close(fig_stack);
        
        % 复制堆叠图到当前figure
        copyobj(allchild(ax_stack), ax1);
        set(ax1, 'FontName', 'Arial', 'FontSize', 11);
        
        % 子图2: 等高线图
        ax2 = subplot(1, 2, 2);
        [fig_contour, ax_contour] = plot_contour_map(subData, ...
            'ColorRange', [0.5 1.0], ...
            'ContourLevels', 50, ...
            'Colormap', 'hot', ...
            'ShowEnergyAxis', false, ...
            'Title', 'Intensity Map');
        close(fig_contour);
        
        % 复制等高线图到当前figure
        copyobj(allchild(ax_contour), ax2);
        set(ax2, 'FontName', 'Arial', 'FontSize', 11);
        
        fprintf('✓ 光谱演变分析图已生成\n');
        
    otherwise
        fprintf('❌ 未知的绘图模式\n');
        return;
end

%% ============================================================================
%  完成和总结
% ============================================================================

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    ✓ 绘图完成！                                 ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

fprintf('【绘图总结】\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('  ├─ 数据文件: %s\n', fileName);
fprintf('  ├─ 选中曲线数: %d / %d\n', numSelected, numCurves);
fprintf('  ├─ 绘图模式: %s\n', plotMode);
fprintf('  └─ 参数范围: %.6f → %.6f\n\n', ...
    min(paramAxis(selectedIndices)), max(paramAxis(selectedIndices)));

%% ============================================================================
%  可选功能：保存图表
% ============================================================================

fprintf('【可选操作】\n');
fprintf('  S - 保存当前图表\n');
fprintf('  D - 导出数据\n');
fprintf('  Q - 退出\n\n');

postAction = input('请选择操作 (S/D/Q，默认 Q): ', 's');
if isempty(postAction)
    postAction = 'Q';
end
postAction = upper(postAction);

switch postAction
    case 'S'
        % 保存图表
        [saveName, savePath] = uiputfile({'*.png';'*.pdf';'*.fig';'*.*'}, ...
            '保存图表为');
        
        if saveName ~= 0
            fullSavePath = fullfile(savePath, saveName);
            try
                saveas(gcf, fullSavePath);
                fprintf('✓ 图表已保存: %s\n', saveName);
            catch
                fprintf('❌ 保存失败\n');
            end
        end
        
    case 'D'
        % 导出数据
        [saveName, savePath] = uiputfile({'*.mat';'*.csv';'*.*'}, ...
            '导出数据为');
        
        if saveName ~= 0
            fullSavePath = fullfile(savePath, saveName);
            try
                exportData.selected_spectra = spectra(selectedIndices, :);
                exportData.selected_parameters = paramAxis(selectedIndices);
                exportData.wavelength = wavelengthAxis;
                
                if endsWith(saveName, '.csv')
                    % CSV格式
                    output = [paramAxis(selectedIndices)', spectra(selectedIndices, :)];
                    writematrix(output, fullSavePath);
                else
                    % MAT格式
                    save(fullSavePath, 'exportData');
                end
                fprintf('✓ 数据已导出: %s\n', saveName);
            catch ME
                fprintf('❌ 导出失败: %s\n', ME.message);
            end
        end
        
    otherwise
        fprintf('✓ 程序完成\n');
end

fprintf('\n');
