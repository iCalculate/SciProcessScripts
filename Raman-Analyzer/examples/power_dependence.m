%% 拉曼功率依赖性分析示例
% 演示 Raman 随激光功率变化的分析流程

clear; clc; close all;
addpath(genpath('.'));
addpath(genpath('../core'));

fprintf('=== 拉曼功率依赖性分析示例 ===\n\n');

%% ===== 直接指定文件路径 =====
% 如果下面的路径为空字符串，则会弹出对话框选择文件
% 否则直接使用指定的路径
ramanFilePath = 'D:\NUS Dropbox\Xinchuan Du\PendingPrj\Publication\MoS2 in-plane electrostriction\Data\Raman\Series Pow--4515--Spec.Data_RamanvsLaser.mat';
powerFilePath = 'D:\NUS Dropbox\Xinchuan Du\PendingPrj\Publication\MoS2 in-plane electrostriction\Data\Raman\Series Pow--4515--Laser Power_RamanvsLaser.mat';

% ===== 激发光波长设置 =====
% 用于计算相对拉曼位移 (相对于激发光的位移)
% 如果数据中的轴已经是相对位移，设置为 0（禁用转换）
excitationWavelength_nm = 488;  % 激发光波长 (nm)，设为0表示禁用转换

% ===== 拉曼位移绘图范围 =====
% 注意: 必须在实际数据范围内！实际数据范围在加载数据后显示
% MoS₂数据实际范围: 486-528 cm⁻¹ (或其他，取决于轴的定义)
ramanPlotRange = [340 450];  % 聚焦范围 (将自动验证有效性)

% ===== 如果不想使用固定路径，请改为空字符串 =====
% ramanFilePath = '';
% powerFilePath = '';

%% 加载数据
fprintf('Step 1: 加载数据\n');
fprintf('-------------------------------------------\n');

% 如果没有指定路径，则弹出对话框选择
if isempty(ramanFilePath)
    [ramanFilename, ramanPathname] = uigetfile('*.mat', '选择拉曼数据文件 (.mat)');
    if isequal(ramanFilename, 0)
        error('用户取消了拉曼数据文件选择');
    end
    ramanFilePath = fullfile(ramanPathname, ramanFilename);
end

if isempty(powerFilePath)
    [powerFilename, powerPathname] = uigetfile('*.mat', '选择激光功率数据文件 (.mat)');
    if isequal(powerFilename, 0)
        error('用户取消了激光功率数据文件选择');
    end
    powerFilePath = fullfile(powerPathname, powerFilename);
end

fprintf('拉曼数据文件: %s\n', ramanFilePath);
fprintf('功率数据文件: %s\n\n', powerFilePath);

% 加载数据
try
    RamanData = load_raman_power_data(ramanFilePath, powerFilePath);
catch ME
    error('数据加载失败: %s', ME.message);
end

% 获取数据维度和轴信息
powers = RamanData.axisscale{1,1}(:);
ramanShift = RamanData.axisscale{2,1}(:);
spectra = RamanData.data;
numPowers = length(powers);
numRamanPoints = length(ramanShift);

fprintf('数据大小: %d × %d\n', size(RamanData.data));
fprintf('功率范围: %.4f - %.4f mW\n', min(powers), max(powers));
actualRamanMin = min(ramanShift);
actualRamanMax = max(ramanShift);
fprintf('轴范围（原始）: %.2f - %.2f nm\n', actualRamanMin, actualRamanMax);

% ===== 从波长(nm)转换为相对拉曼位移(cm⁻¹) =====
fprintf('\n转换波长为拉曼位移...\n');

if excitationWavelength_nm > 0 && actualRamanMax < 1000
    % 原始轴是波长(nm)，需要转换
    fprintf('  激发光: %.1f nm\n', excitationWavelength_nm);
    
    % 转换公式: Raman Shift [cm^-1] = (1/lambda_0 - 1/lambda) * 10^7
    % 其中 lambda_0 和 lambda 都以 nm 为单位
    ramanShift = (1/excitationWavelength_nm - 1./ramanShift) * 1e7;
    
    % 转换后轴可能是反向排序的，需要对轴和数据同时排序
    [ramanShift, sortIdx] = sort(ramanShift);
    spectra = spectra(:, sortIdx);  % 对应调整数据列
    
    % 更新 RamanData 中的轴数据和光谱数据，以便绘图函数能正确处理
    RamanData.axisscale{2,1} = ramanShift;
    RamanData.data = spectra;
    
    actualRamanMin = min(ramanShift);
    actualRamanMax = max(ramanShift);
    fprintf('  ✓ 转换完成\n');
    fprintf('  转换后范围: %.1f - %.1f cm⁻¹\n\n', actualRamanMin, actualRamanMax);
else
    fprintf('  跳过转换（禁用或轴不是波长）\n\n');
end

% 检查指定的拉曼范围是否有效
% 注意：如果进行了波数转换，范围可能是反向的
ramanPlotMin = min(ramanPlotRange);
ramanPlotMax = max(ramanPlotRange);
dataMin = min(ramanShift);
dataMax = max(ramanShift);

if ramanPlotMin < dataMin || ramanPlotMax > dataMax
    fprintf('\n⚠ 警告: 指定的绘图范围 [%.0f - %.0f cm⁻¹] 超出实际数据范围！\n', ramanPlotMin, ramanPlotMax);
    fprintf('实际数据范围: [%.0f - %.0f cm⁻¹]\n', dataMin, dataMax);
    fprintf('已自动调整为实际数据范围\n\n');
    ramanPlotRange = [dataMin, dataMax];
else
    fprintf('✓ 指定范围有效: [%.0f - %.0f cm⁻¹]\n\n', ramanPlotMin, ramanPlotMax);
end

%% 分析1-2: 绘制等高线图和堆叠谱线
fprintf('Step 2-3: 光谱演变分析\n');
fprintf('-------------------------------------------\n');

figure('Position', [100 100 1800 600]);

% 等高线图
subplot(1,2,1);
[fig1, ax1] = plot_raman_contour(RamanData, ...
    'ColorRange', [0.4 1.0], ...
    'ContourLevels', 60, ...
    'Colormap', 'hot', ...
    'Title', 'Raman Intensity Map vs Laser Power');

% 堆叠谱线图
subplot(1,2,2);
indices = 1:4:numPowers;  % 每隔4条绘制
[fig2, ax2] = plot_raman_stack(RamanData, ...
    'Indices', indices, ...
    'RamanRange', ramanPlotRange, ...  % 使用指定的拉曼范围
    'NormalizeMethod', 'max', ...
    'Spacing', 0.02, ...
    'Colormap', 'power', ...
    'Title', sprintf('Raman Spectra Evolution (%.0f-%.0f cm^{-1})', ramanPlotRange(1), ramanPlotRange(2)));

%% 分析3-4: 功率依赖性分析和详细堆叠图
fprintf('\nStep 4-5: 功率依赖性和详细分析\n');
fprintf('-------------------------------------------\n');

% 自动检测最强峰（在指定范围内）
fprintf('检测光谱主峰...\n');
fprintf('  搜索范围: %.0f - %.0f cm⁻¹\n', ramanPlotRange(1), ramanPlotRange(2));

% 找出指定范围内的索引
rangeIndices = ramanShift >= ramanPlotRange(1) & ramanShift <= ramanPlotRange(2);
if sum(rangeIndices) < 2
    warning('指定范围内数据点不足，使用全范围检测');
    rangeIndices = ones(size(ramanShift), 'logical');
end

meanSpectrum = mean(spectra(:, rangeIndices), 1);
[~, localPeakIdx] = max(meanSpectrum);
globalPeakIdx = find(rangeIndices);
peakIdx = globalPeakIdx(localPeakIdx);
mainPeakShift = ramanShift(peakIdx);

fprintf('  主峰位移: %.1f cm⁻¹\n', mainPeakShift);

% 定义峰范围 - 自动适应不同样品的拉曼谱
peakWidth = 30;  % 峰宽度设为 ±30 cm⁻¹
peakIndices = ramanShift >= (mainPeakShift - peakWidth) & ramanShift <= (mainPeakShift + peakWidth);

% 如果定义的范围内没有数据点，则扩大范围
if sum(peakIndices) < 10
    peakWidth = max(ramanShift(rangeIndices)) - min(ramanShift(rangeIndices));
    peakIndices = rangeIndices;
end

% 提取主峰强度
peakIntensities = max(spectra(:, peakIndices), [], 2);

fprintf('  拟合范围: %.1f - %.1f cm⁻¹\n', mainPeakShift - peakWidth, mainPeakShift + peakWidth);

% 拟合 I ∝ P^n
fitOpts = fitoptions('Method', 'NonlinearLeastSquares', 'Display', 'off');
fitOpts.StartPoint = [1, 1.2];
fitOpts.Lower = [0.01, 0.1];
fitOpts.Upper = [10000, 3];

fitType = fittype('a*x^n', 'independent', 'x', 'dependent', 'y');

try
    % 主峰拟合
    foPeak = fit(powers, peakIntensities, fitType, fitOpts);
    nValuePeak = foPeak.n;
    aValuePeak = foPeak.a;
    
    fprintf('✓ 功率依赖性拟合结果：\n');
    fprintf('  主峰: I = %.2f × P^%.3f\n', aValuePeak, nValuePeak);
    
    % 绘制功率依赖性和详细堆叠图
    figure('Position', [100 700 1800 600]);
    
    % 左上图：功率依赖性
    subplot(1,3,1);
    plot(powers, peakIntensities, 'bo', 'MarkerSize', 8, 'LineWidth', 2, ...
        'DisplayName', 'Peak Intensity'); hold on;
    
    powersFit = linspace(min(powers), max(powers), 100);
    peakFit = aValuePeak * powersFit.^nValuePeak;
    plot(powersFit, peakFit, 'b-', 'LineWidth', 2.5, ...
        'DisplayName', sprintf('Fit: I = %.2f·P^{%.3f}', aValuePeak, nValuePeak));
    
    set(gca, 'FontName', 'Arial', 'FontSize', 12, 'XScale', 'log', 'YScale', 'log');
    xlabel('Laser Power (mW)', 'FontSize', 13);
    ylabel('Peak Intensity (a.u.)', 'FontSize', 13);
    title('Power-Dependent Raman Intensity', 'FontSize', 13);
    legend('FontSize', 11, 'Location', 'best');
    grid on;
    
    % 中间图：规范化谱线随功率变化
    subplot(1,3,2);
    numPlots = min(8, ceil(numPowers / 4));  % 选择8条或更少的谱线显示
    indices = round(linspace(1, numPowers, numPlots));
    
    % 创建颜色映射
    colors = parula(numPlots);
    
    % 找出指定范围内的拉曼位移索引
    ramanRangeIndices = ramanShift >= ramanPlotRange(1) & ramanShift <= ramanPlotRange(2);
    ramanShiftRange = ramanShift(ramanRangeIndices);
    
    for i = 1:numPlots
        idx = indices(i);
        spectrumRange = spectra(idx, ramanRangeIndices);
        normSpectrum = spectrumRange / max(spectrumRange);
        plot(ramanShiftRange, normSpectrum + (i-1)*0.5, 'Color', colors(i,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('P = %.2f mW', powers(idx))); hold on;
    end
    
    set(gca, 'FontName', 'Arial', 'FontSize', 12);
    xlabel('Raman Shift (cm^{-1})', 'FontSize', 13);
    ylabel('Intensity (offset for clarity)', 'FontSize', 13);
    title(sprintf('Normalized Spectra (%.0f-%.0f cm^{-1})', ramanPlotRange(1), ramanPlotRange(2)), 'FontSize', 13);
    legend('FontSize', 9, 'Location', 'best');
    grid on;
    
    % 右边图：详细堆叠图
    subplot(1,3,3);
    [fig4, ax4] = plot_raman_stack(RamanData, ...
        'Indices', 1:3:numPowers, ...
        'RamanRange', ramanPlotRange, ...  % 使用指定的聚焦范围
        'NormalizeMethod', 'max', ...
        'Spacing', 0.02, ...
        'Colormap', 'power', ...
        'Title', sprintf('Raman Details (%.0f-%.0f cm^{-1})', ramanPlotRange(1), ramanPlotRange(2)));
    
catch ME
    fprintf('✗ 功率拟合失败: %s\n', ME.message);
end

fprintf('\n✓ 拉曼功率依赖性分析完成！\n');
