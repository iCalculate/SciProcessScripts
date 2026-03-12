%% 拉曼数据诊断脚本
% 用于检查数据的轴定义和范围
% Raman Data Diagnostic Script

clear; clc;
addpath(genpath('.'));
addpath(genpath('../core'));

fprintf('=== 拉曼数据诊断工具 ===\n\n');

% 设置文件路径
ramanFilePath = 'D:\NUS Dropbox\Xinchuan Du\PendingPrj\Publication\MoS2 in-plane electrostriction\Data\Raman\Series Pow--4515--Spec.Data_RamanvsLaser.mat';
powerFilePath = 'D:\NUS Dropbox\Xinchuan Du\PendingPrj\Publication\MoS2 in-plane electrostriction\Data\Raman\Series Pow--4515--Laser Power_RamanvsLaser.mat';

% 加载数据
fprintf('Step 1: 加载数据文件\n');
fprintf('────────────────────────────────────────\n');

try
    RamanData = load_raman_power_data(ramanFilePath, powerFilePath);
    fprintf('✓ 数据加载成功\n\n');
catch ME
    fprintf('✗ 数据加载失败: %s\n', ME.message);
    return;
end

% 提取数据
powers = RamanData.axisscale{1,1}(:);
ramanAxis = RamanData.axisscale{2,1}(:);
spectra = RamanData.data;

% 基本信息
fprintf('Step 2: 数据基本信息\n');
fprintf('────────────────────────────────────────\n');
fprintf('数据矩阵大小: %d × %d\n', size(spectra));
fprintf('  第一维 (行): %d 个功率点\n', size(spectra, 1));
fprintf('  第二维 (列): %d 个拉曼点\n\n', size(spectra, 2));

fprintf('功率数据:\n');
fprintf('  范围: %.4f - %.4f mW\n', min(powers), max(powers));
fprintf('  步长: %.4f mW\n', mean(diff(powers)));
fprintf('  点数: %d\n\n', length(powers));

fprintf('拉曼轴数据:\n');
fprintf('  最小值: %.4f\n', min(ramanAxis));
fprintf('  最大值: %.4f\n', max(ramanAxis));
fprintf('  范围: %.4f\n', max(ramanAxis) - min(ramanAxis));
fprintf('  步长: %.6f (平均)\n', mean(diff(ramanAxis)));
fprintf('  点数: %d\n\n', length(ramanAxis));

% 轴类型诊断
fprintf('Step 3: 轴类型诊断\n');
fprintf('────────────────────────────────────────\n');

axisMin = min(ramanAxis);
axisMax = max(ramanAxis);

if axisMax < 10000  % 可能是波长(nm) 或 相对拉曼位移
    if axisMax < 1000 && axisMin > 200  % 典型的波长范围
        fprintf('✓ 检测到轴数据是波长 (nm)\n');
        fprintf('  范围: %.2f - %.2f nm\n', axisMin, axisMax);
        fprintf('  → 需要转换为相对拉曼位移\n');
        fprintf('  → 使用公式: Shift = (1/λ₀ - 1/λ) × 10^7\n\n');
        
        fprintf('  对于488nm激发光，转换后的范围将是:\n');
        excWl = 488;
        raman_min = (1/excWl - 1/axisMax) * 1e7;
        raman_max = (1/excWl - 1/axisMin) * 1e7;
        fprintf('    %.1f - %.1f cm⁻¹\n\n', raman_min, raman_max);
        
    elseif axisMax < 1000
        fprintf('✓ 轴的数值在合理范围内 (min=%.1f, max=%.1f)\n', axisMin, axisMax);
        fprintf('  → 很可能是相对拉曼位移 (Raman Shift)\n');
        fprintf('  → 不需要进行坐标转换\n\n');
    end
elseif axisMax > 10000  % 绝对波数
    fprintf('⚠ 检测到轴的数值很大 (min=%.0f, max=%.0f)\n', axisMin, axisMax);
    fprintf('  → 可能是绝对波数 (Absolute Wavenumber)\n');
    fprintf('  → 不是相对拉曼位移 (Raman Shift)\n\n');
end

% 光谱强度分析
fprintf('Step 4: 光谱强度分析\n');
fprintf('────────────────────────────────────────\n');

meanSpectrum = mean(spectra, 1);
[maxIntensity, maxIdx] = max(meanSpectrum);
mainPeakWavenumber = ramanAxis(maxIdx);

fprintf('平均光谱的最强峰:\n');
fprintf('  位置: %.2f cm⁻¹\n', mainPeakWavenumber);
fprintf('  强度: %.2e\n', maxIntensity);
fprintf('  索引: %d (共 %d 个点)\n\n', maxIdx, length(ramanAxis));

% 显示光谱特征
fprintf('光谱特征:\n');

% 找出几个局部最大值
windowSize = max(5, round(length(ramanAxis) / 50));
localMax = [];
for i = windowSize+1 : length(ramanAxis) - windowSize
    if meanSpectrum(i) > max(meanSpectrum(i-windowSize:i-1)) && ...
       meanSpectrum(i) > max(meanSpectrum(i+1:i+windowSize))
        localMax = [localMax, i];
    end
end

% 排序并显示前5个
[~, idx] = sort(meanSpectrum(localMax), 'descend');
topPeaks = localMax(idx(1:min(5, length(idx))));
topPeaks = sort(topPeaks);

fprintf('  检测到的主要峰（从强到弱）:\n');
for i = 1:length(topPeaks)
    idx = topPeaks(i);
    fprintf('    %d. 位移 %.2f cm⁻¹, 强度 %.2e\n', i, ramanAxis(idx), meanSpectrum(idx));
end

fprintf('\n');

% 建议
fprintf('Step 5: 建议\n');
fprintf('────────────────────────────────────────\n');

if axisMax < 1000 && axisMin > 200  % 波长(nm)
    fprintf('1. 数据轴是波长(nm)，需要转换为相对拉曼位移\n');
    fprintf('   设置 excitationWavelength_nm = 488\n\n');
    
    excWl = 488;
    raman_min = (1/excWl - 1/axisMax) * 1e7;
    raman_max = (1/excWl - 1/axisMin) * 1e7;
    
    fprintf('2. 转换后的拉曼位移范围:\n');
    fprintf('   %.0f - %.0f cm⁻¹\n\n', raman_min, raman_max);
    
    fprintf('3. 因此，设置:\n');
    fprintf('   ramanPlotRange = [%d %d]\n\n', round(raman_min), round(raman_max));
    
elseif axisMax < 1000
    fprintf('1. 数据轴已经是相对拉曼位移\n');
    fprintf('   设置 excitationWavelength_nm = 0 (禁用转换)\n\n');
    
    fprintf('2. 实际拉曼范围:\n');
    fprintf('   %.0f - %.0f cm⁻¹\n\n', axisMin, axisMax);
    
    fprintf('3. 您想要的范围 [340 - 450] 是否在此范围内？\n');
    if (340 >= axisMin && 340 <= axisMax) || (450 >= axisMin && 450 <= axisMax)
        fprintf('   ✓ 范围有交集\n');
    else
        fprintf('   ✗ 范围无交集 - 您的范围不在数据中！\n');
    end
    fprintf('\n');
    
else  % 绝对波数
    excWl = 488;
    fprintf('1. 数据轴是绝对波数\n');
    fprintf('   设置 excitationWavelength_nm = %d\n\n', excWl);
    
    excWn = 1 / (excWl * 1e-7);
    fprintf('2. 转换为相对拉曼位移后的范围:\n');
    fprintf('   %.0f - %.0f cm⁻¹\n\n', excWn - axisMax, excWn - axisMin);
    
    fprintf('3. 因此，设置:\n');
    fprintf('   ramanPlotRange = [%d %d]\n\n', round(excWn - axisMax), round(excWn - axisMin));
end

% 总结
fprintf('Step 6: 总结\n');
fprintf('────────────────────────────────────────\n');
fprintf('请根据上述诊断信息修改 power_dependence.m 中的：\n');
fprintf('  • excitationWavelength_nm\n');
fprintf('  • ramanPlotRange\n\n');

fprintf('更多帮助，请查看 EXCITATION_GUIDE.m\n');
fprintf('════════════════════════════════════════\n\n');
