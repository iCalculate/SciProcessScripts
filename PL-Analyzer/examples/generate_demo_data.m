%% 生成演示数据 / Generate Demo Data
%
% 这个脚本生成一个模拟的 PL 数据系列，用于测试 multi_curve_plotter
%
% 用法:
%   >> generate_demo_data
%
% 输出:
%   demo_pl_power_sweep.mat - 功率扫描演示数据
%   demo_pl_angle_scan.mat  - 角度扫描演示数据

clear; clc;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║          生成 PL 演示数据 / Generating Demo PL Data             ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% ============================================================================
%  演示1: 功率扫描数据
% ============================================================================

fprintf('【演示1】功率扫描数据 / Power Sweep Data\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

fprintf('生成参数...\n');

% 功率参数
powers = (0.01:0.02:1.0)';  % mW
numPowers = length(powers);

% 波长参数
wavelength = (600:0.5:750)';  % nm
numWavelengths = length(wavelength);

fprintf('  ├─ 功率点数: %d (%.3f - %.3f mW)\n', numPowers, min(powers), max(powers));
fprintf('  └─ 波长点数: %d (%.0f - %.0f nm)\n', numWavelengths, min(wavelength), max(wavelength));

fprintf('生成光谱数据...\n');

% 生成光谱
spectra_power = zeros(numPowers, numWavelengths);

for i = 1:numPowers
    % 理想情况: 单个高斯峰，强度随功率增加
    peakCenter = 680;  % nm
    peakWidth = 30;    % nm
    
    % 功率依赖: I ∝ P^1.5 (激子复合)
    amplitude = 500 * powers(i)^1.5;
    
    % 高斯峰
    spectrum = amplitude * exp(-((wavelength - peakCenter).^2) / (2 * (peakWidth/4)^2));
    
    % 添加噪声
    noiseLevel = amplitude * 0.03;  % 3% 相对噪声
    spectrum = spectrum + randn(size(spectrum)) * noiseLevel;
    
    % 保证非负
    spectrum = max(spectrum, 0);
    
    spectra_power(i, :) = spectrum;
end

% 构造数据结构
PL_Data_Power.data = spectra_power;
PL_Data_Power.axisscale{1,1} = powers;
PL_Data_Power.axisscale{2,1} = wavelength;

fprintf('✓ 功率扫描数据已生成\n');
fprintf('  数据大小: %d × %d\n\n', size(spectra_power));

% 保存
savePath = fullfile(pwd, 'demo_pl_power_sweep.mat');
save(savePath, 'PL_Data_Power');
fprintf('✓ 已保存: %s\n\n', savePath);

%% ============================================================================
%  演示2: 角度扫描数据（偏振）
% ============================================================================

fprintf('【演示2】角度扫描数据 / Angle Scan Data (Polarization)\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

fprintf('生成参数...\n');

% 角度参数
angles = (0:3:180)';  % degrees
numAngles = length(angles);

% 波长相同
numWavelengths_angle = length(wavelength);

fprintf('  ├─ 角度点数: %d (%.1f° - %.1f°)\n', numAngles, min(angles), max(angles));
fprintf('  └─ 波长点数: %d (%.0f - %.0f nm)\n', numWavelengths_angle, min(wavelength), max(wavelength));

fprintf('生成光谱数据...\n');

% 生成光谱
spectra_angle = zeros(numAngles, numWavelengths_angle);

for i = 1:numAngles
    % 基础峰（在 0° 处最强）
    peakCenter = 680;
    peakWidth = 30;
    
    % 偏振依赖: I(θ) ∝ cos^2(2θ)
    % 这反映了线性偏振发光的各向异性
    angle_rad = angles(i) * pi / 180;
    polarizationFactor = abs(cos(2 * angle_rad))^2;
    
    amplitude = 400 * (1 + polarizationFactor);  % 基础强度 + 偏振调制
    
    % 高斯峰
    spectrum = amplitude * exp(-((wavelength - peakCenter).^2) / (2 * (peakWidth/4)^2));
    
    % 添加偏振依赖的宽度变化（可选效应）
    peakWidth_angle = peakWidth * (1 + 0.2 * polarizationFactor);
    spectrum = amplitude * exp(-((wavelength - peakCenter).^2) / (2 * (peakWidth_angle/4)^2));
    
    % 添加噪声
    noiseLevel = amplitude * 0.02;
    spectrum = spectrum + randn(size(spectrum)) * noiseLevel;
    
    spectrum = max(spectrum, 0);
    spectra_angle(i, :) = spectrum;
end

% 构造数据结构
PL_Data_Angle.data = spectra_angle;
PL_Data_Angle.axisscale{1,1} = angles;
PL_Data_Angle.axisscale{2,1} = wavelength;

fprintf('✓ 角度扫描数据已生成\n');
fprintf('  数据大小: %d × %d\n\n', size(spectra_angle));

% 保存
savePath2 = fullfile(pwd, 'demo_pl_angle_scan.mat');
save(savePath2, 'PL_Data_Angle');
fprintf('✓ 已保存: %s\n\n', savePath2);

%% ============================================================================
%  演示3: 温度扫描数据
% ============================================================================

fprintf('【演示3】温度扫描数据 / Temperature Scan Data\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

fprintf('生成参数...\n');

% 温度参数
temperatures = (10:10:300)';  % K
numTemps = length(temperatures);

fprintf('  ├─ 温度点数: %d (%d - %d K)\n', numTemps, min(temperatures), max(temperatures));
fprintf('  └─ 波长点数: %d (%.0f - %.0f nm)\n', numWavelengths_angle, min(wavelength), max(wavelength));

fprintf('生成光谱数据...\n');

% 生成光谱
spectra_temp = zeros(numTemps, numWavelengths_angle);

for i = 1:numTemps
    T = temperatures(i);
    
    % 温度依赖效应
    % 1. 峰位置随温度红移 (Varshni eq)
    Eg0 = 2.4;      % eV，室温禁带宽度 (示例)
    alpha = 2.0e-4; % eV/K (Varshni 参数)
    beta = 100;     % K (Varshni 参数)
    
    Eg = Eg0 - alpha * T^2 / (T + beta);  % eV
    peakCenter_T = 1239 / Eg;  % 从 eV 转换到 nm
    peakCenter_T = min(max(peakCenter_T, 620), 750);  % 限制在波长范围内
    
    peakWidth = 25;
    
    % 2. 强度随温度减小 (激子复合效率)
    amplitude = 600 * exp(-30 / T);
    
    % 高斯峰
    spectrum = amplitude * exp(-((wavelength - peakCenter_T).^2) / (2 * (peakWidth/4)^2));
    
    % 添加噪声
    noiseLevel = amplitude * 0.025;
    spectrum = spectrum + randn(size(spectrum)) * noiseLevel;
    
    spectrum = max(spectrum, 0);
    spectra_temp(i, :) = spectrum;
end

% 构造数据结构
PL_Data_Temp.data = spectra_temp;
PL_Data_Temp.axisscale{1,1} = temperatures;
PL_Data_Temp.axisscale{2,1} = wavelength;

fprintf('✓ 温度扫描数据已生成\n');
fprintf('  数据大小: %d × %d\n\n', size(spectra_temp));

% 保存
savePath3 = fullfile(pwd, 'demo_pl_temperature_scan.mat');
save(savePath3, 'PL_Data_Temp');
fprintf('✓ 已保存: %s\n\n', savePath3);

%% ============================================================================
%  演示4: 多峰数据（二维材料典型）
% ============================================================================

fprintf('【演示4】多峰数据 / Multi-Peak Data (2D Material)\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

fprintf('生成参数...\n');

% 激光功率参数
powers_2d = (0.02:0.04:1.0)';
numPowers_2d = length(powers_2d);

fprintf('  ├─ 功率点数: %d\n', numPowers_2d);
fprintf('  └─ 波长点数: %d (包含多个峰)\n', numWavelengths_angle);

fprintf('生成光谱数据...\n');

% 生成多峰光谱 (典型的二维材料: 激子峰 + 三离子子态等)
spectra_2d = zeros(numPowers_2d, numWavelengths_angle);

for i = 1:numPowers_2d
    P = powers_2d(i);
    
    % 构造三个峰
    % 峰1: 激子 (A exciton) @ 620 nm
    % 峰2: 激子 (B exciton) @ 660 nm  
    % 峰3: 三离子子态 (Trion) @ 610 nm
    
    peak1 = 800 * P^1.3 * exp(-((wavelength - 620).^2) / (2 * 6^2));
    peak2 = 500 * P^1.4 * exp(-((wavelength - 660).^2) / (2 * 8^2));
    peak3 = 300 * P^1.1 * exp(-((wavelength - 610).^2) / (2 * 5^2));
    
    spectrum = peak1 + peak2 + peak3;
    
    % 添加噪声
    noiseLevel = max(spectrum) * 0.04;
    spectrum = spectrum + randn(size(spectrum)) * noiseLevel;
    spectrum = max(spectrum, 0);
    
    spectra_2d(i, :) = spectrum;
end

% 构造数据结构
PL_Data_2D.data = spectra_2d;
PL_Data_2D.axisscale{1,1} = powers_2d;
PL_Data_2D.axisscale{2,1} = wavelength;

fprintf('✓ 多峰数据已生成\n');
fprintf('  数据大小: %d × %d\n\n', size(spectra_2d));

% 保存
savePath4 = fullfile(pwd, 'demo_pl_multipeak.mat');
save(savePath4, 'PL_Data_2D');
fprintf('✓ 已保存: %s\n\n', savePath4);

%% ============================================================================
%  完成信息
% ============================================================================

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    ✓ 全部演示数据已生成                         ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

fprintf('生成的文件:\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('1. demo_pl_power_sweep.mat      (50条曲线) - 功率依赖\n');
fprintf('2. demo_pl_angle_scan.mat       (61条曲线) - 偏振各向异性\n');
fprintf('3. demo_pl_temperature_scan.mat (30条曲线) - 温度依赖\n');
fprintf('4. demo_pl_multipeak.mat        (25条曲线) - 多个峰\n\n');

fprintf('下一步:\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('1. 运行 multi_curve_plotter\n');
fprintf('2. 选择上述任一 demo_*.mat 文件\n');
fprintf('3. 选择曲线和绘图模式进行分析\n\n');

fprintf('推荐测试:\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('【快速开始】\n');
fprintf('  文件: demo_pl_power_sweep.mat\n');
fprintf('  选择模式: 4 (全部)\n');
fprintf('  绘图模式: E (光谱演变)\n');
fprintf('  归一化: peak_range [650, 700]\n\n');

fprintf('【对比分析】\n');
fprintf('  文件: demo_pl_multipeak.mat\n');
fprintf('  选择模式: 5 (间隔 2)\n');
fprintf('  绘图模式: D (对比图)\n\n');

fprintf('✓ 准备就绪！\n\n');
