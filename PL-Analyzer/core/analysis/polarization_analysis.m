function [intensity_vs_angle, angle_values] = polarization_analysis(PLData, angleAxis, wavelengthRange)
%POLARIZATION_ANALYSIS 分析PL随偏振角度的依赖关系
%   [intensity_vs_angle, angle_values] = polarization_analysis(PLData, angleAxis, wavelengthRange)
%
%   输入:
%       PLData - 光谱数据矩阵 (行: 角度, 列: 波长)
%       angleAxis - 偏振角度数组 (度)
%       wavelengthRange - 波长范围 [min max]
%
%   输出:
%       intensity_vs_angle - 各角度的积分强度
%       angle_values - 对应的角度值
%
%   示例:
%       [intensity, angles] = polarization_analysis(data, 0:5:180, [650 700]);

% 确保输入为列向量
angleAxis = angleAxis(:);

% 找到波长范围内的列索引
% 假设波长从小到大
wavelengthMask = (1:size(PLData, 2))' >= wavelengthRange(1) & ...
                 (1:size(PLData, 2))' <= wavelengthRange(2);

% 对于每个角度，计算该范围内的积分强度
intensity_vs_angle = zeros(size(angleAxis));

for i = 1:length(angleAxis)
    spectrum = PLData(i, :);
    % 积分范围内的数据
    intensity_vs_angle(i) = sum(spectrum(wavelengthMask));
end

% 归一化
intensity_vs_angle = intensity_vs_angle / max(intensity_vs_angle);

% 尝试拟合为 cos^n 或 sin^n 形式
% intensity = a * cos^2(n*(angle - offset))

fitOpts = fitoptions('Method', 'NonlinearLeastSquares', ...
    'MaxIter', 10000, 'Display', 'off');

try
    fitType = fittype('a*cos(n*(x - offset))^2 + bg', 'independent', 'x', 'dependent', 'y');
    fitOpts.StartPoint = [1, 2, 0, 0];
    fitOpts.Lower = [0.1, 0.1, -180, 0];
    fitOpts.Upper = [2, 10, 180, 0.5];
    
    fo = fit(angleAxis, intensity_vs_angle, fitType, fitOpts);
    angle_values = angleAxis;
    
    fprintf('\n=== Polarization Analysis Results ===\n');
    fprintf('Fit: I = a*cos^2(n*(θ - offset)) + bg\n');
    fprintf('a = %.3f\n', fo.a);
    fprintf('n = %.3f\n', fo.n);
    fprintf('offset = %.1f°\n', fo.offset);
    fprintf('background = %.3f\n', fo.bg);
    
catch
    warning('偏振拟合失败，仅返回原始数据');
    angle_values = angleAxis;
end

end
