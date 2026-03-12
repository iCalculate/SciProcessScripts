function [peaks, positions, widths, amplitudes] = peak_analysis(spectrum, wavelength, varargin)
%PEAK_ANALYSIS 光谱峰值分析
%   [peaks, positions, widths, amplitudes] = peak_analysis(spectrum, wavelength, Name, Value)
%
%   输入:
%       spectrum - 光谱强度数据
%       wavelength - 波长数据
%
%   可选参数:
%       'MinPeakHeight' - 最小峰高，默认为信号标准差的3倍
%       'MinPeakDistance' - 相邻峰的最小距离 (nm)，默认 10
%       'NumPeaks' - 要查找的峰数，默认 []（查找所有峰）
%       'Plot' - 是否绘制结果，默认 false
%
%   输出:
%       peaks - 峰值强度
%       positions - 峰位置 (nm)
%       widths - 峰宽 (nm)
%       amplitudes - 峰幅度
%
%   示例:
%       [pk, pos, wid, amp] = peak_analysis(spectrum, wavelength, 'Plot', true);

p = inputParser;
addRequired(p, 'spectrum', @isvector);
addRequired(p, 'wavelength', @isvector);
addParameter(p, 'MinPeakHeight', [], @(x) isempty(x) || isnumeric(x));
addParameter(p, 'MinPeakDistance', 10, @(x) isnumeric(x) && x > 0);
addParameter(p, 'NumPeaks', [], @(x) isempty(x) || (isnumeric(x) && x > 0));
addParameter(p, 'Plot', false, @islogical);
parse(p, spectrum, wavelength, varargin{:});

spectrum = spectrum(:);
wavelength = wavelength(:);

% 自动计算最小峰高
if isempty(p.Results.MinPeakHeight)
    minPeakHeight = 3 * std(spectrum);
else
    minPeakHeight = p.Results.MinPeakHeight;
end

% 距离转换为样本点
wavelengthDiff = abs(diff(wavelength(1:2)));
minDistance = round(p.Results.MinPeakDistance / wavelengthDiff);

% 查找峰值
[peaks, locs] = findpeaks(spectrum, 'MinPeakHeight', minPeakHeight, ...
    'MinPeakDistance', minDistance);

% 限制峰数
if ~isempty(p.Results.NumPeaks) && length(peaks) > p.Results.NumPeaks
    [peaks, sortIdx] = sort(peaks, 'descend');
    peaks = peaks(1:p.Results.NumPeaks);
    locs = locs(sortIdx(1:p.Results.NumPeaks));
    [locs, sortIdx2] = sort(locs);
    peaks = peaks(sortIdx2);
end

positions = wavelength(locs);

% 计算峰宽和幅度
widths = zeros(size(peaks));
amplitudes = peaks;

for i = 1:length(peaks)
    % 使用半最大全宽 (FWHM)
    halfMax = peaks(i) / 2;
    leftIdx = find(spectrum(1:locs(i)) >= halfMax, 1, 'last');
    rightIdx = find(spectrum(locs(i):end) >= halfMax, 1) + locs(i) - 1;
    
    if ~isempty(leftIdx) && ~isempty(rightIdx)
        widths(i) = abs(wavelength(rightIdx) - wavelength(leftIdx));
    else
        widths(i) = wavelengthDiff * 2;
    end
end

% 绘图（可选）
if p.Results.Plot
    figure('Color', 'w');
    plot(wavelength, spectrum, 'b-', 'LineWidth', 2); hold on;
    plot(positions, peaks, 'r*', 'MarkerSize', 12, 'LineWidth', 2);
    
    for i = 1:length(peaks)
        text(positions(i), peaks(i) + max(spectrum)*0.05, ...
            sprintf('%.1f nm', positions(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
    end
    
    set(gca, 'FontName', 'Arial', 'FontSize', 12);
    xlabel('Wavelength (nm)', 'FontSize', 14);
    ylabel('Intensity (a.u.)', 'FontSize', 14);
    title('Peak Analysis', 'FontSize', 14);
    grid on;
    legend('Spectrum', 'Peaks');
end

fprintf('Found %d peaks\n', length(peaks));
fprintf('Positions (nm): %s\n', sprintf('%.1f ', positions));

end
