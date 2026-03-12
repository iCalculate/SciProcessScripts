function normalizedSpectrum = normalize_spectrum(spectrum, wavelengthAxis, method, peakRange)
%NORMALIZE_SPECTRUM 标准化光谱数据
%   normalizedSpectrum = normalize_spectrum(spectrum, wavelengthAxis, method, peakRange)
%
%   输入:
%       spectrum - 光谱强度数据
%       wavelengthAxis - 波长轴数据
%       method - 归一化方法:
%           'none' - 不归一化
%           'max' - 按最大值归一化 (0-1)
%           'peak_range' - 按特定范围的峰值归一化 (0-1)
%           'area' - 按积分面积归一化
%       peakRange - [min max] 峰值范围 (nm)，仅用于 'peak_range' 方法
%
%   输出:
%       normalizedSpectrum - 归一化后的光谱
%
%   示例:
%       normalized = normalize_spectrum(spectrum, wavelength, 'peak_range', [650 700]);

switch method
    case 'none'
        normalizedSpectrum = spectrum;
        
    case 'max'
        maxVal = max(spectrum);
        if maxVal > 0
            normalizedSpectrum = spectrum / maxVal;
        else
            normalizedSpectrum = spectrum;
        end
        
    case 'peak_range'
        peakMask = wavelengthAxis >= peakRange(1) & wavelengthAxis <= peakRange(2);
        if any(peakMask)
            peakVal = max(spectrum(peakMask));
        else
            peakVal = max(spectrum);
        end
        
        if peakVal > 0
            normalizedSpectrum = spectrum / peakVal;
        else
            normalizedSpectrum = spectrum;
        end
        
        % 平移至0并重新归一化
        minVal = min(normalizedSpectrum);
        normalizedSpectrum = normalizedSpectrum - minVal;
        maxVal = max(normalizedSpectrum);
        
        if maxVal > 0
            normalizedSpectrum = normalizedSpectrum / maxVal;
        end
        
    case 'area'
        area = trapz(wavelengthAxis, spectrum);
        if area > 0
            normalizedSpectrum = spectrum / area;
        else
            normalizedSpectrum = spectrum;
        end
        
    otherwise
        error('未知的归一化方法: %s', method);
end

end
