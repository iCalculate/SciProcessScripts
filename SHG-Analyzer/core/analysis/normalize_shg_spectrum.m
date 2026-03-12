function spectrum_norm = normalize_shg_spectrum(spectrum, method, varargin)
%NORMALIZE_SHG_SPECTRUM SHG光谱归一化处理
%   spectrum_norm = normalize_shg_spectrum(spectrum, method, Name, Value)
%
%   对SHG光谱进行不同方式的归一化处理
%
%   输入:
%       spectrum - 光谱强度向量或矩阵
%       method - 归一化方法:
%           'max' - 按最大值归一化
%           'peak_range' - 按峰值范围内的最大值归一化
%           'area' - 按曲线下面积归一化
%           'none' - 不进行归一化
%
%   可选参数:
%       'PeakRange' - 峰值范围 [min max] nm (用于 'peak_range' 方法)
%       'Wavelength' - 波长向量 (用于 'area' 方法和 'peak_range' 方法)
%
%   输出:
%       spectrum_norm - 归一化后的光谱
%
%   示例:
%       spectrum = randn(1, 100);
%       spectrum_norm = normalize_shg_spectrum(spectrum, 'max');

p = inputParser;
addRequired(p, 'spectrum', @isnumeric);
addRequired(p, 'method', @(x) ismember(x, {'max', 'peak_range', 'area', 'none'}));
addParameter(p, 'PeakRange', [], @(x) isempty(x) || (isvector(x) && length(x) == 2));
addParameter(p, 'Wavelength', [], @(x) isempty(x) || isvector(x));
parse(p, spectrum, method, varargin{:});

spectrum_norm = spectrum;
is_matrix = ndims(spectrum) > 2 || (ndims(spectrum) == 2 && size(spectrum, 1) > 1);

switch method
    case 'max'
        if is_matrix
            for i = 1:size(spectrum_norm, 1)
                max_val = max(spectrum_norm(i, :));
                if max_val > 0
                    spectrum_norm(i, :) = spectrum_norm(i, :) / max_val;
                end
            end
        else
            max_val = max(spectrum_norm);
            if max_val > 0
                spectrum_norm = spectrum_norm / max_val;
            end
        end
        
    case 'peak_range'
        if isempty(p.Results.Wavelength) || isempty(p.Results.PeakRange)
            error('Wavelength and PeakRange required for peak_range normalization');
        end
        
        wavelength = p.Results.Wavelength(:);
        peak_idx = wavelength >= p.Results.PeakRange(1) & wavelength <= p.Results.PeakRange(2);
        
        if is_matrix
            for i = 1:size(spectrum_norm, 1)
                max_val = max(spectrum_norm(i, peak_idx));
                if max_val > 0
                    spectrum_norm(i, :) = spectrum_norm(i, :) / max_val;
                end
            end
        else
            max_val = max(spectrum_norm(peak_idx));
            if max_val > 0
                spectrum_norm = spectrum_norm / max_val;
            end
        end
        
    case 'area'
        if isempty(p.Results.Wavelength)
            error('Wavelength required for area normalization');
        end
        
        wavelength = p.Results.Wavelength(:);
        
        if is_matrix
            for i = 1:size(spectrum_norm, 1)
                area_val = trapz(wavelength, spectrum_norm(i, :));
                if area_val > 0
                    spectrum_norm(i, :) = spectrum_norm(i, :) / area_val;
                end
            end
        else
            area_val = trapz(wavelength, spectrum_norm);
            if area_val > 0
                spectrum_norm = spectrum_norm / area_val;
            end
        end
        
    case 'none'
        % 不做任何处理
end

end
