function SHGData = load_shg_power_data(shg_filepath, power_filepath)
%LOAD_SHG_POWER_DATA 加载SHG光谱和功率数据
%   SHGData = load_shg_power_data(shg_filepath, power_filepath)
%
%   从两个.mat文件中加载SHG光谱数据和对应的功率扫描参数
%
%   输入:
%       shg_filepath - 包含SHG光谱的.mat文件路径
%       power_filepath - 包含功率参数的.mat文件路径
%
%   输出:
%       SHGData - 包含以下字段的结构体:
%           .data - (N×M) 矩阵，SHG强度数据
%           .axisscale{1,1} - (N×1) 功率向量 (mW)
%           .axisscale{2,1} - (M×1) 波长向量 (nm)
%
%   示例:
%       SHGData = load_shg_power_data('shg_spectra.mat', 'power.mat');

if ~isfile(shg_filepath)
    error('SHG data file not found: %s', shg_filepath);
end

if ~isfile(power_filepath)
    error('Power file not found: %s', power_filepath);
end

% 导入SHG光谱
fprintf('Loading SHG data...\n');
SHGData = import_shg_data(shg_filepath);

% 加载功率数据
fprintf('Loading power data...\n');
power_data = load(power_filepath);
power_field = fieldnames(power_data);

% 提取功率向量
power_vector = [];
for i = 1:length(power_field)
    field_val = power_data.(power_field{i});
    if isvector(field_val) && length(field_val) == size(SHGData.data, 1)
        power_vector = field_val(:);
        break;
    end
end

if isempty(power_vector)
    error('Could not find matching power vector for SHG data');
end

% 更新功率轴
SHGData.axisscale{1,1} = power_vector;

fprintf('✓ 成功加载SHG功率扫描数据\n');
fprintf('  数据尺寸: %d × %d\n', size(SHGData.data, 1), size(SHGData.data, 2));
fprintf('  功率范围: %.4f - %.4f mW\n', min(power_vector), max(power_vector));
fprintf('  波长范围: %.2f - %.2f nm\n', min(SHGData.axisscale{2,1}), max(SHGData.axisscale{2,1}));

end
