function SHGData = import_shg_data(filepath)
%IMPORT_SHG_DATA 从.mat文件导入单个SHG数据
%   SHGData = import_shg_data(filepath)
%
%   从MATLAB .mat文件中导入SHG光谱数据
%
%   输入:
%       filepath - .mat文件的路径
%
%   输出:
%       SHGData - 包含以下字段的结构体:
%           .data - (N×M) 矩阵，SHG强度数据
%           .axisscale{1,1} - (N×1) 第一轴参数(可选)
%           .axisscale{2,1} - (M×1) 波长向量 (nm)
%
%   示例:
%       SHGData = import_shg_data('shg_data.mat');

if ~isfile(filepath)
    error('File not found: %s', filepath);
end

% 加载数据
data = load(filepath);
field_names = fieldnames(data);

% 查找包含数据的字段
SHGData = struct();

if length(field_names) == 1
    % 如果只有一个字段，假设是数据
    SHGData.data = data.(field_names{1});
else
    % 尝试查找特定字段名
    for i = 1:length(field_names)
        field_val = data.(field_names{i});
        if isnumeric(field_val) && size(field_val, 2) > 10  % 假设波长维度较大
            SHGData.data = field_val;
            break;
        end
    end
    
    if ~isfield(SHGData, 'data')
        error('Could not identify SHG data matrix in file');
    end
end

% 初始化坐标轴
if size(SHGData.data, 1) == 1
    % 单条光谱
    SHGData.axisscale{1,1} = 1;
else
    % 多条光谱，假设第一轴为索引
    SHGData.axisscale{1,1} = (1:size(SHGData.data, 1))';
end

% 第二轴为波长(默认为400-900 nm的均匀分布)
if isfield(data, 'wavelength') || isfield(data, 'wl') || isfield(data, 'lambda')
    % 尝试从文件中提取波长信息
    if isfield(data, 'wavelength')
        SHGData.axisscale{2,1} = data.wavelength(:);
    elseif isfield(data, 'wl')
        SHGData.axisscale{2,1} = data.wl(:);
    else
        SHGData.axisscale{2,1} = data.lambda(:);
    end
else
    % 默认生成波长轴
    n_wavelength = size(SHGData.data, 2);
    SHGData.axisscale{2,1} = linspace(400, 900, n_wavelength)';
end

fprintf('✓ 成功导入SHG数据\n');
fprintf('  数据尺寸: %d × %d\n', size(SHGData.data, 1), size(SHGData.data, 2));
fprintf('  第一轴范围: %.2f - %.2f\n', min(SHGData.axisscale{1,1}), max(SHGData.axisscale{1,1}));
fprintf('  波长范围: %.2f - %.2f nm\n', min(SHGData.axisscale{2,1}), max(SHGData.axisscale{2,1}));

end
