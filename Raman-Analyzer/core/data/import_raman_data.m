function [RamanData] = import_raman_data(filepath)
%IMPORT_RAMAN_DATA 导入单个拉曼数据文件
%   [RamanData] = import_raman_data(filepath)
%   
%   输入:
%       filepath - 数据文件路径 (.mat 文件)
%   
%   输出:
%       RamanData - 标准数据结构
%           .data - 拉曼光谱数据矩阵 (行: 序列/功率, 列: 拉曼位移)
%           .axisscale{1,1} - 第一轴数据 (通常为功率/温度 等)
%           .axisscale{2,1} - 第二轴数据 (拉曼位移 cm⁻¹)
%
%   示例:
%       RamanData = import_raman_data("raman.mat");

% 加载并提取第一个结构体变量
S = load(filepath);
varNames = fieldnames(S);
if isempty(varNames)
    error('文件 "%s" 不包含任何变量', filepath);
end

RamanData = S.(varNames{1});

% 验证数据结构
if ~isstruct(RamanData) || ~isfield(RamanData, 'data') || ~isfield(RamanData, 'axisscale')
    error('数据结构不符合要求，需要包含 .data 和 .axisscale 字段');
end

fprintf('✓ 拉曼数据导入成功\n');
fprintf('  数据大小: %d × %d\n', size(RamanData.data));
fprintf('  拉曼位移范围: %.0f - %.0f cm⁻¹\n', min(RamanData.axisscale{2,1}), max(RamanData.axisscale{2,1}));

end
