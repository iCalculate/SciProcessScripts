function [PowSerPL] = import_pl_data(filepath)
%IMPORT_PL_DATA 导入单个PL数据文件
%   [PowSerPL] = import_pl_data(filepath)
%   
%   输入:
%       filepath - 数据文件路径 (.mat 文件)
%   
%   输出:
%       PowSerPL - 标准数据结构
%           .data - 光谱数据矩阵 (行: 序列/功率, 列: 波长)
%           .axisscale{1,1} - 第一轴数据 (通常为功率/角度)
%           .axisscale{2,1} - 第二轴数据 (通常为波长 nm)
%
%   示例:
%       PowSerPL = import_pl_data("data.mat");

% 加载并提取第一个结构体变量
S = load(filepath);
varNames = fieldnames(S);
if isempty(varNames)
    error('文件 "%s" 不包含任何变量', filepath);
end

PowSerPL = S.(varNames{1});

% 验证数据结构
if ~isstruct(PowSerPL) || ~isfield(PowSerPL, 'data') || ~isfield(PowSerPL, 'axisscale')
    error('数据结构不符合要求，需要包含 .data 和 .axisscale 字段');
end

end
