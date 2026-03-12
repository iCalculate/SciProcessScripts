function [PowSerPL] = load_pl_power_data(pl_filepath, power_filepath)
%LOAD_PL_POWER_DATA 加载PL光谱数据和激光功率数据
%   [PowSerPL] = load_pl_power_data(pl_filepath, power_filepath)
%   
%   输入:
%       pl_filepath - PL光谱数据文件路径
%       power_filepath - 激光功率数据文件路径
%   
%   输出:
%       PowSerPL - 合并后的数据结构
%
%   示例:
%       PowSerPL = load_pl_power_data('pl_data.mat', 'power_data.mat');

try
    PowSerPL = import_pl_data(pl_filepath);
    fprintf('✓ PL数据加载成功\n');
catch ME
    error('PL数据加载失败: %s', ME.message);
end

numPowers = size(PowSerPL.data, 1);

try
    laserPower = extract_laser_power(power_filepath);
    fprintf('✓ 激光功率数据加载成功\n');
catch ME
    error('激光功率数据加载失败: %s', ME.message);
end

% 合并数据
laserPower = laserPower(:);

if length(laserPower) ~= numPowers
    minLen = min(length(laserPower), numPowers);
    if minLen < numPowers
        warning('功率数据长度 (%d) 少于PL数据行数 (%d)。已截断。', ...
            length(laserPower), numPowers);
    else
        warning('功率数据长度 (%d) 多于PL数据行数 (%d)。已截断。', ...
            length(laserPower), numPowers);
    end
    PowSerPL.data = PowSerPL.data(1:minLen, :);
    laserPower = laserPower(1:minLen);
end

PowSerPL.axisscale{1,1} = laserPower;

fprintf('✓ 数据合并完成\n');
fprintf('  有效功率点数: %d\n', length(laserPower));
fprintf('  功率范围: %.4f - %.4f mW\n', min(laserPower), max(laserPower));

end

function laserPower = extract_laser_power(power_filepath)
S = load(power_filepath);
varNames = fieldnames(S);

laserPower = [];
foundVarName = '';

% 优先查找包含 'Power' 或 'Laser' 的结构体
for i = 1:length(varNames)
    varName = varNames{i};
    if (contains(varName, 'Power', 'IgnoreCase', true) || ...
        contains(varName, 'Laser', 'IgnoreCase', true)) && ...
        isstruct(S.(varName)) && isfield(S.(varName), 'data')
        laserPower = S.(varName).data;
        foundVarName = varName;
        break;
    end
end

if isempty(laserPower)
    for i = 1:length(varNames)
        varName = varNames{i};
        if isstruct(S.(varName)) && isfield(S.(varName), 'data')
            data = S.(varName).data;
            if isvector(data) && length(data) > 0 && length(data) <= 10000
                laserPower = data;
                foundVarName = varName;
                break;
            end
        end
    end
end

if isempty(laserPower)
    error('未能找到激光功率数据。请检查文件结构。');
end

fprintf('  找到功率数据: %s (%d 个数据点)\n', foundVarName, length(laserPower));

end
