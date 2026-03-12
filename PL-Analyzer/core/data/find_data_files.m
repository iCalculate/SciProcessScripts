function files = find_data_files(directory)
%FIND_DATA_FILES 在指定目录中查找所有 .mat 文件
%   files = find_data_files(directory)
%   
%   输入:
%       directory - 数据所在目录
%   
%   输出:
%       files - 找到的所有 .mat 文件名称（单元数组）

if ~isfolder(directory)
    error('目录不存在: %s', directory);
end

matFiles = dir(fullfile(directory, '*.mat'));

if isempty(matFiles)
    fprintf('未在 "%s" 中找到 .mat 文件\n', directory);
    files = {};
    return;
end

fprintf('在目录中找到 %d 个 .mat 文件:\n%s\n\n', length(matFiles), directory);
for i = 1:length(matFiles)
    fprintf('  [%d] %s\n', i, matFiles(i).name);
end
fprintf('\n');

files = {matFiles.name};

end
