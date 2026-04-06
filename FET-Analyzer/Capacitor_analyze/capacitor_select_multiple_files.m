function filePaths = capacitor_select_multiple_files(defaultDir, dialogTitle, filterSpec)
fprintf('\nOpening file selection dialog: %s\n', dialogTitle);
[fileNames, folderPath] = uigetfile(filterSpec, dialogTitle, defaultDir, 'MultiSelect', 'on');
if isequal(fileNames, 0)
    error('No file selected. Analysis cancelled.');
end

if ischar(fileNames) || isstring(fileNames)
    fileNames = {char(fileNames)};
end

filePaths = cellfun(@(name) fullfile(folderPath, name), fileNames, 'UniformOutput', false);
fprintf('Selected %d file(s).\n', numel(filePaths));
for k = 1:numel(filePaths)
    fprintf('  [%d] %s\n', k, filePaths{k});
end
end
