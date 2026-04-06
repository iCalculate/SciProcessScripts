function filePath = capacitor_select_single_file(defaultDir, dialogTitle, filterSpec)
fprintf('\nOpening file selection dialog: %s\n', dialogTitle);
[fileName, folderPath] = uigetfile(filterSpec, dialogTitle, defaultDir, 'MultiSelect', 'off');
if isequal(fileName, 0)
    error('No file selected. Analysis cancelled.');
end
filePath = fullfile(folderPath, fileName);
fprintf('Selected file: %s\n', filePath);
end
