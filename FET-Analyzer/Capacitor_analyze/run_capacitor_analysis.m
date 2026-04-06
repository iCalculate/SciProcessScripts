function result = run_capacitor_analysis()
% RUN_CAPACITOR_ANALYSIS  Main entry point for capacitor analysis workflows.

scriptDir = fileparts(mfilename('fullpath'));
defaultDir = fullfile(scriptDir, 'raw_data');
if ~isfolder(defaultDir)
    defaultDir = scriptDir;
end

fprintf('\nCapacitor analysis mode:\n');
fprintf('  1. Single-file mode\n');
fprintf('  2. Multi-file mode\n');
fprintf('  3. Fitting mode\n');
fprintf('  4. Breakdown mode\n');
modeChoice = input('Select mode number: ');

if isempty(modeChoice) || ~ismember(modeChoice, [1, 2, 3, 4])
    error('Invalid mode selection.');
end

switch modeChoice
    case 1
        result = capacitor_mode_single_file(defaultDir);
    case 2
        result = capacitor_mode_multi_file(defaultDir);
    case 3
        result = capacitor_mode_fitting(defaultDir);
    case 4
        result = capacitor_mode_breakdown(defaultDir);
end
end
