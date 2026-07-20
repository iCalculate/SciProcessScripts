function test_single_wip_export(wip_file, output_folder, config_path)
%TEST_SINGLE_WIP_EXPORT Convenience wrapper for manual MATLAB testing.

if nargin < 2
    error('Usage: test_single_wip_export(wip_file, output_folder, config_path)');
end
if nargin < 3
    config_path = '';
end

summary = export_single_wip_spectra(wip_file, output_folder, config_path);
disp(summary);
end
