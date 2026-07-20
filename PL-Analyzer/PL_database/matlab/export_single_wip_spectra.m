function summary = export_single_wip_spectra(wip_file, output_folder, config_path)
%EXPORT_SINGLE_WIP_SPECTRA Export extracted 1D spectra from a single WIP file.

if nargin < 3
    config_path = '';
end

options = read_pldb_config(config_path);
[~, stem_name, ~] = fileparts(wip_file);
safe_name = make_safe_name(stem_name);
file_output = fullfile(output_folder, safe_name);
if ~exist(file_output, 'dir')
    mkdir(file_output);
end

summary = struct( ...
    'source_wip', wip_file, ...
    'status', 'success', ...
    'metadata_path', '', ...
    'exported_spectra', 0, ...
    'detected_inventory', struct(), ...
    'spectra', {{}}, ...
    'extraction_backend', '', ...
    'error_message', '');

try
    [spectra, extraction_backend] = extract_spectra_headlessly(wip_file, config_path, options.witec);

    metadata = cell(0, 1);
    for idx = 1:numel(spectra)
        spectrum_id = generate_spectrum_id(wip_file, spectra(idx).source_tree_path, idx);
        csv_path = fullfile(file_output, sprintf('%s_%03d.csv', safe_name, idx));
        write_spectrum_csv(csv_path, spectra(idx).x_axis, spectra(idx).intensity);
        metadata{end + 1, 1} = build_metadata_entry(spectra(idx), wip_file, extraction_backend, csv_path, spectrum_id); %#ok<AGROW>
    end

    metadata_path = fullfile(file_output, [safe_name '_metadata.json']);
    write_text_file(metadata_path, jsonencode(metadata, 'PrettyPrint', true));
    summary.metadata_path = metadata_path;
    summary.exported_spectra = numel(metadata);
    summary.detected_inventory = summarize_inventory(metadata);
    summary.spectra = metadata;
    summary.extraction_backend = extraction_backend;
    if isempty(metadata)
        summary.status = 'no_spectra';
        summary.error_message = sprintf('No supported spectra candidates were found via %s.', extraction_backend);
    end
catch ME
    summary.status = 'failed';
    summary.error_message = getReport(ME, 'extended', 'hyperlinks', 'off');
end

root_summary = struct( ...
    'input_path', wip_file, ...
    'mock_mode', false, ...
    'total_files', 1, ...
    'processed_files', 1, ...
    'exported_spectra', summary.exported_spectra, ...
    'failed_files', double(~strcmp(summary.status, 'success') && ~strcmp(summary.status, 'no_spectra')), ...
    'detected_inventory', summary.detected_inventory, ...
    'files', {{summary}}, ...
    'generated_at', char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX')));
write_text_file(fullfile(output_folder, 'import_summary.json'), jsonencode(root_summary, 'PrettyPrint', true));
end


function metadata = build_metadata_entry(spectrum, wip_file, extraction_backend, csv_path, spectrum_id)
metadata = struct( ...
    'spectrum_id', spectrum_id, ...
    'source_wip', wip_file, ...
    'source_tree_path', spectrum.source_tree_path, ...
    'spectrum_type', spectrum.spectrum_type, ...
    'x_axis_unit', spectrum.x_axis_unit, ...
    'n_points', spectrum.n_points, ...
    'acquisition_mode', get_optional_field(spectrum, 'acquisition_mode', ''), ...
    'trace_index', get_optional_field(spectrum, 'trace_index', []), ...
    'trace_count', get_optional_field(spectrum, 'trace_count', []), ...
    'scan_size_x', get_optional_field(spectrum, 'scan_size_x', []), ...
    'scan_size_y', get_optional_field(spectrum, 'scan_size_y', []), ...
    'grid_x', get_optional_field(spectrum, 'grid_x', []), ...
    'grid_y', get_optional_field(spectrum, 'grid_y', []), ...
    'measurement_time', get_optional_field(spectrum, 'measurement_time', ''), ...
    'laser_wavelength', get_optional_field(spectrum, 'laser_wavelength', ''), ...
    'laser_power', get_optional_field(spectrum, 'laser_power', ''), ...
    'integration_time', get_optional_field(spectrum, 'integration_time', ''), ...
    'grating', get_optional_field(spectrum, 'grating', ''), ...
    'objective', get_optional_field(spectrum, 'objective', ''), ...
    'measurement_config', build_measurement_config(spectrum, extraction_backend), ...
    'csv_path', csv_path, ...
    'status', 'success', ...
    'error_message', '');
end


function config = build_measurement_config(spectrum, extraction_backend)
config = struct( ...
    'acquisition_mode', get_optional_field(spectrum, 'acquisition_mode', ''), ...
    'trace_index', get_optional_field(spectrum, 'trace_index', []), ...
    'trace_count', get_optional_field(spectrum, 'trace_count', []), ...
    'scan_size_x', get_optional_field(spectrum, 'scan_size_x', []), ...
    'scan_size_y', get_optional_field(spectrum, 'scan_size_y', []), ...
    'grid_x', get_optional_field(spectrum, 'grid_x', []), ...
    'grid_y', get_optional_field(spectrum, 'grid_y', []), ...
    'secondary_axis_kind', get_optional_field(spectrum, 'secondary_axis_kind', ''), ...
    'secondary_axis_unit', get_optional_field(spectrum, 'secondary_axis_unit', ''), ...
    'secondary_axis_value', get_optional_field(spectrum, 'secondary_axis_value', []), ...
    'x_min', min(double(spectrum.x_axis(:))), ...
    'x_max', max(double(spectrum.x_axis(:))), ...
    'n_points', double(spectrum.n_points), ...
    'x_axis_unit', spectrum.x_axis_unit, ...
    'extraction_backend', extraction_backend);
end


function inventory = summarize_inventory(metadata)
inventory = struct();
for idx = 1:numel(metadata)
    if iscell(metadata)
        entry = metadata{idx};
    else
        entry = metadata(idx);
    end
    label = char(get_optional_field(entry, 'acquisition_mode', 'unknown'));
    if isempty(label)
        label = 'unknown';
    end
    if ~isfield(inventory, label)
        inventory.(label) = 0;
    end
    inventory.(label) = inventory.(label) + 1;
end
end


function value = get_optional_field(payload, field_name, default_value)
if isfield(payload, field_name) && ~isempty(payload.(field_name))
    value = payload.(field_name);
else
    value = default_value;
end
end


function write_spectrum_csv(csv_path, x_axis, intensity)
fid = fopen(csv_path, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'x_axis,intensity\n');
for idx = 1:numel(x_axis)
    fprintf(fid, '%.12g,%.12g\n', x_axis(idx), intensity(idx));
end
end


function safe_name = make_safe_name(raw_name)
safe_name = regexprep(raw_name, '[^a-zA-Z0-9_\-]+', '_');
if isempty(safe_name)
    safe_name = 'wip_export';
end
end


function spectrum_id = generate_spectrum_id(wip_file, source_tree_path, index)
payload = uint8(char(sprintf('%s|%s|%d', wip_file, source_tree_path, index)));
engine = java.security.MessageDigest.getInstance('SHA-1');
engine.update(payload);
digest = typecast(engine.digest(), 'uint8');
hex_text = lower(reshape(dec2hex(digest, 2).', 1, []));
spectrum_id = ['plspec-' hex_text(1:16)];
end


function write_text_file(path_text, content)
fid = fopen(path_text, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', content);
end


function [spectra, extraction_backend] = extract_spectra_headlessly(wip_file, config_path, witec_options)
[O_wid, ~, O_wit] = read_wip_with_witio(wip_file, config_path);
S_DT = WITio.obj.wit.DataTree_get(O_wit);
spectra = extract_1d_spectra_from_wid(O_wid, wip_file, witec_options, S_DT);
extraction_backend = 'WITio wid objects';

if ~isempty(spectra)
    return;
end

spectra = extract_1d_spectra_from_witree(S_DT, wip_file, witec_options);
extraction_backend = 'WITio DataTree fallback';
end
