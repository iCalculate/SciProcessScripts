function summary = export_folder_wip_spectra(root_folder, output_folder, force_reimport, config_path)
%EXPORT_FOLDER_WIP_SPECTRA Recursively export 1D spectra from a folder of WIP files.

if nargin < 3
    force_reimport = false;
end
if nargin < 4
    config_path = '';
end
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

wip_files = find_wip_files(root_folder);
file_summaries = repmat(struct(), 0, 1);
log_lines = cell(0, 1);
failed_files = 0;
exported_spectra = 0;
detected_inventory = struct();

for idx = 1:numel(wip_files)
    wip_file = wip_files{idx};
    [~, stem_name, ~] = fileparts(wip_file);
    safe_name = regexprep(stem_name, '[^a-zA-Z0-9_\-]+', '_');
    metadata_path = fullfile(output_folder, safe_name, [safe_name '_metadata.json']);

    if ~force_reimport && isfile(metadata_path)
        entry = struct( ...
            'source_wip', wip_file, ...
            'status', 'skipped', ...
            'metadata_path', metadata_path, ...
            'exported_spectra', 0, ...
            'detected_inventory', struct(), ...
            'spectra', {{}}, ...
            'error_message', 'Skipped because prior metadata exists.');
        file_summaries(end + 1) = entry; %#ok<AGROW>
        log_lines{end + 1} = sprintf('[%s] skipped %s', timestamp_utc(), wip_file); %#ok<AGROW>
        continue;
    end

    entry = export_single_wip_spectra(wip_file, output_folder, config_path);
    file_summaries(end + 1) = entry; %#ok<AGROW>
    exported_spectra = exported_spectra + double(entry.exported_spectra);
    detected_inventory = merge_inventory(detected_inventory, entry.detected_inventory);
    if strcmp(entry.status, 'failed')
        failed_files = failed_files + 1;
    end
    log_lines{end + 1} = sprintf('[%s] %s %s', timestamp_utc(), lower(entry.status), wip_file); %#ok<AGROW>
end

summary = struct( ...
    'input_path', root_folder, ...
    'mock_mode', false, ...
    'total_files', numel(wip_files), ...
    'processed_files', numel(wip_files), ...
    'exported_spectra', exported_spectra, ...
    'failed_files', failed_files, ...
    'detected_inventory', detected_inventory, ...
    'files', {file_summaries}, ...
    'generated_at', timestamp_utc());

write_text_file(fullfile(output_folder, 'import_summary.json'), jsonencode(summary, 'PrettyPrint', true));
write_text_file(fullfile(output_folder, 'import_log.txt'), strjoin(log_lines, newline));
end


function files = find_wip_files(root_folder)
entries = dir(fullfile(root_folder, '**', '*.wip'));
files = cell(numel(entries), 1);
for idx = 1:numel(entries)
    files{idx} = fullfile(entries(idx).folder, entries(idx).name);
end
end


function inventory = merge_inventory(inventory, incoming)
if ~isstruct(incoming)
    return;
end

fields = fieldnames(incoming);
for idx = 1:numel(fields)
    field_name = fields{idx};
    if ~isfield(inventory, field_name)
        inventory.(field_name) = 0;
    end
    inventory.(field_name) = inventory.(field_name) + incoming.(field_name);
end
end


function write_text_file(path_text, content)
fid = fopen(path_text, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', content);
end


function text_value = timestamp_utc()
text_value = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
end
