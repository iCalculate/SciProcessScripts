function options = read_pldb_config(config_path)
%READ_PLDB_CONFIG Read the small subset of config.yaml needed by MATLAB helpers.

if nargin < 1
    config_path = '';
end

options.matlab = struct( ...
    'witio_root', '');
options.witec = struct( ...
    'extraction_mode', 'auto', ...
    'manual_x_paths', {{}}, ...
    'manual_y_paths', {{}}, ...
    'skip_keywords', {{'image', 'map', 'mapping', 'topography', 'video', 'camera', 'hyperspectral'}}, ...
    'spectrum_keywords', {{'spectrum', 'graph', 'raman', 'pl', 'photoluminescence', 'intensity'}}, ...
    'include_point_spectra', true, ...
    'include_line_scans', false, ...
    'include_area_maps', false, ...
    'include_series_scans', false);

if isempty(config_path) || ~isfile(config_path)
    return;
end

lines = splitlines(string(fileread(config_path)));
section = "";
list_key = "";

for idx = 1:numel(lines)
    line = strtrim(lines(idx));
    if line == "" || startsWith(line, "#")
        continue;
    end
    if (line == "matlab:" || line == "witec:") && ~startsWith(line, "-")
        section = erase(line, ":");
        list_key = "";
        continue;
    end
    if endsWith(line, ":") && ~startsWith(line, "-")
        list_key = erase(line, ":");
        if section == "witec"
            switch char(list_key)
                case 'manual_x_paths'
                    options.witec.manual_x_paths = {};
                case 'manual_y_paths'
                    options.witec.manual_y_paths = {};
                case 'skip_keywords'
                    options.witec.skip_keywords = {};
                case 'spectrum_keywords'
                    options.witec.spectrum_keywords = {};
            end
        end
        continue;
    end

    switch char(section)
        case 'matlab'
            if contains(line, ":")
                parts = split(line, ":");
                key = strtrim(parts(1));
                value = sanitize_scalar_value(join(parts(2:end), ":"));
                list_key = "";
                switch char(key)
                    case 'witio_root'
                        options.matlab.witio_root = char(value);
                end
            end

        case 'witec'
            if startsWith(line, "- ")
                value = sanitize_scalar_value(extractAfter(line, 2));
                switch char(list_key)
                    case 'manual_x_paths'
                        options.witec.manual_x_paths{end + 1} = char(value); %#ok<AGROW>
                    case 'manual_y_paths'
                        options.witec.manual_y_paths{end + 1} = char(value); %#ok<AGROW>
                    case 'skip_keywords'
                        options.witec.skip_keywords{end + 1} = char(value); %#ok<AGROW>
                    case 'spectrum_keywords'
                        options.witec.spectrum_keywords{end + 1} = char(value); %#ok<AGROW>
                end
            elseif contains(line, ":")
                parts = split(line, ":");
                key = strtrim(parts(1));
                value = sanitize_scalar_value(join(parts(2:end), ":"));
                list_key = "";
                switch char(key)
                    case 'extraction_mode'
                        options.witec.extraction_mode = char(value);
                    case 'include_point_spectra'
                        options.witec.include_point_spectra = parse_logical_value(value, true);
                    case 'include_line_scans'
                        options.witec.include_line_scans = parse_logical_value(value, false);
                    case 'include_area_maps'
                        options.witec.include_area_maps = parse_logical_value(value, false);
                    case 'include_series_scans'
                        options.witec.include_series_scans = parse_logical_value(value, false);
                end
            end
    end
end
end


function value = sanitize_scalar_value(raw_value)
value = strtrim(string(raw_value));
value = strip(value, '"');
value = strip(value, "'");
end


function tf = parse_logical_value(value, default_value)
if nargin < 2
    default_value = false;
end

text_value = lower(strtrim(char(value)));
if any(strcmp(text_value, {'true', 'yes', '1', 'on'}))
    tf = true;
elseif any(strcmp(text_value, {'false', 'no', '0', 'off'}))
    tf = false;
else
    tf = default_value;
end
end
