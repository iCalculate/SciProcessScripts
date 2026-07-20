function spectra = extract_1d_spectra_from_witree(data_tree, source_wip, options)
%EXTRACT_1D_SPECTRA_FROM_WITREE Conservatively extract pure 1D spectra.

if nargin < 3 || isempty(options)
    options = default_options();
else
    options = merge_default_options(options, default_options());
end

numeric_entries = collect_numeric_entries(data_tree, 'S_DT', empty_numeric_entries());
vector_entries = numeric_entries(strcmp({numeric_entries.kind}, 'vector'));
vector_entries = vector_entries(arrayfun(@(item) is_valid_vector(item, options), vector_entries));

if strcmpi(options.extraction_mode, 'manual') && ...
        ~isempty(options.manual_x_paths) && ~isempty(options.manual_y_paths)
    spectra = extract_manual_pairs(vector_entries, source_wip, options);
    if ~isempty(spectra)
        return;
    end
end

spectra = extract_auto_pairs(vector_entries, source_wip, options);
end


function options = default_options()
options.extraction_mode = 'auto';
options.manual_x_paths = {};
options.manual_y_paths = {};
options.skip_keywords = {'image', 'map', 'mapping', 'topography', 'video', 'camera', 'hyperspectral'};
options.spectrum_keywords = {'spectrum', 'graph', 'raman', 'pl', 'photoluminescence', 'intensity'};
options.include_point_spectra = true;
options.include_line_scans = false;
options.include_area_maps = false;
options.include_series_scans = false;
end


function entries = empty_numeric_entries()
entries = struct( ...
    'path', {}, ...
    'data', {}, ...
    'original', {}, ...
    'kind', {}, ...
    'length', {}, ...
    'score', {});
end


function spectra = empty_spectra()
spectra = struct( ...
    'x_axis', {}, ...
    'intensity', {}, ...
    'source_tree_path', {}, ...
    'spectrum_type', {}, ...
    'x_axis_unit', {}, ...
    'n_points', {}, ...
    'source_wip', {}, ...
    'acquisition_mode', {}, ...
    'trace_index', {}, ...
    'trace_count', {}, ...
    'scan_size_x', {}, ...
    'scan_size_y', {}, ...
    'grid_x', {}, ...
    'grid_y', {}, ...
    'secondary_axis_kind', {}, ...
    'secondary_axis_unit', {}, ...
    'secondary_axis_value', {}, ...
    'measurement_time', {}, ...
    'laser_wavelength', {}, ...
    'laser_power', {}, ...
    'integration_time', {}, ...
    'grating', {}, ...
    'objective', {});
end


function merged = merge_default_options(input_options, defaults)
merged = defaults;
fields = fieldnames(defaults);
for idx = 1:numel(fields)
    field_name = fields{idx};
    if isfield(input_options, field_name) && ~isempty(input_options.(field_name))
        merged.(field_name) = input_options.(field_name);
    end
end
end


function entries = collect_numeric_entries(node, path_text, entries)
if isnumeric(node) || islogical(node)
    if ~isempty(node)
        entry.path = path_text;
        entry.data = double(node(:));
        entry.original = double(node);
        entry.kind = classify_array(node);
        entry.length = numel(node);
        entry.score = keyword_score(path_text, {'xaxis', 'yaxis', 'spectrum', 'graph', 'data'});
        entries(end + 1) = entry; %#ok<AGROW>
    end
    return;
end

if isstruct(node)
    fields = fieldnames(node);
    for idx = 1:numel(fields)
        field_name = fields{idx};
        entries = collect_numeric_entries(node.(field_name), [path_text '.' field_name], entries);
    end
    return;
end

if iscell(node)
    for idx = 1:numel(node)
        entries = collect_numeric_entries(node{idx}, sprintf('%s{%d}', path_text, idx), entries);
    end
    return;
end

if isobject(node)
    props = properties(node);
    for idx = 1:numel(props)
        prop_name = props{idx};
        try
            child = node.(prop_name);
        catch
            continue;
        end
        entries = collect_numeric_entries(child, [path_text '.' prop_name], entries);
    end
end
end


function kind = classify_array(value)
if isvector(value)
    kind = 'vector';
elseif ndims(value) == 2
    kind = 'matrix';
elseif ndims(value) == 3
    kind = 'cube';
else
    kind = 'other';
end
end


function tf = is_valid_vector(entry, options)
path_text = lower(entry.path);
if entry.length < 16
    tf = false;
    return;
end
if should_skip_path(path_text, options.skip_keywords)
    tf = false;
    return;
end
values = entry.data;
if any(~isfinite(values))
    tf = false;
    return;
end
if max(values) == min(values)
    tf = false;
    return;
end
tf = true;
end


function spectra = extract_manual_pairs(vector_entries, source_wip, options)
spectra = empty_spectra();
if ~options.include_point_spectra
    return;
end
x_entries = resolve_paths(vector_entries, options.manual_x_paths);
y_entries = resolve_paths(vector_entries, options.manual_y_paths);
pair_count = min(numel(x_entries), numel(y_entries));
for idx = 1:pair_count
    x_entry = x_entries(idx);
    y_entry = y_entries(idx);
    if x_entry.length ~= y_entry.length
        continue;
    end
    spectra(end + 1) = make_spectrum_struct(x_entry, y_entry, source_wip); %#ok<AGROW>
end
end


function matched = resolve_paths(entries, path_list)
matched = empty_numeric_entries();
for idx = 1:numel(path_list)
    expected = lower(strtrim(path_list{idx}));
    for jdx = 1:numel(entries)
        if strcmpi(entries(jdx).path, expected) || strcmpi(lower(entries(jdx).path), expected)
            matched(end + 1) = entries(jdx); %#ok<AGROW>
            break;
        end
    end
end
end


function spectra = extract_auto_pairs(vector_entries, source_wip, options)
spectra = empty_spectra();
if ~options.include_point_spectra
    return;
end
used_y_paths = {};
for idx = 1:numel(vector_entries)
    y_entry = vector_entries(idx);
    if is_probable_x_axis(y_entry.original)
        continue;
    end
    if ~is_intensity_candidate(y_entry.path, options)
        continue;
    end
    [x_entry, score] = find_best_x_axis(vector_entries, y_entry, options);
    if score < 2
        continue;
    end
    if any(strcmp(used_y_paths, y_entry.path))
        continue;
    end
    spectra(end + 1) = make_spectrum_struct(x_entry, y_entry, source_wip); %#ok<AGROW>
    used_y_paths{end + 1} = y_entry.path; %#ok<AGROW>
end
end


function tf = is_intensity_candidate(path_text, options)
lowered = lower(path_text);
tf = keyword_score(lowered, options.spectrum_keywords) >= 1 || contains(lowered, 'data');
if contains(lowered, 'xaxis')
    tf = false;
end
end


function [best_entry, best_score] = find_best_x_axis(entries, y_entry, options)
best_entry = struct([]);
best_score = -inf;
for idx = 1:numel(entries)
    x_entry = entries(idx);
    if strcmp(x_entry.path, y_entry.path)
        continue;
    end
    if x_entry.length ~= y_entry.length
        continue;
    end
    if ~is_probable_x_axis(x_entry.original)
        continue;
    end
    score = 0;
    if shares_parent_path(x_entry.path, y_entry.path)
        score = score + 2;
    end
    if contains(lower(x_entry.path), 'xaxis')
        score = score + 3;
    end
    if contains(lower(x_entry.path), 'wavelength') || contains(lower(x_entry.path), 'energy')
        score = score + 2;
    end
    if keyword_score(lower(y_entry.path), options.spectrum_keywords) >= 1
        score = score + 1;
    end
    if score > best_score
        best_entry = x_entry;
        best_score = score;
    end
end
end


function tf = shares_parent_path(path_a, path_b)
parent_a = regexprep(path_a, '\.[^\.]+$', '');
parent_b = regexprep(path_b, '\.[^\.]+$', '');
tf = strcmp(parent_a, parent_b);
end


function tf = should_skip_path(path_text, skip_keywords)
tf = false;
for idx = 1:numel(skip_keywords)
    if contains(path_text, lower(skip_keywords{idx}))
        tf = true;
        return;
    end
end
end


function score = keyword_score(path_text, keywords)
score = 0;
for idx = 1:numel(keywords)
    if contains(lower(path_text), lower(keywords{idx}))
        score = score + 1;
    end
end
end


function tf = is_probable_x_axis(array_value)
values = double(array_value(:));
if numel(values) < 16
    tf = false;
    return;
end
delta = diff(values);
tf = all(delta >= 0) || all(delta <= 0);
end


function spectrum = make_spectrum_struct(x_entry, y_entry, source_wip)
x_axis = double(x_entry.original(:));
intensity = double(y_entry.original(:));
spectrum.x_axis = x_axis;
spectrum.intensity = intensity;
spectrum.source_tree_path = y_entry.path;
spectrum.x_axis_unit = guess_x_axis_unit(x_axis);
spectrum.spectrum_type = infer_spectrum_type_from_axis_local(x_axis, spectrum.x_axis_unit);
spectrum.n_points = numel(x_axis);
spectrum.source_wip = source_wip;
spectrum.acquisition_mode = 'point_spectrum';
spectrum.trace_index = 0;
spectrum.trace_count = 1;
spectrum.scan_size_x = 1;
spectrum.scan_size_y = 1;
spectrum.grid_x = 0;
spectrum.grid_y = 0;
spectrum.secondary_axis_kind = '';
spectrum.secondary_axis_unit = '';
spectrum.secondary_axis_value = [];
spectrum.measurement_time = '';
spectrum.laser_wavelength = '';
spectrum.laser_power = '';
spectrum.integration_time = '';
spectrum.grating = '';
spectrum.objective = '';
end


function kind = infer_spectrum_type_from_axis_local(x_axis, x_axis_unit)
unit_name = lower(strtrim(char(x_axis_unit)));
minimum = min(x_axis);
maximum = max(x_axis);
span = maximum - minimum;

if contains(unit_name, 'cm^-1') || contains(unit_name, 'cm-1') || contains(unit_name, 'raman')
    kind = 'Raman';
elseif contains(unit_name, 'nm') || contains(unit_name, 'nanometer') || contains(unit_name, 'ev')
    kind = 'PL';
elseif minimum >= 0 && maximum <= 10
    kind = 'PL';
elseif minimum >= 300 && maximum <= 1200
    kind = 'PL';
elseif minimum >= 0 && maximum <= 4500 && (minimum < 250 || maximum > 1400 || span > 1000)
    kind = 'Raman';
else
    kind = 'unknown';
end
end


function unit_name = guess_x_axis_unit(x_axis)
minimum = min(x_axis);
maximum = max(x_axis);
if minimum >= 0 && maximum <= 5
    unit_name = 'eV';
elseif minimum >= 300 && maximum <= 1200
    unit_name = 'nm';
elseif minimum >= 0 && maximum <= 5000
    unit_name = 'cm^-1';
else
    unit_name = 'unknown';
end
end
