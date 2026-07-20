function spectra = extract_1d_spectra_from_wid(O_wid, source_wip, options, root_data_tree)
%EXTRACT_1D_SPECTRA_FROM_WID Extract supported 1D traces from WITec wid objects.

if nargin < 3 || isempty(options)
    options = default_options();
else
    options = merge_default_options(options, default_options());
end

spectra = empty_spectra();

if nargin >= 4 && isstruct(root_data_tree)
    spectra = collect_raw_tree_spectra(O_wid, root_data_tree, source_wip, options);
    if ~isempty(spectra)
        return;
    end
end

spectra = collect_spectra(O_wid, source_wip, options, true);
if isempty(spectra)
    spectra = collect_spectra(O_wid, source_wip, options, false);
end
end


function spectra = collect_spectra(O_wid, source_wip, options, require_point_subtype)
spectra = empty_spectra();
for idx = 1:numel(O_wid)
    spectrum = extract_single_spectrum(O_wid(idx), idx, source_wip, options, require_point_subtype);
    if ~isempty(spectrum)
        spectra(end + 1) = spectrum; %#ok<AGROW>
    end
end
end


function spectra = collect_raw_tree_spectra(O_wid, root_data_tree, source_wip, options)
spectra = empty_spectra();
data_entries = collect_root_data_entries(root_data_tree);
for idx = 1:numel(O_wid)
    extracted = extract_raw_tree_graph_spectra(O_wid(idx), idx, data_entries, source_wip, options);
    if ~isempty(extracted)
        spectra = [spectra extracted]; %#ok<AGROW>
    end
end
end


function spectrum = extract_single_spectrum(obj, fallback_index, source_wip, options, require_point_subtype)
spectrum = struct([]);

if ~strcmp(obj.Type, 'TDGraph')
    return;
end

name_text = char(obj.Name);
subtype_text = char(obj.SubType);
if should_skip_name(lower(strtrim([name_text ' ' subtype_text])), options.skip_keywords)
    return;
end

if ~options.include_point_spectra
    return;
end

try
    data = double(obj.Data);
    dims = padded_size(data);
    if dims(3) < 16
        return;
    end
    if dims(1) * dims(2) * dims(4) ~= 1
        return;
    end

    if require_point_subtype
        if ~strcmpi(subtype_text, 'Point')
            return;
        end
    elseif ~is_scalar_spectrum_name(name_text, subtype_text, options.spectrum_keywords)
        return;
    end

    info = obj.Info;
    x_axis = double(info.Graph(:));
    intensity = reshape(data, [], 1);
    if numel(x_axis) ~= numel(intensity)
        return;
    end

    if isempty(name_text)
        name_text = sprintf('TDGraph_%03d', fallback_index);
    end

    ordinal_value = double(obj.OrdinalNumber);
    if isempty(ordinal_value) || ~isfinite(ordinal_value)
        ordinal_value = fallback_index;
    end

    x_axis_unit = resolve_x_axis_unit(info.GraphUnit, x_axis);
    source_tree_path = sprintf('/WITioFallback/point_spectrum/%03d/%s', ...
        round(ordinal_value), ...
        make_safe_name(name_text));

    spectrum = build_spectrum_struct( ...
        x_axis, ...
        intensity, ...
        source_tree_path, ...
        x_axis_unit, ...
        source_wip, ...
        'point_spectrum', ...
        0, ...
        1, ...
        1, ...
        1, ...
        0, ...
        0, ...
        '', ...
        '', ...
        []);
catch
    spectrum = struct([]);
end
end


function spectra = extract_raw_tree_graph_spectra(obj, fallback_index, data_entries, source_wip, options)
spectra = empty_spectra();

if ~strcmp(obj.Type, 'TDGraph')
    return;
end

name_text = char(obj.Name);
subtype_text = char(obj.SubType);
combined_name = lower(strtrim([name_text ' ' subtype_text]));
if should_skip_graph_candidate(combined_name, options)
    return;
end

try
    tree = obj.DataTree;
    if ~isfield(tree, 'TDGraph')
        return;
    end

    graph_tree = tree.TDGraph;
    size_x = double(graph_tree.SizeX);
    size_y = double(graph_tree.SizeY);
    size_graph = double(graph_tree.SizeGraph);
    if size_graph < 16 || size_x < 1 || size_y < 1
        return;
    end

    decoded_cube = decode_graph_data_cube(graph_tree);
    [x_axis, x_axis_unit] = decode_graph_axis(graph_tree, data_entries, size_graph);
    if isempty(x_axis)
        x_axis = (1:size_graph).';
        x_axis_unit = 'pixel';
    end
    if numel(x_axis) ~= size_graph
        return;
    end

    if isempty(name_text)
        name_text = sprintf('TDGraph_%03d', fallback_index);
    end

    ordinal_value = double(obj.OrdinalNumber);
    if isempty(ordinal_value) || ~isfinite(ordinal_value)
        ordinal_value = fallback_index;
    end

    [acquisition_mode, secondary_axis_kind, secondary_axis_unit] = classify_graph_mode( ...
        name_text, subtype_text, size_x, size_y);
    if ~should_include_mode(acquisition_mode, options)
        return;
    end

    trace_count = size_x * size_y;
    laser_wavelength = extract_laser_wavelength(graph_tree, data_entries);
    safe_name = make_safe_name(name_text);

    for grid_y = 1:size_y
        for grid_x = 1:size_x
            trace_index = (grid_y - 1) * size_x + (grid_x - 1);
            intensity = reshape(decoded_cube(grid_x, grid_y, :), [], 1);
            if numel(intensity) ~= numel(x_axis)
                continue;
            end

            secondary_axis_value = resolve_secondary_axis_value(acquisition_mode, trace_index, grid_x, grid_y);
            source_tree_path = sprintf('/WITioRaw/%s/%03d/%s/trace-%04d', ...
                acquisition_mode, ...
                round(ordinal_value), ...
                safe_name, ...
                trace_index);

            spectra(end + 1) = build_spectrum_struct( ... %#ok<AGROW>
                x_axis, ...
                intensity, ...
                source_tree_path, ...
                x_axis_unit, ...
                source_wip, ...
                acquisition_mode, ...
                trace_index, ...
                trace_count, ...
                size_x, ...
                size_y, ...
                grid_x - 1, ...
                grid_y - 1, ...
                secondary_axis_kind, ...
                secondary_axis_unit, ...
                secondary_axis_value);

            if ~isempty(laser_wavelength)
                spectra(end).laser_wavelength = laser_wavelength;
            end
        end
    end
catch
    spectra = empty_spectra();
end
end


function spectrum = build_spectrum_struct( ...
    x_axis, intensity, source_tree_path, x_axis_unit, source_wip, acquisition_mode, ...
    trace_index, trace_count, scan_size_x, scan_size_y, grid_x, grid_y, ...
    secondary_axis_kind, secondary_axis_unit, secondary_axis_value)

spectrum = struct( ...
    'x_axis', double(x_axis(:)), ...
    'intensity', double(intensity(:)), ...
    'source_tree_path', source_tree_path, ...
    'spectrum_type', infer_spectrum_type_from_axis_local(x_axis, x_axis_unit), ...
    'x_axis_unit', char(x_axis_unit), ...
    'n_points', numel(x_axis), ...
    'source_wip', source_wip, ...
    'acquisition_mode', acquisition_mode, ...
    'trace_index', trace_index, ...
    'trace_count', trace_count, ...
    'scan_size_x', scan_size_x, ...
    'scan_size_y', scan_size_y, ...
    'grid_x', grid_x, ...
    'grid_y', grid_y, ...
    'secondary_axis_kind', secondary_axis_kind, ...
    'secondary_axis_unit', secondary_axis_unit, ...
    'secondary_axis_value', secondary_axis_value, ...
    'measurement_time', '', ...
    'laser_wavelength', '', ...
    'laser_power', '', ...
    'integration_time', '', ...
    'grating', '', ...
    'objective', '');
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


function data_entries = collect_root_data_entries(root_data_tree)
data_entries = struct( ...
    'id', {}, ...
    'payload', {});

if ~isfield(root_data_tree, 'Data') || ~isstruct(root_data_tree.Data)
    return;
end

entry_names = fieldnames(root_data_tree.Data);
for idx = 1:numel(entry_names)
    entry = root_data_tree.Data.(entry_names{idx});
    if isstruct(entry) && isfield(entry, 'TData') && isfield(entry.TData, 'ID')
        data_entries(end + 1).id = double(entry.TData.ID); %#ok<AGROW>
        data_entries(end).payload = entry;
    end
end
end


function cube = decode_graph_data_cube(graph_tree)
size_x = double(graph_tree.SizeX);
size_y = double(graph_tree.SizeY);
size_graph = double(graph_tree.SizeGraph);
decoded = decode_data_type_bytes(uint8(graph_tree.GraphData.Data(:)), double(graph_tree.GraphData.DataType));
if isfield(graph_tree, 'DataFieldInverted') && logical(graph_tree.DataFieldInverted)
    cube = permute(reshape(decoded, [size_graph, size_x, size_y]), [2, 3, 1]);
else
    cube = permute(reshape(decoded, [size_graph, size_y, size_x]), [3, 2, 1]);
end
end


function values = decode_data_type_bytes(raw_bytes, data_type)
supported_classes = {'int64', 'int32', 'int16', 'int8', 'uint32', 'uint16', 'uint8', 'logical', 'single', 'double'};
if data_type < 1 || data_type > numel(supported_classes)
    error('Unsupported raw DataType: %d', data_type);
end
if data_type == 8
    values = cast(raw_bytes, 'logical');
else
    values = typecast(raw_bytes(:).', supported_classes{data_type});
end
values = values(:);
end


function [x_axis, unit_name] = decode_graph_axis(graph_tree, data_entries, size_graph)
x_axis = [];
unit_name = 'unknown';

transform_entry = lookup_data_entry(data_entries, double(graph_tree.XTransformationID));
if isempty(transform_entry)
    return;
end

raw_axis = apply_raw_transformation(transform_entry, (1:size_graph + 1).');
if isempty(raw_axis)
    return;
end
if size(raw_axis, 3) == 3
    raw_axis = sqrt(sum((raw_axis(:, :, :) - repmat(raw_axis(1, :, :), [size(raw_axis, 1), 1, 1])).^2, 3));
end
if numel(raw_axis) < size_graph + 1
    return;
end

raw_axis = raw_axis(1:end - 1);
unit_name = infer_transform_output_unit(transform_entry);

interpretation_entry = lookup_data_entry(data_entries, double(graph_tree.XInterpretationID));
if ~isempty(interpretation_entry)
    [unit_name, raw_axis] = apply_raw_interpretation(interpretation_entry, unit_name, raw_axis);
elseif isfield(transform_entry, 'TDTransformation') && isfield(transform_entry.TDTransformation, 'InterpretationID')
    transform_interp = lookup_data_entry(data_entries, double(transform_entry.TDTransformation.InterpretationID));
    if ~isempty(transform_interp)
        [unit_name, raw_axis] = apply_raw_interpretation(transform_interp, unit_name, raw_axis);
    end
end

x_axis = double(raw_axis(:));
end


function entry = lookup_data_entry(data_entries, entry_id)
entry = struct([]);
for idx = 1:numel(data_entries)
    if data_entries(idx).id == entry_id
        entry = data_entries(idx).payload;
        return;
    end
end
end


function unit_name = infer_transform_output_unit(transform_entry)
if isfield(transform_entry, 'TDSpectralTransformation')
    unit_name = WITio.obj.wip.DefaultSpectralUnit;
elseif isfield(transform_entry, 'TDSpaceTransformation')
    unit_name = WITio.obj.wip.DefaultSpaceUnit;
elseif isfield(transform_entry, 'TDLinearTransformation')
    unit_name = WITio.obj.wip.ArbitraryUnit;
elseif isfield(transform_entry, 'TDLUTTransformation')
    unit_name = WITio.obj.wip.ArbitraryUnit;
else
    unit_name = 'unknown';
end
end


function values = apply_raw_transformation(transform_entry, pixel_values)
values = double(pixel_values);

if isfield(transform_entry, 'TDLinearTransformation')
    t_linear = transform_entry.TDLinearTransformation;
    values = t_linear.Scale_D .* (values - 1 - t_linear.ModelOrigin_D) + t_linear.WorldOrigin_D;
    return;
end

if isfield(transform_entry, 'TDSpaceTransformation')
    t_space = transform_entry.TDSpaceTransformation;
    model_origin = t_space.ViewPort3D.ModelOrigin(:);
    world_origin = t_space.ViewPort3D.WorldOrigin(:);
    scale = reshape(t_space.ViewPort3D.Scale, [3, 3]);
    rotation = reshape(t_space.ViewPort3D.Rotation, [3, 3]);
    original_size = size(values);
    values = permute(values, [3, 1, 2]);
    values(size(values, 3) + 1:3, :, :) = 1;
    values = bsxfun(@minus, values(:,:) - 1, model_origin);
    values = reshape(rotation * scale * values, [3, size(values, 2)]);
    values = bsxfun(@plus, values, world_origin);
    values = ipermute(reshape(values, [3, original_size(1), max(1, size(values, 2) / original_size(1))]), [3, 1, 2]);
    return;
end

if isfield(transform_entry, 'TDSpectralTransformation')
    values = apply_raw_spectral_transformation(transform_entry.TDSpectralTransformation, values);
    return;
end

if isfield(transform_entry, 'TDLUTTransformation')
    t_lut = transform_entry.TDLUTTransformation;
    lut = double(t_lut.LUT);
    lut_size = min(double(t_lut.LUTSize), numel(lut));
    if lut_size < 1
        values = nan(size(values));
    elseif lut_size == 1
        values = lut(1) .* ones(size(values));
    else
        below = values < 1;
        above = values > lut_size;
        values = interp1(double(1:lut_size), lut, values, 'linear');
        values(below) = lut(1);
        values(above) = lut(end);
    end
    return;
end

values = [];
end


function values = apply_raw_spectral_transformation(t_spectral, pixel_values)
values = double(pixel_values);
transform_type = double(t_spectral.SpectralTransformationType);

if transform_type == 0
    polynom = t_spectral.Polynom;
    x = values - 1;
    n_terms = min(numel(polynom), 3);
    values = zeros(size(x));
    for idx = 1:n_terms
        values = values + polynom(idx) .* x.^(idx - 1);
    end
    return;
end

if transform_type == 1
    n_c = t_spectral.nC;
    lambda_c = t_spectral.LambdaC;
    gamma = t_spectral.Gamma;
    delta = t_spectral.Delta;
    order_m = t_spectral.m;
    groove_d = t_spectral.d;
    pixel_width = t_spectral.x;
    focal_length = t_spectral.f;

    alpha = asin(lambda_c .* order_m ./ groove_d ./ (2 .* cos(gamma ./ 2))) - gamma ./ 2;
    l_h = focal_length .* cos(delta);
    h_b_lambda_c = focal_length .* sin(delta);
    h_b_lambda_n = pixel_width .* (n_c - (values - 1)) - h_b_lambda_c;
    beta_lambda_c = gamma + alpha;
    beta_h = beta_lambda_c - delta;
    beta_lambda_n = beta_h - atan2(h_b_lambda_n, l_h);
    values = groove_d ./ order_m .* (sin(alpha) + sin(beta_lambda_n));
    return;
end

if transform_type == 2
    free_polynom = t_spectral.FreePolynom;
    free_order = min(double(t_spectral.FreePolynomOrder), numel(free_polynom) - 1);
    x = values - 1;
    transformed = zeros(size(x));

    x_start = t_spectral.FreePolynomStartBin;
    x_stop = max(x_start, t_spectral.FreePolynomStopBin);
    value_start = 0;
    value_stop = 0;
    for idx = 1:free_order + 1
        value_start = value_start + free_polynom(idx) .* x_start.^(idx - 1);
        value_stop = value_stop + free_polynom(idx) .* x_stop.^(idx - 1);
    end

    below = x <= x_start;
    above = x >= x_stop;
    inside = ~below & ~above;
    transformed(below) = value_start;
    transformed(above) = value_stop;
    for idx = 1:free_order + 1
        transformed(inside) = transformed(inside) + free_polynom(idx) .* x(inside).^(idx - 1);
    end
    values = transformed;
    return;
end

values = [];
end


function [unit_name, values] = apply_raw_interpretation(interpretation_entry, unit_name, values)
descriptor = build_raw_interpretation_descriptor(interpretation_entry);
if isempty(descriptor)
    return;
end

[unit_name, values] = WITio.obj.wip.interpret(descriptor, [], unit_name, values);
end


function descriptor = build_raw_interpretation_descriptor(interpretation_entry)
descriptor = [];

if isempty(interpretation_entry) || ~isstruct(interpretation_entry)
    return;
end

if isfield(interpretation_entry, 'TDSpectralInterpretation')
    excitation = nan;
    if isfield(interpretation_entry.TDSpectralInterpretation, 'ExcitationWaveLength')
        excitation = double(interpretation_entry.TDSpectralInterpretation.ExcitationWaveLength);
    end
    descriptor = {'TDSpectralInterpretation', excitation};
    return;
end

if isfield(interpretation_entry, 'TDZInterpretation')
    z_unit = nan;
    if isfield(interpretation_entry.TDZInterpretation, 'UnitName') && ~isempty(interpretation_entry.TDZInterpretation.UnitName)
        z_unit = char(interpretation_entry.TDZInterpretation.UnitName);
    end
    descriptor = {'TDZInterpretation', z_unit};
    return;
end

raw_types = { ...
    'TDSpaceInterpretation', ...
    'TDTimeInterpretation', ...
    'TDFrequencyInterpretation', ...
    'TDInverseSpaceInterpretation', ...
    'TDPhaseInterpretation'};

for idx = 1:numel(raw_types)
    if isfield(interpretation_entry, raw_types{idx})
        descriptor = raw_types{idx};
        return;
    end
end
end


function value = extract_laser_wavelength(graph_tree, data_entries)
value = '';

interpretation_entry = lookup_data_entry(data_entries, double(graph_tree.XInterpretationID));
if ~isempty(interpretation_entry) && isfield(interpretation_entry, 'TDSpectralInterpretation')
    if isfield(interpretation_entry.TDSpectralInterpretation, 'ExcitationWaveLength')
        candidate = double(interpretation_entry.TDSpectralInterpretation.ExcitationWaveLength);
        if isfinite(candidate)
            value = num2str(candidate);
            return;
        end
    end
end

transform_entry = lookup_data_entry(data_entries, double(graph_tree.XTransformationID));
if ~isempty(transform_entry) && isfield(transform_entry, 'TDSpectralTransformation')
    if isfield(transform_entry.TDSpectralTransformation, 'CertifiedExcitationWavelength')
        candidate = double(transform_entry.TDSpectralTransformation.CertifiedExcitationWavelength);
        if isfinite(candidate)
            value = num2str(candidate);
        end
    end
end
end


function dims = padded_size(value)
dims = size(value);
while numel(dims) < 4
    dims(end + 1) = 1; %#ok<AGROW>
end
end


function tf = should_skip_name(name_text, skip_keywords)
tf = false;
if contains(name_text, 'mask')
    tf = true;
    return;
end
for idx = 1:numel(skip_keywords)
    if contains(name_text, lower(skip_keywords{idx}))
        tf = true;
        return;
    end
end
end


function tf = should_skip_graph_candidate(name_text, options)
tf = false;

if contains(name_text, 'topography') || contains(name_text, 'video') || contains(name_text, 'camera') || contains(name_text, 'hyperspectral')
    tf = true;
    return;
end

if (contains(name_text, 'image') || contains(name_text, 'map') || contains(name_text, 'mapping')) && options.include_area_maps
    tf = false;
    return;
end

if (contains(name_text, 'time') || contains(name_text, 'power') || contains(name_text, 'series')) && options.include_series_scans
    tf = false;
    return;
end

tf = should_skip_name(name_text, options.skip_keywords);
end


function tf = is_scalar_spectrum_name(name_text, subtype_text, spectrum_keywords)
combined = lower(strtrim([name_text ' ' subtype_text]));
if strcmpi(subtype_text, 'Point')
    tf = true;
    return;
end

tf = false;
for idx = 1:numel(spectrum_keywords)
    if contains(combined, lower(spectrum_keywords{idx}))
        tf = true;
        return;
    end
end
end


function [mode_name, secondary_axis_kind, secondary_axis_unit] = classify_graph_mode(name_text, subtype_text, size_x, size_y)
combined = lower(strtrim([name_text ' ' subtype_text]));
secondary_axis_kind = '';
secondary_axis_unit = '';

if strcmpi(subtype_text, 'Point') || (size_x == 1 && size_y == 1)
    mode_name = 'point_spectrum';
    return;
end

if contains(combined, 'time')
    mode_name = 'series_scan';
    secondary_axis_kind = 'time';
    secondary_axis_unit = 'index';
    return;
end

if contains(combined, 'power')
    mode_name = 'series_scan';
    secondary_axis_kind = 'power';
    secondary_axis_unit = 'index';
    return;
end

if contains(combined, 'series')
    mode_name = 'series_scan';
    secondary_axis_kind = 'series';
    secondary_axis_unit = 'index';
    return;
end

if strcmpi(subtype_text, 'Image') || (size_x > 1 && size_y > 1)
    mode_name = 'area_map';
    secondary_axis_kind = 'xy_position';
    secondary_axis_unit = 'pixel';
    return;
end

mode_name = 'line_scan';
secondary_axis_kind = 'distance';
secondary_axis_unit = 'pixel';
end


function tf = should_include_mode(mode_name, options)
switch char(mode_name)
    case 'point_spectrum'
        tf = logical(options.include_point_spectra);
    case 'line_scan'
        tf = logical(options.include_line_scans);
    case 'area_map'
        tf = logical(options.include_area_maps);
    case 'series_scan'
        tf = logical(options.include_series_scans);
    otherwise
        tf = false;
end
end


function value = resolve_secondary_axis_value(acquisition_mode, trace_index, grid_x, grid_y)
switch char(acquisition_mode)
    case {'line_scan', 'series_scan'}
        value = trace_index;
    case 'area_map'
        value = [grid_x - 1, grid_y - 1];
    otherwise
        value = [];
end
end


function safe_name = make_safe_name(raw_name)
safe_name = regexprep(char(raw_name), '[^a-zA-Z0-9_\-]+', '_');
if isempty(safe_name)
    safe_name = 'wip_export';
end
end


function spectrum_type = infer_spectrum_type_from_axis_local(x_axis, x_axis_unit)
unit_name = lower(strtrim(char(x_axis_unit)));
values = double(x_axis(:));
if isempty(values)
    spectrum_type = 'unknown';
    return;
end

minimum = min(values);
maximum = max(values);
span = maximum - minimum;

if contains(unit_name, 'cm^-1') || contains(unit_name, 'cm-1') || contains(unit_name, 'raman')
    spectrum_type = 'Raman';
elseif contains(unit_name, 'nm') || contains(unit_name, 'nanometer') || contains(unit_name, 'ev')
    spectrum_type = 'PL';
elseif minimum >= 0 && maximum <= 10
    spectrum_type = 'PL';
elseif minimum >= 300 && maximum <= 1200
    spectrum_type = 'PL';
elseif minimum >= 0 && maximum <= 4500 && (minimum < 250 || maximum > 1400 || span > 1000)
    spectrum_type = 'Raman';
else
    spectrum_type = 'unknown';
end
end


function unit_name = resolve_x_axis_unit(graph_unit, x_axis)
unit_name = strtrim(char(graph_unit));
if ~isempty(unit_name)
    return;
end

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
