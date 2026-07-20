function report = dump_wip_numeric_tree(wip_file, output_file, config_path)
%DUMP_WIP_NUMERIC_TREE Inspect numeric arrays inside a WITec DataTree.
%   report = dump_wip_numeric_tree(wip_file)
%   report = dump_wip_numeric_tree(wip_file, output_file)
%   report = dump_wip_numeric_tree(wip_file, output_file, config_path)

if nargin < 2
    output_file = '';
end
if nargin < 3
    config_path = '';
end

[~, ~, O_wit] = read_wip_with_witio(wip_file, config_path);
S_DT = WITio.obj.wit.DataTree_get(O_wit);
report = scan_numeric_node(S_DT, 'S_DT', empty_report());

lines = cell(numel(report), 1);
for idx = 1:numel(report)
    lines{idx} = sprintf('%s    size=%s    class=%s    %s', ...
        report(idx).path, report(idx).size_text, report(idx).class_name, report(idx).shape_kind);
    disp(lines{idx});
end

if ~isempty(output_file)
    fid = fopen(output_file, 'w');
    cleanup = onCleanup(@() fclose(fid));
    for idx = 1:numel(lines)
        fprintf(fid, '%s\n', lines{idx});
    end
end
end


function report = empty_report()
report = struct( ...
    'path', {}, ...
    'size_text', {}, ...
    'class_name', {}, ...
    'shape_kind', {});
end


function report = scan_numeric_node(node, path_text, report)
if isnumeric(node) || islogical(node)
    if ~isempty(node)
        entry.path = path_text;
        entry.size_text = size_to_text(size(node));
        entry.class_name = class(node);
        entry.shape_kind = classify_numeric_array(node);
        report(end + 1) = entry; %#ok<AGROW>
    end
    return;
end

if isstruct(node)
    if numel(node) > 1
        for idx = 1:numel(node)
            report = scan_numeric_node(node(idx), sprintf('%s(%d)', path_text, idx), report);
        end
        return;
    end
    fields = fieldnames(node);
    for idx = 1:numel(fields)
        child_name = fields{idx};
        report = scan_numeric_node(node.(child_name), [path_text '.' child_name], report);
    end
    return;
end

if iscell(node)
    for idx = 1:numel(node)
        child_path = sprintf('%s{%d}', path_text, idx);
        report = scan_numeric_node(node{idx}, child_path, report);
    end
    return;
end

if isobject(node)
    if numel(node) > 1
        for idx = 1:numel(node)
            report = scan_numeric_node(node(idx), sprintf('%s(%d)', path_text, idx), report);
        end
        return;
    end
    props = properties(node);
    for idx = 1:numel(props)
        prop_name = props{idx};
        try
            child = node.(prop_name);
        catch
            continue;
        end
        report = scan_numeric_node(child, [path_text '.' prop_name], report);
    end
end
end


function text_value = size_to_text(sz)
parts = arrayfun(@num2str, sz, 'UniformOutput', false);
text_value = ['[' strjoin(parts, ',') ']'];
end


function kind = classify_numeric_array(array_value)
sz = size(array_value);
if isvector(array_value)
    kind = 'vector';
elseif numel(sz) == 2
    if sz(1) <= 4 || sz(2) <= 4
        kind = 'matrix-like';
    else
        kind = 'image-like';
    end
elseif numel(sz) == 3
    if sz(3) <= 4
        kind = 'rgb-or-stack';
    else
        kind = 'cube-like';
    end
else
    kind = 'high-dimensional';
end
end
