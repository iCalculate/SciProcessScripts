function toolbox_root = ensure_witio_ready(options, config_path)
%ENSURE_WITIO_READY Make the external wit_io / WITio toolbox available headlessly.

if nargin < 2
    config_path = '';
end
if nargin < 1 || isempty(options)
    options = read_pldb_config(config_path);
end

toolbox_root = '';
if isfield(options, 'matlab') && isfield(options.matlab, 'witio_root')
    toolbox_root = strtrim(char(options.matlab.witio_root));
end

if ~isempty(toolbox_root)
    toolbox_root = resolve_witio_root(toolbox_root, config_path);
    if ~isfolder(toolbox_root)
        error('PLDB:WITioPathMissing', ...
            'Configured matlab.witio_root does not exist: %s', toolbox_root);
    end
    addpath(toolbox_root, '-begin');

    third_party_root = fullfile(toolbox_root, 'third party');
    if isfolder(third_party_root)
        addpath(genpath(third_party_root), '-begin');
    end
end

if isempty(which('WITio.read'))
    error('PLDB:WITioMissing', [ ...
        'WITio is not available on the MATLAB path. ' ...
        'Install or clone https://gitlab.com/jtholmi/wit_io and set matlab.witio_root in config.yaml.' ...
    ]);
end
end


function resolved_path = resolve_witio_root(toolbox_root, config_path)
resolved_path = toolbox_root;

if isfolder(resolved_path)
    return;
end

if ~isempty(config_path)
    config_root = fileparts(config_path);
    candidate = fullfile(config_root, toolbox_root);
    if isfolder(candidate)
        resolved_path = candidate;
        return;
    end
end

resolved_path = char(java.io.File(toolbox_root).getAbsolutePath());
end
