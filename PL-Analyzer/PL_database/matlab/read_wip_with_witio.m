function [O_wid, O_wip, O_wit] = read_wip_with_witio(wip_file, config_path)
%READ_WIP_WITH_WITIO Read a WITec project headlessly through WITio / wit_io.

if nargin < 2
    config_path = '';
end

options = read_pldb_config(config_path);
ensure_witio_ready(options, config_path);

try
    [O_wid, O_wip, O_wit] = WITio.read(wip_file, '-all');
catch ME
    error('PLDB:WITioReadFailed', ...
        'Unable to read WIP file with WITio in headless mode.\n%s', ...
        getReport(ME, 'extended', 'hyperlinks', 'off'));
end
end
