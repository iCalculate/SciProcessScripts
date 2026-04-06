function areaData = import_capacitor_area_csv(csvFile)
% IMPORT_CAPACITOR_AREA_CSV  Load capacitor area data from a text/CSV file.
%
% Expected layout:
%   - Column 1: device label / identifier
%   - Column 2: area in um^2
%
% Matching rule for fitting:
%   The identifier in column 1 is preserved for display only.
%   Actual matching is by row order after the selected C-V files are sorted
%   by device index parsed from the file name.

if ~isfile(csvFile)
    error('Area CSV file not found: %s', csvFile);
end

rawText = fileread(csvFile);
rawText = strrep(rawText, sprintf('\r\n'), sprintf('\n'));
rawText = strrep(rawText, sprintf('\r'), sprintf('\n'));
lines = splitlines(rawText);

labels = strings(0, 1);
areas = [];

for i = 1:numel(lines)
    line = strtrim(lines{i});
    if line == ""
        continue;
    end

    parts = regexp(line, '[,\t;]+', 'split');
    parts = parts(~cellfun(@isempty, parts));
    if numel(parts) < 2
        continue;
    end

    areaVal = str2double(strtrim(parts{2}));
    if ~isfinite(areaVal)
        continue;
    end

    labels(end+1, 1) = string(strtrim(parts{1})); %#ok<AGROW>
    areas(end+1, 1) = areaVal; %#ok<AGROW>
end

if isempty(areas)
    error('No valid area values found in the second column of the CSV/text file.');
end

areaData = struct();
areaData.label = labels;
areaData.areaUm2 = areas;
end
