function datasets = import_capacitor_measurements(filePaths)
% IMPORT_CAPACITOR_MEASUREMENTS  Import capacitor C-V and breakdown I-V CSV files.
%
% datasets = import_capacitor_measurements(filePaths)
%
% Input:
%   filePaths : char/string/cellstr of one or more CSV files.
%
% Output:
%   datasets(k) with a lightweight layout:
%       .meta
%           .filePath
%           .fileName
%           .sampleName
%           .deviceTag
%           .deviceIndex
%           .setupTitle
%           .primitiveTest
%           .testType
%           .iterationIndex
%           .recordTime
%       .config
%           .sweepVar
%           .sweepStart
%           .sweepStop
%           .sweepStep
%           .sweepCount
%           .frequenciesHz
%           .acLevelV
%           .impedanceModel
%           .complianceA
%       .raw
%           .xName
%           .xUnit
%           .x
%           .cp
%           .g
%           .yNames
%           .yUnits
%           .y
%           .table

    if ischar(filePaths) || (isstring(filePaths) && isscalar(filePaths))
        filePaths = {char(filePaths)};
    elseif isstring(filePaths)
        filePaths = cellstr(filePaths);
    end

    if ~iscell(filePaths) || isempty(filePaths)
        error('filePaths must contain at least one CSV path.');
    end

    datasets = repmat(empty_dataset(), 1, numel(filePaths));
    for k = 1:numel(filePaths)
        datasets(k) = import_one_file(filePaths{k});
    end
end


function ds = import_one_file(filePath)
    if ~isfile(filePath)
        error('File not found: %s', filePath);
    end

    rawText = fileread(filePath);
    rawText = strrep(rawText, sprintf('\r\n'), sprintf('\n'));
    rawText = strrep(rawText, sprintf('\r'), sprintf('\n'));
    lines = splitlines(rawText);

    info = parse_header(lines);
    dataTable = parse_data_table(lines, info.dataNames);

    testType = classify_test(info.setupTitle, info.primitiveTest, filePath);
    [sampleName, deviceTag, deviceIndex] = parse_file_label(filePath);
    [rawStruct, configStruct] = build_payloads(info, dataTable, testType);

    ds = empty_dataset();
    [~, baseName, ext] = fileparts(filePath);
    ds.meta.filePath = char(filePath);
    ds.meta.fileName = [baseName, ext];
    ds.meta.sampleName = sampleName;
    ds.meta.deviceTag = deviceTag;
    ds.meta.deviceIndex = deviceIndex;
    ds.meta.setupTitle = info.setupTitle;
    ds.meta.primitiveTest = info.primitiveTest;
    ds.meta.testType = testType;
    ds.meta.iterationIndex = get_numeric_meta(info.metaData, 'TestRecord_IterationIndex', NaN);
    ds.meta.recordTime = char(string(get_test_param(info.metaData, 'TestRecord_RecordTime', "")));

    ds.config = configStruct;
    ds.raw = rawStruct;
end


function info = parse_header(lines)
    info.setupTitle = "";
    info.primitiveTest = "";
    info.testParameter = struct();
    info.metaData = struct();
    info.dim1 = [];
    info.dim2 = [];
    info.dataNames = strings(0, 1);

    for i = 1:numel(lines)
        line = strtrim(lines{i});
        if line == ""
            continue;
        end

        parts = split(line, ',');
        parts = strip(parts);
        tag = string(parts{1});

        switch tag
            case "SetupTitle"
                if numel(parts) >= 2
                    info.setupTitle = string(strjoin(parts(2:end), ", "));
                end
            case "PrimitiveTest"
                if numel(parts) >= 2
                    info.primitiveTest = string(strjoin(parts(2:end), ", "));
                end
            case "TestParameter"
                if numel(parts) >= 3
                    key = make_valid_field(parts{2});
                    value = parse_values(parts(3:end));
                    info.testParameter.(key) = value;
                end
            case "Dimension1"
                info.dim1 = str2double(parts(2:end));
            case "Dimension2"
                info.dim2 = str2double(parts(2:end));
            case "MetaData"
                if numel(parts) >= 3
                    key = make_valid_field(parts{2});
                    value = parse_values(parts(3:end));
                    info.metaData.(key) = value;
                end
            case "DataName"
                info.dataNames = string(parts(2:end));
                return;
        end
    end

    error('DataName header not found. Unsupported CSV format.');
end


function T = parse_data_table(lines, dataNames)
    rows = [];
    startCollect = false;
    for i = 1:numel(lines)
        line = strtrim(lines{i});
        if startsWith(line, "DataName")
            startCollect = true;
            continue;
        end

        if ~startCollect || ~startsWith(line, "DataValue")
            continue;
        end

        parts = split(line, ',');
        numericRow = str2double(strip(parts(2:end)));
        rows = [rows; numericRow.']; %#ok<AGROW>
    end

    if isempty(rows)
        error('No DataValue rows found in CSV.');
    end

    T = array2table(rows, 'VariableNames', matlab.lang.makeValidName(cellstr(dataNames)));
end


function testType = classify_test(setupTitle, primitiveTest, filePath)
    textBlob = lower(strjoin([string(setupTitle), string(primitiveTest), string(filePath)], " "));
    if contains(textBlob, "cpg-v") || contains(textBlob, "c-v")
        testType = 'cv';
    elseif contains(textBlob, "breakdown") || contains(textBlob, "i/v")
        testType = 'breakdown_iv';
    else
        testType = 'unknown';
    end
end


function [rawStruct, configStruct] = build_payloads(info, T, testType)
    varNames = string(T.Properties.VariableNames);
    xName = "";
    xUnit = "";
    yNames = {};
    yUnits = {};
    y = [];
    cpMat = [];
    gMat = [];

    switch testType
        case 'cv'
            xName = pick_first(varNames, ["VBias", "VBias_1", "V"]);
            if xName == ""
                xName = varNames(1);
            end

            cpName = pick_first(varNames, ["Cp"]);
            gName = pick_first(varNames, ["G"]);
            x = T.(xName);

            freqs = get_test_param(info.testParameter, 'Measurement_Secondary_Frequency', []);
            freqs = numeric_row(freqs);

            if cpName ~= ""
                cp = T.(cpName);
                if isempty(freqs)
                    cpMat = cp(:);
                    y = cpMat;
                    yNames = {'Cp'};
                else
                    cpMat = reshape_by_frequency(cp, freqs, x);
                    x = reshape_x_by_frequency(x, freqs);
                    y = cpMat;
                    yNames = repmat({'Cp'}, 1, size(cpMat, 2));
                end
            end

            if gName ~= ""
                g = T.(gName);
                if isempty(freqs)
                    gMat = g(:);
                else
                    gMat = reshape_by_frequency(g, freqs, T.(xName));
                end
            else
                gMat = [];
            end
            yUnits = repmat({'F'}, 1, size(y, 2));

            configStruct = struct( ...
                'sweepVar', 'VBias', ...
                'sweepStart', get_test_param(info.testParameter, 'Measurement_Primary_Start', NaN), ...
                'sweepStop', get_test_param(info.testParameter, 'Measurement_Primary_Stop', NaN), ...
                'sweepStep', get_test_param(info.testParameter, 'Measurement_Primary_Step', NaN), ...
                'sweepCount', scalar_or_nan(info.dim1), ...
                'frequenciesHz', freqs, ...
                'acLevelV', get_test_param(info.testParameter, 'Measurement_Secondary_ACLevel', NaN), ...
                'impedanceModel', char(string(get_test_param(info.testParameter, 'Measurement_ImpedanceMode_Model', ""))), ...
                'complianceA', NaN);

            rawStruct = struct( ...
                'xName', char(xName), ...
                'xUnit', 'V', ...
                'x', x(:), ...
                'cp', cpMat, ...
                'g', gMat, ...
                'yNames', {yNames}, ...
                'yUnits', {yUnits}, ...
                'y', y, ...
                'table', T);

        case 'breakdown_iv'
            xName = pick_first(varNames, ["Vt", "V", "Voltage"]);
            if xName == ""
                xName = varNames(1);
            end
            yName = pick_first(varNames, ["absI", "It", "I", "Current"]);
            if yName == ""
                yName = varNames(min(2, numel(varNames)));
            end

            configStruct = struct( ...
                'sweepVar', char(xName), ...
                'sweepStart', get_test_param(info.testParameter, 'Measurement_Primary_Start', NaN), ...
                'sweepStop', get_test_param(info.testParameter, 'Measurement_Primary_Stop', NaN), ...
                'sweepStep', get_test_param(info.testParameter, 'Measurement_Primary_Step', NaN), ...
                'sweepCount', scalar_or_nan(info.dim1), ...
                'frequenciesHz', [], ...
                'acLevelV', NaN, ...
                'impedanceModel', '', ...
                'complianceA', get_test_param(info.testParameter, 'Measurement_Primary_Compliance', NaN));

            rawStruct = struct( ...
                'xName', char(xName), ...
                'xUnit', 'V', ...
                'x', T.(xName), ...
                'cp', [], ...
                'g', [], ...
                'yNames', {{char(yName)}}, ...
                'yUnits', {{'A'}}, ...
                'y', T.(yName), ...
                'table', T);

        otherwise
            xName = varNames(1);
            rawStruct = struct( ...
                'xName', char(xName), ...
                'xUnit', '', ...
                'x', T.(xName), ...
                'cp', [], ...
                'g', [], ...
                'yNames', {cellstr(varNames(2:end))}, ...
                'yUnits', {repmat({''}, 1, max(numel(varNames)-1, 0))}, ...
                'y', T{:, 2:end}, ...
                'table', T);

            configStruct = struct( ...
                'sweepVar', char(xName), ...
                'sweepStart', NaN, ...
                'sweepStop', NaN, ...
                'sweepStep', NaN, ...
                'sweepCount', scalar_or_nan(info.dim1), ...
                'frequenciesHz', [], ...
                'acLevelV', NaN, ...
                'impedanceModel', '', ...
                'complianceA', NaN);
    end
end


function value = parse_values(parts)
    if isempty(parts)
        value = [];
        return;
    end

    parts = strip(string(parts));
    nums = str2double(parts);
    if all(~isnan(nums))
        value = nums(:).';
        return;
    end

    if numel(parts) == 1
        value = char(parts);
    else
        value = cellstr(parts);
    end
end


function fieldName = make_valid_field(rawKey)
    fieldName = matlab.lang.makeValidName(strrep(strtrim(char(rawKey)), '.', '_'));
end


function out = get_test_param(S, fieldName, defaultValue)
    if isfield(S, fieldName)
        out = S.(fieldName);
    else
        out = defaultValue;
    end
end


function out = get_numeric_meta(S, fieldName, defaultValue)
    out = get_test_param(S, fieldName, defaultValue);
    if isnumeric(out)
        out = out(1);
        return;
    end
    tmp = str2double(string(out));
    if isfinite(tmp)
        out = tmp;
    else
        out = defaultValue;
    end
end


function name = pick_first(varNames, candidates)
    name = "";
    lowerVars = lower(varNames);
    for k = 1:numel(candidates)
        idx = find(strcmp(lowerVars, lower(candidates(k))), 1, 'first');
        if ~isempty(idx)
            name = varNames(idx);
            return;
        end
    end
end


function [sampleName, deviceTag, deviceIndex] = parse_file_label(filePath)
    [~, fileName, ext] = fileparts(filePath);
    fullName = [fileName, ext];
    token = regexp(fullName, '\[(.*?)\]', 'tokens', 'once');
    if isempty(token)
        sampleName = fileName;
    else
        sampleName = strtrim(token{1});
    end

    tagToken = regexp(sampleName, '\(([^)]+)\)', 'tokens', 'once');
    if isempty(tagToken)
        deviceTag = '';
        deviceIndex = NaN;
    else
        deviceTag = strtrim(tagToken{1});
        deviceIndex = str2double(deviceTag);
        if isnan(deviceIndex)
            numToken = regexp(deviceTag, '\d+', 'match', 'once');
            if isempty(numToken)
                deviceIndex = NaN;
            else
                deviceIndex = str2double(numToken);
            end
        end
    end
end


function yMat = reshape_by_frequency(yVec, freqs, xVec)
    nFreq = max(numel(freqs), 1);
    if mod(numel(yVec), nFreq) ~= 0
        yMat = yVec(:);
        return;
    end

    nPts = numel(yVec) / nFreq;
    yMat = reshape(yVec, [nPts, nFreq]);

    % Clarius files are grouped by frequency, so align x to the first group length.
    if numel(xVec) ~= numel(yVec)
        return;
    end
end


function xOut = reshape_x_by_frequency(xVec, freqs)
    nFreq = max(numel(freqs), 1);
    if mod(numel(xVec), nFreq) ~= 0
        xOut = xVec(:);
        return;
    end

    nPts = numel(xVec) / nFreq;
    xMat = reshape(xVec, [nPts, nFreq]);
    xOut = xMat(:, 1);
end


function out = numeric_row(value)
    if isnumeric(value)
        out = value(:).';
    else
        out = [];
    end
end


function out = scalar_or_nan(value)
    if isempty(value)
        out = NaN;
        return;
    end
    value = value(~isnan(value));
    if isempty(value)
        out = NaN;
    else
        out = value(1);
    end
end


function ds = empty_dataset()
    ds = struct( ...
        'meta', struct( ...
            'filePath', '', ...
            'fileName', '', ...
            'sampleName', '', ...
            'deviceTag', '', ...
            'deviceIndex', NaN, ...
            'setupTitle', '', ...
            'primitiveTest', '', ...
            'testType', '', ...
            'iterationIndex', NaN, ...
            'recordTime', ''), ...
        'config', struct( ...
            'sweepVar', '', ...
            'sweepStart', NaN, ...
            'sweepStop', NaN, ...
            'sweepStep', NaN, ...
            'sweepCount', NaN, ...
            'frequenciesHz', [], ...
            'acLevelV', NaN, ...
            'impedanceModel', '', ...
            'complianceA', NaN), ...
        'raw', struct( ...
            'xName', '', ...
            'xUnit', '', ...
            'x', [], ...
            'cp', [], ...
            'g', [], ...
            'yNames', {{}}, ...
            'yUnits', {{}}, ...
            'y', [], ...
            'table', table()));
end
