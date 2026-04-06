function result = capacitor_mode_multi_file(defaultDir)
filePaths = capacitor_select_multiple_files(defaultDir, 'Select capacitor measurement CSV file(s)', {'*.csv', 'CSV files (*.csv)'});
datasets = import_capacitor_measurements(filePaths);

fprintf('\nImported %d file(s).\n', numel(datasets));
for k = 1:numel(datasets)
    ds = datasets(k);
    fprintf('[%d] %s | type=%s | sample=%s | points=%d\n', ...
        k, ds.meta.fileName, ds.meta.testType, ds.meta.sampleName, numel(ds.raw.x));
end

plot_capacitor_datasets(datasets);
fprintf('Multi-file plotting completed.\n');
result = datasets;
end


function plot_capacitor_datasets(datasets)
testTypes = arrayfun(@(s) string(s.meta.testType), datasets, 'UniformOutput', true);
cvIdx = find(testTypes == "cv");
ivIdx = find(testTypes == "breakdown_iv");

if ~isempty(cvIdx)
    cvData = datasets(cvIdx);
    deviceIndices = arrayfun(@(s) s.meta.deviceIndex, cvData);
    deviceIndices(isnan(deviceIndices)) = inf;
    [~, order] = sort(deviceIndices);
    cvData = cvData(order);
    colors = turbo(max(numel(cvData), 2));
    refFreqs = cvData(1).config.frequenciesHz;
    if isempty(refFreqs)
        refFreqs = 1;
    end

    figure('Name', 'Capacitor C-V Preview By Frequency');
    set(gcf, 'Units', 'pixels', 'Position', [140, 60, 960, 960]);
    if numel(refFreqs) == 4
        draw_cv_square_layout(cvData, refFreqs, colors);
    else
        draw_cv_flow_layout(cvData, refFreqs, colors);
    end
end

if ~isempty(ivIdx)
    ivData = datasets(ivIdx);
    deviceIndices = arrayfun(@(s) s.meta.deviceIndex, ivData);
    deviceIndices(isnan(deviceIndices)) = inf;
    [~, order] = sort(deviceIndices);
    ivData = ivData(order);

    figure('Name', 'Capacitor Breakdown I-V Preview');
    tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
    for idx = 1:numel(ivData)
        ds = ivData(idx);
        nexttile;
        semilogy(ds.raw.x, max(abs(ds.raw.y(:, 1)), eps), 'LineWidth', 1.1);
        grid on;
        xlabel(sprintf('%s (%s)', ds.raw.xName, ds.raw.xUnit));
        ylabel('|I| (A)');
        title(sprintf('Device %s', format_device_index(ds.meta.deviceIndex, ds.meta.deviceTag)), 'Interpreter', 'none');
    end
end
end


function draw_cv_square_layout(cvData, refFreqs, colors)
nDevices = numel(cvData);
squareLeft = 0.07;
squareBottom = 0.08;
squareSize = 0.86;
blockGapX = 0.10;
blockGapY = 0.055;
innerGap = 0.01;
titlePad = 0.045;

blockW = (squareSize - blockGapX) / 2;
blockH = (squareSize - blockGapY) / 2;
axesH = blockH - titlePad;
axW = (blockW - innerGap) / 2;

for j = 1:4
    row = ceil(j / 2);
    col = mod(j - 1, 2) + 1;
    blockLeft = squareLeft + (col - 1) * (blockW + blockGapX);
    blockBottom = squareBottom + (2 - row) * (blockH + blockGapY);

    annotation('textbox', [blockLeft, blockBottom + axesH, blockW, titlePad], ...
        'String', format_frequency_label(refFreqs(j)), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'FontWeight', 'bold', 'FontSize', 13);

    cpPos = [blockLeft, blockBottom, axW, axesH];
    gPos = [blockLeft + axW + innerGap, blockBottom, axW, axesH];

    axCp = axes('Position', cpPos); hold(axCp, 'on');
    axG = axes('Position', gPos); hold(axG, 'on');
    for idx = 1:nDevices
        ds = cvData(idx);
        if ~isempty(ds.raw.cp)
            plot(axCp, ds.raw.x, ds.raw.cp(:, min(j, size(ds.raw.cp, 2))), 'LineWidth', 1.15, 'Color', colors(idx, :));
        end
        if ~isempty(ds.raw.g)
            plot(axG, ds.raw.x, ds.raw.g(:, min(j, size(ds.raw.g, 2))), 'LineWidth', 1.15, 'Color', colors(idx, :));
        end
    end
    grid(axCp, 'on'); grid(axG, 'on');
    xlabel(axCp, 'V_{bias} (V)'); ylabel(axCp, 'Cp (F)'); axCp.YAxisLocation = 'left'; axCp.FontSize = 11; box(axCp, 'on');
    xlabel(axG, 'V_{bias} (V)'); ylabel(axG, 'G (S)'); axG.YAxisLocation = 'right'; axG.FontSize = 11; box(axG, 'on');
end
end


function draw_cv_flow_layout(cvData, refFreqs, colors)
nFreq = numel(refFreqs);
nDevices = numel(cvData);
tl = tiledlayout(nFreq, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, 'C-V By Frequency', 'FontSize', 14, 'FontWeight', 'bold');

for j = 1:nFreq
    nexttile; hold on;
    for idx = 1:nDevices
        ds = cvData(idx);
        if ~isempty(ds.raw.cp)
            plot(ds.raw.x, ds.raw.cp(:, min(j, size(ds.raw.cp, 2))), 'LineWidth', 1.15, 'Color', colors(idx, :));
        end
    end
    grid on; xlabel('V_{bias} (V)'); ylabel('Cp (F)');
    title(sprintf('%s | Cp', format_frequency_label(refFreqs(j)))); set(gca, 'FontSize', 11); box on; hold off;

    nexttile; hold on;
    for idx = 1:nDevices
        ds = cvData(idx);
        if ~isempty(ds.raw.g)
            plot(ds.raw.x, ds.raw.g(:, min(j, size(ds.raw.g, 2))), 'LineWidth', 1.15, 'Color', colors(idx, :));
        end
    end
    grid on; xlabel('V_{bias} (V)'); ylabel('G (S)'); set(gca, 'YAxisLocation', 'right');
    title(sprintf('%s | G', format_frequency_label(refFreqs(j)))); set(gca, 'FontSize', 11); box on; hold off;
end
end


function out = format_frequency_label(freqHz)
if freqHz >= 1e6
    out = sprintf('%.3g MHz', freqHz / 1e6);
elseif freqHz >= 1e3
    out = sprintf('%.3g kHz', freqHz / 1e3);
else
    out = sprintf('%.3g Hz', freqHz);
end
end


function out = format_device_index(deviceIndex, deviceTag)
if isfinite(deviceIndex)
    out = sprintf('%d', round(deviceIndex));
elseif ~isempty(deviceTag)
    out = deviceTag;
else
    out = 'Unknown';
end
end
