function result = capacitor_mode_breakdown(defaultDir)
filePaths = capacitor_select_multiple_files(defaultDir, 'Select breakdown I-V CSV file(s)', {'*.csv', 'CSV files (*.csv)'});
datasets = import_capacitor_measurements(filePaths);
datasets = datasets(arrayfun(@(s) strcmp(s.meta.testType, 'breakdown_iv'), datasets));
if isempty(datasets)
    error('Breakdown mode requires at least one breakdown I-V file.');
end

toxNm = input('Enter dielectric thickness (nm) for field calculation: ');
if isempty(toxNm) || ~isfinite(toxNm) || toxNm <= 0
    error('Dielectric thickness must be a positive number in nm.');
end

result = analyze_breakdown_datasets(datasets, toxNm);
plot_breakdown_results(result);

fprintf('\nBreakdown analysis completed.\n');
for k = 1:numel(result.devices)
    dev = result.devices(k);
    fprintf('\nDevice %s\n', format_device_index(dev.deviceIndex, ''));
    disp(dev.summary);
end
end


function result = analyze_breakdown_datasets(datasets, toxNm)
deviceIndices = arrayfun(@(s) s.meta.deviceIndex, datasets);
deviceKeys = unique(deviceIndices(isfinite(deviceIndices)));
devices = repmat(struct('deviceIndex', NaN, 'cycles', [], 'firstCycle', [], 'summary', table()), 1, numel(deviceKeys));

for i = 1:numel(deviceKeys)
    key = deviceKeys(i);
    grp = datasets(deviceIndices == key);
    iterVals = arrayfun(@(s) s.meta.iterationIndex, grp);
    iterVals(~isfinite(iterVals)) = inf;
    [~, order] = sort(iterVals);
    grp = grp(order);

    cycles = struct('fileName', {}, 'iterationIndex', {}, 'cycleIndex', {}, 'voltage', {}, 'current', {});
    for j = 1:numel(grp)
        splitCycles = split_breakdown_cycles(grp(j).raw.x(:), abs(grp(j).raw.y(:, 1)));
        for c = 1:numel(splitCycles)
            cycles(end+1).fileName = grp(j).meta.fileName; %#ok<AGROW>
            cycles(end).iterationIndex = grp(j).meta.iterationIndex;
            cycles(end).cycleIndex = c;
            cycles(end).voltage = splitCycles(c).voltage;
            cycles(end).current = splitCycles(c).current;
        end
    end

    if isempty(cycles)
        continue;
    end

    firstCycleIdx = pick_initial_cycle_index(cycles);
    firstCycle = analyze_first_breakdown_cycle(cycles(firstCycleIdx).voltage, cycles(firstCycleIdx).current, grp(1).config.complianceA, toxNm);
    firstCycle.cycleIndex = firstCycleIdx;

    devices(i).deviceIndex = key;
    devices(i).cycles = cycles;
    devices(i).firstCycle = firstCycle;
    devices(i).summary = table(key, firstCycle.Ileak_ref_A, firstCycle.Ileak_ref_V, ...
        firstCycle.rise_slope_dec_per_V, firstCycle.breakdown_voltage_V, firstCycle.breakdown_field_MV_um, ...
        firstCycle.uplimit_V, firstCycle.uplimit_field_MV_um, ...
        'VariableNames', {'DeviceIndex', 'Ileak_A', 'Ileak_V', 'Slope_dec_per_V', 'Vbd_V', 'Ebd_MV_um', 'uplimit_V', 'Euplimit_MV_um'});
end

result = struct('devices', devices, 'toxNm', toxNm);
end


function idx = pick_initial_cycle_index(cycles)
nC = numel(cycles);
scores = nan(1, nC);
for k = 1:nC
    V = cycles(k).voltage(:);
    I = abs(cycles(k).current(:));
    valid = isfinite(V) & isfinite(I) & I > 0;
    V = V(valid);
    I = I(valid);
    if isempty(V)
        continue;
    end
    nLow = max(5, round(0.15 * numel(V)));
    scores(k) = median(I(1:min(nLow, numel(I))), 'omitnan');
end

[~, idx] = min(scores);
if isempty(idx) || ~isfinite(scores(idx))
    idx = 1;
end
end


function splitCycles = split_breakdown_cycles(V, I)
V = V(:);
I = I(:);
valid = isfinite(V) & isfinite(I);
V = V(valid);
I = I(valid);
if isempty(V)
    splitCycles = struct('voltage', {}, 'current', {});
    return;
end

dV = diff(V);
resetIdx = find(dV < 0);
startIdx = [1; resetIdx + 1];
endIdx = [resetIdx; numel(V)];

splitCycles = struct('voltage', {}, 'current', {});
for k = 1:numel(startIdx)
    idx = startIdx(k):endIdx(k);
    if numel(idx) < 3
        continue;
    end
    splitCycles(end+1).voltage = V(idx); %#ok<AGROW>
    splitCycles(end).current = I(idx);
end

if isempty(splitCycles)
    splitCycles = struct('voltage', V, 'current', I);
end
end


function out = analyze_first_breakdown_cycle(V, I, complianceA, toxNm)
V = V(:);
I = I(:);
I(~isfinite(I) | I <= 0) = eps;
logI = log10(I);
N = numel(V);

nBase = max(6, round(0.15 * N));
idxBase = 1:nBase;
pBase = polyfit(V(idxBase), logI(idxBase), 1);

winPts = max(6, round(N / 20));
[idxRise, pRise] = find_best_linear_window(V, logI, winPts, nBase);

if abs(pRise(1) - pBase(1)) > eps
    uplimitV = (pBase(2) - pRise(2)) / (pRise(1) - pBase(1));
else
    uplimitV = NaN;
end

% Breakdown point: largest downward deviation from the best rising linear fit.
residual = polyval(pRise, V) - logI;
searchMask = false(size(V));
searchMask(idxRise(1):end) = true;
searchMask = searchMask & isfinite(residual);
if isfinite(complianceA) && complianceA > 0
    searchMask = searchMask & (I < 0.995 * complianceA);
end
if any(searchMask)
    residualMasked = residual;
    residualMasked(~searchMask) = -inf;
    [~, bdIdx] = max(residualMasked);
else
    [~, bdIdx] = max(residual);
end

ileakRefV = V(min(max(round(0.1 * N), 1), N));
ileakRefA = 10^(polyval(pBase, ileakRefV));
toxUm = toxNm * 1e-3;
Ebd = V(bdIdx) / toxUm;
if isfinite(uplimitV)
    Euplimit = uplimitV / toxUm;
    uplimitCurrent = 10^(polyval(pBase, uplimitV));
else
    Euplimit = NaN;
    uplimitCurrent = NaN;
end

out = struct( ...
    'voltage', V, ...
    'current', I, ...
    'logCurrent', logI, ...
    'idxBase', idxBase, ...
    'idxRise', idxRise, ...
    'baseFit', pBase, ...
    'riseFit', pRise, ...
    'Ileak_ref_V', ileakRefV, ...
    'Ileak_ref_A', ileakRefA, ...
    'rise_slope_dec_per_V', pRise(1), ...
    'breakdown_voltage_V', V(bdIdx), ...
    'breakdown_current_A', I(bdIdx), ...
    'breakdown_field_MV_um', Ebd, ...
    'uplimit_V', uplimitV, ...
    'uplimit_current_A', uplimitCurrent, ...
    'uplimit_field_MV_um', Euplimit);
end


function [bestIdx, bestFit] = find_best_linear_window(V, logI, winPts, nBase)
N = numel(V);
bestScore = -inf;
bestIdx = [];
bestFit = [NaN NaN];
startMin = min(max(nBase + 1, 1), max(1, N - winPts + 1));
endMax = max(startMin, N - winPts + 1);

for s = startMin:endMax
    idx = s:(s + winPts - 1);
    x = V(idx);
    y = logI(idx);
    if any(~isfinite(x)) || any(~isfinite(y))
        continue;
    end

    p = polyfit(x, y, 1);
    yFit = polyval(p, x);
    ssTot = sum((y - mean(y)).^2);
    ssRes = sum((y - yFit).^2);
    r2 = 1 - ssRes / max(ssTot, eps);
    ySpan = max(y) - min(y);
    slope = p(1);
    if slope <= 0 || ySpan < 0.35
        continue;
    end

    score = r2 + 0.02 * ySpan;
    if score > bestScore
        bestScore = score;
        bestIdx = idx;
        bestFit = p;
    end
end

if isempty(bestIdx)
    fallbackStart = min(max(nBase + 1, 1), max(1, N - winPts + 1));
    bestIdx = fallbackStart:min(N, fallbackStart + winPts - 1);
    bestFit = polyfit(V(bestIdx), logI(bestIdx), 1);
end
end


function plot_breakdown_results(result)
devices = result.devices;
nDev = numel(devices);
figure('Name', 'Breakdown Analysis');
set(gcf, 'Color', 'w', 'Units', 'pixels', 'Position', [140, 70, 900, max(900, 860*nDev)]);
tl = tiledlayout(nDev, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:nDev
    dev = devices(i);
    ax = nexttile(tl);
    hold(ax, 'on');
    rawX = [];
    rawY = [];

    for j = 1:numel(dev.cycles)
        cycV = dev.cycles(j).voltage(:);
        cycI = dev.cycles(j).current(:);
        cycI(~isfinite(cycI) | cycI <= 0) = NaN;
        rawX = [rawX; cycV(:)]; %#ok<AGROW>
        rawY = [rawY; cycI(:)]; %#ok<AGROW>
        if j == dev.firstCycle.cycleIndex
            continue;
        end
        semilogy(ax, cycV, cycI, '-', 'Color', [0.80 0.83 0.86], 'LineWidth', 0.7);
    end

    first = dev.firstCycle;
    firstI = first.current(:);
    firstI(~isfinite(firstI) | firstI <= 0) = NaN;
    rawX = [rawX; first.voltage(:)]; %#ok<AGROW>
    rawY = [rawY; firstI(:)]; %#ok<AGROW>
    semilogy(ax, first.voltage, firstI, '-', 'Color', [0.12 0.37 0.67], 'LineWidth', 1.9);

    rawX = rawX(isfinite(rawX));
    rawY = rawY(isfinite(rawY) & rawY > 0);
    if ~isempty(rawX) && ~isempty(rawY)
        xMin = min(rawX);
        xMax = max(rawX);
        yMin = min(rawY);
        yMax = max(rawY);
        xSpan = max(xMax - xMin, max(abs([xMin, xMax])) * 0.05);
        yLo = 10^(floor(log10(yMin)));
        yHi = 10^(ceil(log10(yMax)));
        xlim(ax, [xMin - 0.02*xSpan, xMax + 0.02*xSpan]);
        ylim(ax, [yLo, yHi]);
    end

    Vfit = linspace(min(first.voltage), max(first.voltage), 200);
    semilogy(ax, Vfit, 10.^(polyval(first.baseFit, Vfit)), '--', 'Color', [0.25 0.55 0.38], 'LineWidth', 1.2);
    semilogy(ax, Vfit, 10.^(polyval(first.riseFit, Vfit)), '--', 'Color', [0.82 0.37 0.23], 'LineWidth', 1.2);

    grid(ax, 'on');
    set(ax, 'YScale', 'log');
    xlabel(ax, 'Voltage (V)');
    ylabel(ax, '|I| (A)');
    ax.FontSize = 11;
    title(ax, sprintf('Device %s', format_device_index(dev.deviceIndex, '')), 'Interpreter', 'none');
    pbaspect(ax, [1 1 1]);

    yl = ylim(ax);
    xl = xlim(ax);

    plot(ax, first.breakdown_voltage_V, first.breakdown_current_A, 'o', ...
        'MarkerSize', 6, 'MarkerFaceColor', [0.55 0.10 0.10], 'MarkerEdgeColor', 'w', 'LineWidth', 0.8);
    semilogy(ax, [first.breakdown_voltage_V, first.breakdown_voltage_V], [yl(1), first.breakdown_current_A], ':', ...
        'Color', [0.55 0.10 0.10], 'LineWidth', 1.0);
    semilogy(ax, [xl(1), first.breakdown_voltage_V], [first.breakdown_current_A, first.breakdown_current_A], ':', ...
        'Color', [0.55 0.10 0.10], 'LineWidth', 1.0);

    if isfinite(first.uplimit_V) && isfinite(first.uplimit_current_A) && first.uplimit_current_A > 0
        plot(ax, first.uplimit_V, first.uplimit_current_A, 's', ...
            'MarkerSize', 6, 'MarkerFaceColor', [0.45 0.25 0.65], 'MarkerEdgeColor', 'w', 'LineWidth', 0.8);
        semilogy(ax, [first.uplimit_V, first.uplimit_V], [yl(1), first.uplimit_current_A], ':', ...
            'Color', [0.45 0.25 0.65], 'LineWidth', 1.0);
        semilogy(ax, [xl(1), first.uplimit_V], [first.uplimit_current_A, first.uplimit_current_A], ':', ...
            'Color', [0.45 0.25 0.65], 'LineWidth', 1.0);
    end

    txt = sprintf(['I_{leak}=%.3g pA @ %.3g V\n' ...
        '\n' ...
        'slope=%.3g dec/V\n' ...
        '\n' ...
        'V_{bd}=%.3g V | V_{uplimit}=%.3g V\n' ...
        '\n' ...
        'E_{bd}=%.3g MV/um | E_{uplimit}=%.3g MV/um'], ...
        first.Ileak_ref_A * 1e12, first.Ileak_ref_V, first.rise_slope_dec_per_V, ...
        first.breakdown_voltage_V, first.uplimit_V, ...
        first.breakdown_field_MV_um, first.uplimit_field_MV_um);
    text(ax, 0.05, 0.95, txt, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontSize', 10.4, 'BackgroundColor', [1 1 1 0.75], 'Margin', 6);
    hold(ax, 'off');
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

