function result = capacitor_mode_fitting(defaultDir)
filePaths = capacitor_select_multiple_files(defaultDir, 'Select C-V CSV file(s) for fitting', {'*.csv', 'CSV files (*.csv)'});
datasets = import_capacitor_measurements(filePaths);
datasets = datasets(arrayfun(@(s) strcmp(s.meta.testType, 'cv'), datasets));
if isempty(datasets)
    error('Fitting mode requires at least one C-V file.');
end

areaCsvPath = capacitor_select_single_file(defaultDir, 'Select area CSV file', {'*.csv', 'CSV files (*.csv)'});
areaData = import_capacitor_area_csv(areaCsvPath);
toxNm = input('Enter oxide thickness (nm): ');
if isempty(toxNm) || ~isfinite(toxNm) || toxNm <= 0
    error('Oxide thickness must be a positive number in nm.');
end

fitResult = fit_capacitor_area_curve(datasets, areaData, toxNm);
plot_capacitor_fit(fitResult);
for j = 1:numel(fitResult.perFrequency)
    fprintf('\nFrequency %s\n', fitResult.perFrequency(j).freqLabel);
    disp(fitResult.perFrequency(j).summary);
end
fprintf('Fitting completed for %d frequency/frequencies.\n', numel(fitResult.perFrequency));
result = fitResult;
end


function fitResult = fit_capacitor_area_curve(datasets, areaData, toxNm)
deviceIndices = arrayfun(@(s) s.meta.deviceIndex, datasets);
[~, order] = sort(deviceIndices);
datasets = datasets(order);

if numel(areaData.areaUm2) < numel(datasets)
    error('Area CSV has fewer rows than the number of selected C-V files.');
end

nUse = min(numel(datasets), numel(areaData.areaUm2));
refFreqs = datasets(1).config.frequenciesHz;
if isempty(refFreqs)
    refFreqs = 1:size(datasets(1).raw.cp, 2);
end

nFreq = numel(refFreqs);
perFrequency = repmat(struct('frequencyHz', NaN, 'freqLabel', '', 'deviceIndex', [], ...
    'areaLabel', strings(0,1), 'areaUm2', [], 'capacitance', [], 'coxPerUm2', NaN, ...
    'cox_uF_per_cm2', NaN, 'coxPerM2', NaN, 'intercept', NaN, 'r2', NaN, ...
    'epsr', NaN, 'summary', table()), 1, nFreq);

for j = 1:nFreq
    areasUm2 = nan(nUse, 1);
    cValues = nan(nUse, 1);
    usedDeviceIndices = nan(nUse, 1);
    usedLabels = strings(nUse, 1);

    for k = 1:nUse
        ds = datasets(k);
        if ~isfinite(ds.meta.deviceIndex) || isempty(ds.raw.cp)
            continue;
        end
        cValues(k) = max(ds.raw.cp(:, min(j, size(ds.raw.cp, 2))), [], 'omitnan');
        areasUm2(k) = areaData.areaUm2(k);
        usedDeviceIndices(k) = ds.meta.deviceIndex;
        usedLabels(k) = areaData.label(k);
    end

    mask = isfinite(cValues) & isfinite(areasUm2);
    areasUm2 = areasUm2(mask);
    cValues = cValues(mask);
    usedDeviceIndices = usedDeviceIndices(mask);
    usedLabels = usedLabels(mask);
    if numel(areasUm2) < 2
        error('At least two valid devices with matched areas are required for fitting.');
    end

    coef = polyfit(areasUm2, cValues, 1);
    fitY = polyval(coef, areasUm2);
    coxPerUm2 = coef(1);
    coxPerM2 = coxPerUm2 * 1e12;
    toxM = toxNm * 1e-9;
    eps0 = 8.854187817e-12;

    perFrequency(j).frequencyHz = refFreqs(j);
    perFrequency(j).freqLabel = format_frequency_label(refFreqs(j));
    perFrequency(j).deviceIndex = usedDeviceIndices(:);
    perFrequency(j).areaLabel = usedLabels(:);
    perFrequency(j).areaUm2 = areasUm2(:);
    perFrequency(j).capacitance = cValues(:);
    perFrequency(j).coxPerUm2 = coxPerUm2;
    perFrequency(j).cox_uF_per_cm2 = coxPerUm2 * 1e14;
    perFrequency(j).coxPerM2 = coxPerM2;
    perFrequency(j).intercept = coef(2);
    perFrequency(j).r2 = calc_r2(cValues, fitY);
    perFrequency(j).epsr = coxPerM2 * toxM / eps0;
    perFrequency(j).summary = table(usedDeviceIndices(:), usedLabels(:), areasUm2(:), cValues(:), ...
        'VariableNames', {'DeviceIndex', 'AreaLabel', 'Area_um2', 'Cmax'});
end

fitResult = struct('toxNm', toxNm, 'perFrequency', perFrequency);
end


function plot_capacitor_fit(fitResult)
perFrequency = fitResult.perFrequency;
nFreq = numel(perFrequency);
figure('Name', 'Capacitor Area Fit');
set(gcf, 'Color', 'w', 'Units', 'pixels', 'Position', [120, 80, 1080, 760]);
ax = axes('Parent', gcf, 'Position', [0.1, 0.11, 0.83, 0.8]);
hold(ax, 'on'); grid(ax, 'on');
ax.GridColor = [0.86 0.88 0.9]; ax.GridAlpha = 0.8; ax.Box = 'on'; ax.LineWidth = 0.8; ax.FontSize = 11;
ax.XColor = [0.2 0.22 0.25]; ax.YColor = [0.2 0.22 0.25];

colors = [0.13 0.38 0.67; 0.82 0.37 0.23; 0.18 0.56 0.42; 0.55 0.33 0.64; 0.72 0.51 0.16; 0.15 0.56 0.62];
for j = 1:min(nFreq, 4)
    pf = perFrequency(j);
    c = colors(mod(j - 1, size(colors,1)) + 1, :);
    scatter(ax, pf.areaUm2, pf.capacitance, 48, 'MarkerFaceColor', c, 'MarkerEdgeColor', 0.95*[1 1 1], 'LineWidth', 0.8);
    xFit = linspace(min(pf.areaUm2), max(pf.areaUm2), 200);
    yFit = pf.coxPerUm2 * xFit + pf.intercept;
    plot(ax, xFit, yFit, '-', 'LineWidth', 1.8, 'Color', darken_color(c, 0.18));
end

xlabel(ax, 'Device Area (\mum^2)');
ylabel(ax, 'C (F)');
[allX, allY] = collect_fit_limits(perFrequency, min(nFreq, 4));
if ~isempty(allX)
    xMin = min(allX); xMax = max(allX); yMin = min(allY); yMax = max(allY);
    xSpan = max(xMax - xMin, max(abs([xMin, xMax])) * 0.1);
    ySpan = max(yMax - yMin, max(abs([yMin, yMax])) * 0.1);
    xlim(ax, [xMin - 0.05*xSpan, xMax + 0.08*xSpan]);
    ylim(ax, [yMin - 0.08*ySpan, yMax + 0.32*ySpan]);
end

for j = 1:min(nFreq, 4)
    pf = perFrequency(j);
    c = colors(mod(j - 1, size(colors,1)) + 1, :);
    xNorm = (j - 1.0) / min(nFreq, 4);
    infoText = sprintf('%s \nR^2=%.4f \nC_{ox}=%.4g \\muF/cm^2 \n\\epsilon_r=%.4g', ...
        pf.freqLabel, pf.r2, pf.cox_uF_per_cm2, pf.epsr);
    text(ax, xNorm, 0.97, infoText, 'Units', 'normalized', 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'top', 'FontSize', 10.5, 'FontWeight', 'bold', ...
        'Color', darken_color(c, 0.18), 'Interpreter', 'tex');
end
hold(ax, 'off');
end


function [allX, allY] = collect_fit_limits(perFrequency, nKeep)
allX = []; allY = [];
for j = 1:nKeep
    allX = [allX; perFrequency(j).areaUm2(:)]; %#ok<AGROW>
    allY = [allY; perFrequency(j).capacitance(:)]; %#ok<AGROW>
end
allX = allX(isfinite(allX));
allY = allY(isfinite(allY));
end


function r2 = calc_r2(y, yhat)
y = y(:); yhat = yhat(:);
ssTot = sum((y - mean(y)).^2);
ssRes = sum((y - yhat).^2);
r2 = 1 - ssRes / max(ssTot, eps);
end


function out = darken_color(c, amount)
out = max(c - amount, 0);
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
