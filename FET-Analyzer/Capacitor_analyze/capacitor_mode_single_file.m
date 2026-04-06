function result = capacitor_mode_single_file(defaultDir)
filePath = capacitor_select_single_file(defaultDir, 'Select one C-V CSV file', {'*.csv', 'CSV files (*.csv)'});
datasets = import_capacitor_measurements(filePath);
dataset = datasets(1);

fprintf('\nImported 1 file.\n');
fprintf('%s | type=%s | sample=%s | points=%d\n', ...
    dataset.meta.fileName, dataset.meta.testType, dataset.meta.sampleName, numel(dataset.raw.x));

if ~strcmp(dataset.meta.testType, 'cv')
    error('Single-file mode currently expects a C-V file.');
end

selectedFreqIdx = prompt_frequency_selection(dataset);
plot_single_cv_file(dataset, selectedFreqIdx);
fprintf('Single-file plot completed.\n');
result = dataset;
end


function freqIdx = prompt_frequency_selection(dataset)
freqs = dataset.config.frequenciesHz;
if isempty(freqs)
    freqs = 1:size(dataset.raw.cp, 2);
end

fprintf('\nAvailable frequencies in the selected file:\n');
for k = 1:numel(freqs)
    fprintf('  %d. %s\n', k, format_frequency_label(freqs(k)));
end
raw = strtrim(input('Enter one or more frequency indices, e.g. 1 or [1 3 4]: ', 's'));
if isempty(raw)
    freqIdx = 1:numel(freqs);
else
    freqIdx = str2num(raw); %#ok<ST2NM>
end
if isempty(freqIdx) || any(~ismember(freqIdx, 1:numel(freqs)))
    error('Invalid frequency selection.');
end
freqIdx = unique(freqIdx(:).');
fprintf('Selected frequency indices: %s\n', mat2str(freqIdx));
end


function plot_single_cv_file(dataset, selectedFreqIdx)
freqs = dataset.config.frequenciesHz;
cp = dataset.raw.cp;
g = dataset.raw.g;
x = dataset.raw.x;

if isempty(cp)
    error('Selected file does not contain Cp data.');
end
if isempty(freqs)
    freqs = 1:size(cp, 2);
end

colors = turbo(max(numel(selectedFreqIdx), 2));
figure('Name', sprintf('Single File C-V: %s', dataset.meta.fileName));
set(gcf, 'Units', 'pixels', 'Position', [160, 120, 860, 620]);
ax = axes('Position', [0.12, 0.12, 0.76, 0.76]);
hold(ax, 'on');

yyaxis(ax, 'left');
for n = 1:numel(selectedFreqIdx)
    j = selectedFreqIdx(n);
    plot(ax, x, cp(:, j), '-', 'LineWidth', 1.3, 'Color', colors(n, :));
end
ylabel(ax, 'Cp (F)');

yyaxis(ax, 'right');
if ~isempty(g)
    for n = 1:numel(selectedFreqIdx)
        j = selectedFreqIdx(n);
        if j <= size(g, 2)
            plot(ax, x, g(:, j), '--', 'LineWidth', 1.2, 'Color', colors(n, :));
        end
    end
end
ylabel(ax, 'G (S)');

grid(ax, 'on');
xlabel(ax, 'V_{bias} (V)');
title(ax, dataset.meta.sampleName, 'Interpreter', 'none');
ax.FontSize = 12;
legendEntries = arrayfun(@format_frequency_label, freqs(selectedFreqIdx), 'UniformOutput', false);
legend(ax, legendEntries, 'Location', 'eastoutside');
hold(ax, 'off');
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
