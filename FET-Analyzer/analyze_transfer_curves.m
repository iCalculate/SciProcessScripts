function res = analyze_transfer_curves(data, varargin)
% ANALYZE_TRANSFER_CURVES  Analyze Id–Vg (transfer) curves with fixed log-scale y-axis.
% Features:
%   - Plot of |Id| vs Vg with a forced logarithmic y-axis and fixed y-limits
%   - Auto-select the subthreshold window just after leaving Ioff
%   - Highlight selected window points and draw the fitted line with extension
%
% INPUT:
%   data : struct returned by import_clarius_csv(FileType="transfer"), or
%          a table with variables Vg, Id (optionally Vd)
%
% Name-Value options:
%   "YLim"         : [1e-13, 1e-3]  % fixed log-scale y-axis range (>0)
%   "DoPlot"       : true
%   "SmoothingPts" : 5              % moving average points for d(log10|Id|)/dVg
%   "PeakFrac"     : 0.8            % window is where slope >= PeakFrac * peak slope
%   "IoffFrac"     : 3              % define leaving Ioff as Id >= Ioff*IoffFrac
%   "PadVg"        : 0.1            % extend the fitted line by ±PadVg (V)
%   "MinWinPts"    : 6              % minimal points in SS window
%
% OUTPUT:
%   res.PerCurve(k) with fields:
%       Vd, Ioff_est, Vg_win[1x2], idx_win, slope_dec_per_V, intercept_dec, SS_mV_dec
%   res.Summary : table across curves
%   res.Figures : figure handles when DoPlot is true
%
% SS definition:
%   Fit log10(|Id|) = a*Vg + b inside the first steep window.
%   SS = (1/a) * 1e3 mV/dec.

    % -------- Options --------
    p = inputParser;
    addParameter(p, "YLim", [1e-13, 1e-3], @(v)isnumeric(v)&&numel(v)==2&&all(v>0));
    addParameter(p, "DoPlot", true, @(x)islogical(x));
    addParameter(p, "SmoothingPts", 5, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
    addParameter(p, "PeakFrac", 0.8, @(x)isnumeric(x)&&isscalar(x)&&x>0 && x<=1);
    addParameter(p, "IoffFrac", 3, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
    addParameter(p, "PadVg", 0.1, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
    addParameter(p, "MinWinPts", 6, @(x)isnumeric(x)&&isscalar(x)&&x>=3);
    parse(p, varargin{:});
    opt = p.Results;

    figs = struct('overview', [], 'detail', []);
    curves = normalize_input_as_columns(data);

    nC = numel(curves);
    PerCurve = repmat(struct( ...
        'Vd', NaN, ...
        'Ioff_est', NaN, ...
        'Vg_win', [NaN NaN], ...
        'idx_win', [], ...
        'slope_dec_per_V', NaN, ...
        'intercept_dec', NaN, ...
        'SS_mV_dec', NaN ...
    ), 1, nC);

    % -------- Overview figure --------
    if opt.DoPlot
        figs.overview = figure('Name','Transfer Curves: |Id| vs Vg (fixed log y)');
        hold on; grid on;
        xlabel('V_g (V)'); ylabel('|I_d| (A)');
        ylim(opt.YLim);
        set(gca,'YScale','log');   % ★ 强制Y轴对数
    end

    colors = lines(max(nC,7));

    for k = 1:nC
        Vg = curves(k).Vg(:);
        Id = curves(k).Id(:);
        Vd_val = curves(k).Vd;

        % Clean and bounds
        absId = abs(Id);
        tiny = opt.YLim(1);
        absId(~isfinite(absId) | absId<=0) = tiny;   % 防止对数报错
        logId = log10(absId);

        % Estimate Ioff using lowest 10% of currents
        [~,ord] = sort(absId);
        nLow = max(5, round(0.10*numel(absId)));
        Ioff_est = median(absId(ord(1:nLow)), 'omitnan');

        % d(log10|Id|)/dVg with smoothing
        dVg = gradient(Vg);
        dlogId_dVg_raw = gradient(logId)./max(dVg, eps);
        dlogId_dVg = movmean(dlogId_dVg_raw, max(1,round(opt.SmoothingPts)));

        % First index leaving Ioff
        leave_idx = find(absId >= Ioff_est*opt.IoffFrac, 1, 'first');
        if isempty(leave_idx), leave_idx = 1; end

        % First strong slope peak after leaving Ioff
        [peakSlope, peakIdx] = max(dlogId_dVg(leave_idx:end));
        peakIdx = peakIdx + leave_idx - 1;
        if ~isfinite(peakSlope), peakSlope = 0; peakIdx = leave_idx; end

        % Window where slope >= PeakFrac * peakSlope, ensure connectivity
        thr = opt.PeakFrac * peakSlope;
        L = peakIdx; R = peakIdx;
        while L>1 && dlogId_dVg(L-1) >= thr, L=L-1; end
        while R<numel(Vg) && dlogId_dVg(R+1) >= thr, R=R+1; end

        % Ensure minimal window size
        if (R-L+1) < opt.MinWinPts
            pad = ceil((opt.MinWinPts - (R-L+1))/2);
            L = max(1, L-pad); R = min(numel(Vg), R+pad);
        end
        idx_win = (L:R);
        Vg_win = [Vg(L), Vg(R)];

        % Linear regression on log10(|Id|) = a*Vg + b
        vX = Vg(idx_win); vY = logId(idx_win);
        [a,b] = linefit_dec(vX, vY);  % dec per V

        % Subthreshold swing
        SS_mV_dec = (1 / max(a, eps)) * 1e3;

        % Save metrics
        PerCurve(k).Vd = Vd_val;
        PerCurve(k).Ioff_est = Ioff_est;
        PerCurve(k).Vg_win = Vg_win;
        PerCurve(k).idx_win = idx_win;
        PerCurve(k).slope_dec_per_V = a;
        PerCurve(k).intercept_dec = b;
        PerCurve(k).SS_mV_dec = SS_mV_dec;

        % Plot main curve, highlight window, and draw extended fit line
        if opt.DoPlot
            c = colors(mod(k-1,size(colors,1))+1,:);
            plot(Vg, absId, '-', 'Color', c, 'LineWidth', 1.2); hold on;

            % highlight window points
            plot(Vg(idx_win), absId(idx_win), 'o', ...
                'MarkerSize', 5, 'MarkerFaceColor', c, 'MarkerEdgeColor', 'k');

            % extended fit line
            Vg_fit = linspace(Vg_win(1)-opt.PadVg, Vg_win(2)+opt.PadVg, 200);
            logId_fit = a*Vg_fit + b;
            Id_fit = 10.^logId_fit;
            plot(Vg_fit, Id_fit, '--', 'Color', c, 'LineWidth', 1.4);

            % Ioff reference line
            yline(clamp(Ioff_est, opt.YLim(1), opt.YLim(2)), ':', 'Color', [0.3 0.3 0.3]);

            ylim(opt.YLim);
            set(gca,'YScale','log');   % ★ 强制Y轴对数
            grid on;
        end
    end

    if opt.DoPlot
        title('|I_d|–V_g with fixed log y and SS window highlighted');
        hold off;

        % Per curve detail
        figs.detail = figure('Name','Per-curve SS window and fit');
        tiledlayout('flow');
        for k = 1:nC
            nexttile; hold on; grid on;
            Vg = curves(k).Vg(:); absId = abs(curves(k).Id(:));
            absId(~isfinite(absId) | absId<=0) = opt.YLim(1);
            plot(Vg, absId, '-', 'LineWidth', 1.0);

            idx_win = PerCurve(k).idx_win;
            plot(Vg(idx_win), absId(idx_win), 'o', ...
                'MarkerFaceColor',[.9 .3 .2], 'MarkerEdgeColor','k', 'MarkerSize', 5);

            % fit line
            a = PerCurve(k).slope_dec_per_V; b = PerCurve(k).intercept_dec;
            Vg_win = PerCurve(k).Vg_win;
            Vg_fit = linspace(Vg_win(1)-opt.PadVg, Vg_win(2)+opt.PadVg, 200);
            plot(Vg_fit, 10.^(a*Vg_fit + b), '--', 'LineWidth', 1.2);

            ylim(opt.YLim);
            set(gca,'YScale','log');   % ★ 强制Y轴对数
            xlabel('V_g (V)'); ylabel('|I_d| (A)');

            ttl = sprintf('SS = %.1f mV/dec', PerCurve(k).SS_mV_dec);
            if isfinite(PerCurve(k).Vd)
                ttl = sprintf('V_d = %.3g V, %s', PerCurve(k).Vd, ttl);
            end
            title(ttl);
            hold off;
        end
    end

    % -------- Summary table --------
    Vd_col = arrayfun(@(x)x.Vd, PerCurve).';
    SS_col = arrayfun(@(x)x.SS_mV_dec, PerCurve).';
    VgL = arrayfun(@(x)x.Vg_win(1), PerCurve).';
    VgR = arrayfun(@(x)x.Vg_win(2), PerCurve).';
    Ioff_col = arrayfun(@(x)x.Ioff_est, PerCurve).';
    Slope_col = arrayfun(@(x)x.slope_dec_per_V, PerCurve).';
    res.PerCurve = PerCurve;
    res.Summary = table(Vd_col, SS_col, VgL, VgR, Ioff_col, Slope_col, ...
        'VariableNames', {'Vd','SS_mV_per_dec','Vg_left','Vg_right','Ioff_est','slope_dec_per_V'});
    res.Figures = figs;
end

% ===== helpers =====

function curves = normalize_input_as_columns(data)
    % Produce an array of structs with fields Vg, Id, Vd (scalar per curve)
    if isstruct(data) && isfield(data,'Data') && isfield(data.Data,'Matrix')
        M = data.Data.Matrix;
        if ~isfield(M,'Vg') || ~isfield(M,'Id')
            error('Matrix must contain fields Vg and Id for transfer analysis.');
        end
        Vg = M.Vg; Id = M.Id;
        if isfield(M,'Vd'), Vd = M.Vd; else, Vd = nan(size(Id)); end
        nC = size(Id,2);
        curves = repmat(struct('Vg',[],'Id',[],'Vd',NaN), 1, nC);
        for j=1:nC
            curves(j).Vg = Vg(:,min(j,size(Vg,2)));
            curves(j).Id = Id(:,j);
            curves(j).Vd = mean(Vd(:,min(j,size(Vd,2))), 'omitnan');
        end
    elseif istable(data)
        T = data;
        must = {'Vg','Id'};
        for k=1:numel(must)
            if ~ismember(must{k}, T.Properties.VariableNames)
                error('Table must contain variable %s.', must{k});
            end
        end
        if ismember('Vd', T.Properties.VariableNames)
            [~, ~, grp] = unique(round(T.Vd(:), 9)); % group by Vd
            nC = numel(unique(grp));
            curves = repmat(struct('Vg',[],'Id',[],'Vd',NaN), 1, nC);
            uVd = unique(T.Vd(:));
            for j=1:nC
                idx = (grp==j);
                Vg = T.Vg(idx); Id = T.Id(idx);
                [Vg,ord] = sort(Vg); Id = Id(ord);
                curves(j).Vg = Vg(:); curves(j).Id = Id(:);
                curves(j).Vd = uVd(j);
            end
        else
            [Vg,ord] = sort(T.Vg(:)); Id = T.Id(ord);
            curves = struct('Vg', Vg(:), 'Id', Id(:), 'Vd', NaN);
        end
    else
        error('Unsupported input. Pass importer struct or a table with Vg, Id [, Vd].');
    end
end

function [a,b] = linefit_dec(x, y)
    % Linear fit y = a*x + b. Use robustfit if available.
    if exist('robustfit','file')
        coef = robustfit(x,y);  % y = coef(1) + coef(2)*x
        b = coef(1); a = coef(2);
    else
        X = [ones(numel(x),1), x(:)];
        coef = X \ y(:);
        b = coef(1); a = coef(2);
    end
end

function v = clamp(x, lo, hi)
    v = min(max(x, lo), hi);
end
