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
%       Vd, Ioff_est, Vg_win[1x2], idx_win, slope_dec_per_V, intercept_dec, SS_mV_dec,
%       SS_fit_R2 (R² of SS linear fit), Von_fit_R2 (R² of Von linear fit), ...
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
    addParameter(p, "ComplianceFrac", 0.98, @(x)isnumeric(x)&&isscalar(x)&&x>0&&x<=1);
    addParameter(p, "DenoiseWindow", 5, @(x)isnumeric(x)&&isscalar(x)&&x>=3);
    % Von fit quality controls
    addParameter(p, "MinVonPoints", 5, @(x)isnumeric(x)&&isscalar(x)&&x>=3);
    addParameter(p, "MinVonR2", 0.95, @(x)isnumeric(x)&&isscalar(x)&&x>=0&&x<=1);
    addParameter(p, "MinVonSlope", 1e-12, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
    addParameter(p, "MinVonDeltaVg", 0.02, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
    % Vth extrapolation fixed current reference
    addParameter(p, "IrefFixed", 1e-7, @(x)isnumeric(x)&&isscalar(x)&&x>0);
    parse(p, varargin{:});
    opt = p.Results;

    figs = struct('overview', [], 'detail', []);
    curves = normalize_input_as_columns(data);
    % Extract compliance current (if available) from importer struct
    complianceA = NaN;
    if isstruct(data) && isfield(data,'TestParameter') && isfield(data.TestParameter,'Measurement') ...
            && isfield(data.TestParameter.Measurement,'Secondary') && isfield(data.TestParameter.Measurement.Secondary,'Compliance')
        try
            complianceA = data.TestParameter.Measurement.Secondary.Compliance; % in Amps
        catch
            complianceA = NaN;
        end
    end

    nC = numel(curves);
    PerCurve = repmat(struct( ...
        'Vd', NaN, ...
        'Ioff_est', NaN, ...
        'Vg_win', [NaN NaN], ...
        'idx_win', [], ...
        'slope_dec_per_V', NaN, ...
        'intercept_dec', NaN, ...
        'SS_mV_dec', NaN, ...
        'SS_fit_R2', NaN, ...
        'Ion_Ioff_ratio', NaN, ...
        'gm_max', NaN, ...
        'Vth', NaN, ...
        'Vth_extrap_Ioff', NaN, ...
        'Vth_extrap_Iref', NaN, ...
        'Vth_gmmax', NaN, ...
        'Vth_extrap_cross', NaN, ...
        'Von', NaN, ...
        'Von_fit_R2', NaN ...
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
    % Track dynamic y-range for overview (exclude compliance-masked points)
    yMinOverview = inf; yMaxOverview = 0;

    for k = 1:nC
        Vg = curves(k).Vg(:);
        Id = curves(k).Id(:);
        Vd_val = curves(k).Vd;

        % Clean and bounds - handle current saturation/clipping and compliance
        absId = abs(Id);
        tiny = opt.YLim(1);
        
        % Compliance-based mask: drop points at or above compliance
        comp_mask = false(size(absId));
        if isfinite(complianceA) && complianceA > 0
            comp_mask = absId >= opt.ComplianceFrac * complianceA;
        end
        
        % Do not mask low-current flat segments; only use compliance to cut off
        
        % Apply mask as NaN to preserve alignment
        absId(comp_mask) = NaN;
        
        % For analysis, use a NaN-safe copy; for plotting keep NaNs to break lines
        calcAbsId = absId;
        calcAbsId(~isfinite(calcAbsId) | calcAbsId<=0) = tiny;   % 防止对数报错
        logId = log10(calcAbsId);
        % Light adaptive denoising on log current for slope detection only
        % Estimate noise level from off region (lowest 10% by current)
        [~,ord_calc] = sort(calcAbsId);
        nLow_calc = max(5, round(0.10*numel(calcAbsId)));
        off_log = log10(calcAbsId(ord_calc(1:nLow_calc)));
        noise_std = std(off_log, 'omitnan');
        win = max(3, 2*floor(opt.DenoiseWindow/2)+1); % odd window
        logId_smooth = logId;
        if any(isfinite(logId))
            if exist('sgolayfilt','file') && win>=5
                % SG filter preserves slopes with minimal attenuation
                try
                    logId_smooth = sgolayfilt(fillmissing(logId,'nearest'), 2, win);
                catch
                    logId_smooth = movmean(logId, win, 'omitnan');
                end
            else
                % Mild median + mean combo to suppress spikes but keep slope
                logId_med = movmedian(logId, win, 'omitnan');
                logId_smooth = movmean(logId_med, win, 'omitnan');
            end
            % Blend towards smoothed only when noise is significant
            alpha = clamp(noise_std/0.2, 0, 1); % heuristic blend factor
            logId_for_slope = (1-alpha).*logId + alpha.*logId_smooth;
        else
            logId_for_slope = logId;
        end
        % Update overview y-range from valid non-compliance points
        valid_noncomp = isfinite(calcAbsId) & calcAbsId>0 & ~comp_mask;
        if any(valid_noncomp)
            yMinOverview = min(yMinOverview, min(calcAbsId(valid_noncomp)));
            yMaxOverview = max(yMaxOverview, max(calcAbsId(valid_noncomp)));
        end

        % Estimate Ioff using lowest 10% of currents
        [~,ord] = sort(absId);
        nLow = max(5, round(0.10*numel(absId)));
        Ioff_est = median(absId(ord(1:nLow)), 'omitnan');

        % d(log10|Id|)/dVg with smoothing
        dVg = gradient(Vg);
        dlogId_dVg_raw = gradient(logId_for_slope)./max(dVg, eps);
        dlogId_dVg = movmean(dlogId_dVg_raw, max(1,round(opt.SmoothingPts)));

        % First index leaving Ioff
        leave_idx = find(absId >= Ioff_est*opt.IoffFrac, 1, 'first');
        if isempty(leave_idx), leave_idx = 1; end

        % First strong slope peak after leaving Ioff
        [peakSlope, peakIdx] = max(dlogId_dVg(leave_idx:end));
        peakIdx = peakIdx + leave_idx - 1;
        if ~isfinite(peakSlope), peakSlope = 0; peakIdx = leave_idx; end

        % Window where slope >= PeakFrac * peakSlope, ensure connectivity
        % Avoid including noisy early segment by requiring slope exceed a small absolute threshold
        abs_slope_thr = 0.5 * max(peakSlope, eps) * (1-opt.PeakFrac);
        thr = max(opt.PeakFrac * peakSlope, abs_slope_thr);
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

        % Linear regression on log10(|Id|) = a*Vg + b (exclude masked NaNs)
        valid_fit = isfinite(Vg(idx_win)) & isfinite(logId(idx_win));
        vX = Vg(idx_win); vY = logId(idx_win);
        vX = vX(valid_fit); vY = vY(valid_fit);
        if numel(vX) < max(3, opt.MinWinPts)
            % If too few points remain after masking, expand window within bounds
            L2 = max(1, L-1); R2 = min(numel(Vg), R+1);
            idx2 = (L2:R2);
            vf2 = isfinite(Vg(idx2)) & isfinite(logId(idx2));
            vX = Vg(idx2); vY = logId(idx2);
            vX = vX(vf2); vY = vY(vf2);
        end
        [a,b] = linefit_dec(vX, vY);  % dec per V

        % R² of SS fit: log10(|Id|) = a*Vg + b
        if numel(vX) >= 2
            yhat_ss = a*vX + b;
            ybar_ss = mean(vY);
            ss_tot_ss = sum((vY - ybar_ss).^2);
            ss_res_ss = sum((vY - yhat_ss).^2);
            R2_SS = 1 - ss_res_ss / max(ss_tot_ss, eps);
        else
            R2_SS = NaN;
        end

        % Subthreshold swing
        SS_mV_dec = (1 / max(a, eps)) * 1e3;

        % Calculate additional metrics
        % 1. Ion/Ioff ratio (maximum current / minimum current)
        valid_idx = isfinite(absId) & absId > 0;
        if any(valid_idx)
            Ion_Ioff_ratio = max(absId(valid_idx)) / min(absId(valid_idx));
        else
            Ion_Ioff_ratio = NaN;
        end
        
        % 2. Maximum transconductance (gm = dId/dVg), robust to compliance cutoffs
        if numel(Vg) > 1
            dVg = gradient(Vg);
            % Mask compliance in current, then apply light smoothing to avoid spikes
            Id_gm = Id;
            Id_gm(comp_mask) = NaN;
            gmSmoothWin = max(3, 2*floor(opt.DenoiseWindow/2)+1);
            Id_gm_smooth = Id_gm;
            Id_gm_smooth = movmean(Id_gm_smooth, gmSmoothWin, 'omitnan');
            Id_gm_filled = fillmissing(Id_gm_smooth, 'nearest');
            gm_tmp = gradient(Id_gm_filled) ./ max(dVg, eps);
            % Invalidate gm at and next to compliance boundaries to remove artificial jumps
            comp_adj = comp_mask;
            comp_adj(1:end-1) = comp_adj(1:end-1) | comp_mask(2:end);
            comp_adj(2:end) = comp_adj(2:end) | comp_mask(1:end-1);
            gm_tmp(comp_adj) = NaN;
            gm = gm_tmp;
            gm_max = max(gm(isfinite(gm)));
            if any(isfinite(gm))
                [~, gm_idx] = max(gm);
                Vth_gmmax = Vg(gm_idx);
            else
                Vth_gmmax = NaN;
            end
        else
            gm_max = NaN;
            Vth_gmmax = NaN;
        end
        
        % 3. Threshold voltage by extrapolation
        Vth = NaN; Vth_extrap_Ioff = NaN; Vth_extrap_Iref = NaN;
        if isfinite(a) && isfinite(b) && a > 0
            % Using Ioff-based reference (e.g., Ioff*10) and fixed Iref
            target_logId_Ioff = log10(Ioff_est * 10);
            Vth_extrap_Ioff = (target_logId_Ioff - b) / a;
            target_logId_Iref = log10(opt.IrefFixed);
            Vth_extrap_Iref = (target_logId_Iref - b) / a;
            % default Vth to Ioff-based value
            Vth = Vth_extrap_Ioff;
        end

        % 4. Von from linear Id–Vg fit in above-threshold region
        Von = NaN;
        valid_lin = isfinite(Id) & isfinite(Vg) & ~comp_mask;
        if any(valid_lin)
            % choose high-current band (e.g., top 20% of valid Id) to avoid subthreshold
            Id_valid = Id(valid_lin);
            Vg_valid = Vg(valid_lin);
            prc = 80;
            thr_lin = prctile(abs(Id_valid), prc);
            sel = abs(Id_valid) >= thr_lin;
            if nnz(sel) >= opt.MinVonPoints && (max(Vg_valid(sel))-min(Vg_valid(sel))) >= opt.MinVonDeltaVg
                X = [Vg_valid(sel), ones(nnz(sel),1)];
                coef_lin = X \ Id_valid(sel);
                a_lin = coef_lin(1); b_lin = coef_lin(2);
                % Compute simple R^2 to assess linearity
                yhat = X*coef_lin; ybar = mean(Id_valid(sel));
                ss_tot = sum((Id_valid(sel)-ybar).^2);
                ss_res = sum((Id_valid(sel)-yhat).^2);
                R2 = 1 - ss_res/max(ss_tot, eps);
                PerCurve(k).Von_fit_R2 = R2;  % 保存 Von 线性拟合的 R²（无论是否采纳 Von）
                if isfinite(a_lin) && abs(a_lin) >= opt.MinVonSlope && R2 >= opt.MinVonR2
                    Von = -b_lin / a_lin; % intercept with Id=0 on linear scale
                end
            end
        end

        % 5. Vth by extrapolation to linear intersection (solve 10^(a*V+b) = a_lin*V + b_lin)
        Vth_extrap_cross = NaN;
        if exist('a_lin','var') && exist('b_lin','var') && isfinite(a) && isfinite(b) && isfinite(a_lin) && isfinite(b_lin) && abs(a_lin) >= opt.MinVonSlope
            f = @(v) 10.^(a*v + b) - (a_lin*v + b_lin);
            xmin = min(Vg); xmax = max(Vg);
            vgrid = linspace(xmin-0.5, xmax+0.5, 400);
            ygrid = f(vgrid);
            i = find(~isnan(ygrid(1:end-1)) & ~isnan(ygrid(2:end)) & sign(ygrid(1:end-1)) ~= sign(ygrid(2:end)), 1, 'first');
            if ~isempty(i)
                try
                    Vth_extrap_cross = fzero(f, [vgrid(i), vgrid(i+1)]);
                catch
                    Vth_extrap_cross = mean([vgrid(i), vgrid(i+1)]);
                end
            else
                [~, j] = min(abs(ygrid));
                Vth_extrap_cross = vgrid(j);
            end
        end

        % Save metrics
        PerCurve(k).Vd = Vd_val;
        PerCurve(k).Ioff_est = Ioff_est;
        PerCurve(k).Vg_win = Vg_win;
        PerCurve(k).idx_win = idx_win;
        PerCurve(k).slope_dec_per_V = a;
        PerCurve(k).intercept_dec = b;
        PerCurve(k).SS_mV_dec = SS_mV_dec;
        PerCurve(k).SS_fit_R2 = R2_SS;
        PerCurve(k).Ion_Ioff_ratio = Ion_Ioff_ratio;
        PerCurve(k).gm_max = gm_max;
        PerCurve(k).Vth = Vth;
        PerCurve(k).Vth_extrap_Ioff = Vth_extrap_Ioff;
        PerCurve(k).Vth_extrap_Iref = Vth_extrap_Iref;
        PerCurve(k).Vth_gmmax = Vth_gmmax;
        PerCurve(k).Vth_extrap_cross = Vth_extrap_cross;
        PerCurve(k).Von = Von;

        % Plot main curve, highlight window, and draw extended fit line
        if opt.DoPlot
            c = colors(mod(k-1,size(colors,1))+1,:);
            % For plotting: keep NaN at compliance/saturation (comp_mask) to break the line;
            % otherwise floor tiny/invalid values to lower y-limit so off-region shows
            plotAbsId = absId; % already has NaNs where comp_mask was true
            floorMask = (~isfinite(plotAbsId) | plotAbsId<=0) & ~comp_mask;
            plotAbsId(floorMask) = opt.YLim(1);
            plot(Vg, plotAbsId, '-', 'Color', c, 'LineWidth', 1.2); hold on;

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

            % Defer ylim for overview until after all curves are drawn
            set(gca,'YScale','log');   % ★ 强制Y轴对数
            grid on;
        end
    end

    if opt.DoPlot
        % Apply dynamic y-limits so off-state currents are fully visible
        if isfinite(yMinOverview) && yMinOverview>0 && yMaxOverview>0
            yl_lo = max(min(opt.YLim), yMinOverview*0.5);
            yl_hi = max(yl_lo*10, yMaxOverview*1.2);
            ylim([yl_lo, yl_hi]);
        else
            ylim(opt.YLim);
        end
        title('|I_d|–V_g with fixed log y and SS window highlighted');
        hold off;

        % Per curve detail - vertical stack
        figs.detail = figure('Name','Per-curve SS window and fit (vertical stack)');
        try
            set(figs.detail,'Units','pixels');
            scr = get(0,'ScreenSize');
            baseHeightPerPlot = 220; % pixels per subplot
            margins = 80; % top+bottom margins
            figH = min(scr(4)-100, max(350, baseHeightPerPlot*nC + margins));
            % Match width with the overview figure if available
            figW = 900;
            if ishghandle(figs.overview)
                ovPos = get(figs.overview,'Position');
                figW = ovPos(3);
            end
            pos = get(figs.detail,'Position');
            set(figs.detail,'Position',[pos(1), max(50, scr(4)-figH-50), figW, figH]);
            set(figs.detail,'PaperPositionMode','auto');
        catch
        end
        
        % Calculate common x-axis ranges separately: left(SS) and right(Von)
        all_Vg_win_ss = [];
        all_Vg_von_min = [];
        all_Vg_von_max = [];
        all_Von_vals = [];
        for k = 1:nC
            if ~isempty(PerCurve(k).Vg_win) && all(isfinite(PerCurve(k).Vg_win))
                all_Vg_win_ss = [all_Vg_win_ss, PerCurve(k).Vg_win];
            end
            % Estimate Von fit Vg span from high-current band (exclude compliance)
            Vg_k = curves(k).Vg(:); Id_k = curves(k).Id(:);
            cmask_k = false(size(abs(Id_k)));
            if isfinite(complianceA) && complianceA > 0
                cmask_k = abs(Id_k) >= opt.ComplianceFrac * complianceA;
            end
            valid_lin_k = isfinite(Id_k) & isfinite(Vg_k) & ~cmask_k;
            if any(valid_lin_k)
                Id_valid = Id_k(valid_lin_k); Vg_valid = Vg_k(valid_lin_k);
                prc = 80; thr_lin = prctile(abs(Id_valid), prc);
                sel = abs(Id_valid) >= thr_lin;
                if nnz(sel) >= 4
                    all_Vg_von_min(end+1) = min(Vg_valid(sel)); %#ok<AGROW>
                    all_Vg_von_max(end+1) = max(Vg_valid(sel)); %#ok<AGROW>
                end
            end
            if isfield(PerCurve, 'Von') && isfinite(PerCurve(k).Von)
                all_Von_vals(end+1) = PerCurve(k).Von; %#ok<AGROW>
            end
        end
        if ~isempty(all_Vg_win_ss)
            x_range_ss = [min(all_Vg_win_ss) - opt.PadVg, max(all_Vg_win_ss) + opt.PadVg];
        else
            x_range_ss = [-1, 1]; % fallback
        end
        if ~isempty(all_Vg_von_min)
            lo_v = min(all_Vg_von_min);
            hi_v = max(all_Vg_von_max);
            if ~isempty(all_Von_vals)
                lo_v = min(lo_v, min(all_Von_vals));
                hi_v = max(hi_v, max(all_Von_vals));
            end
            x_range_von = [lo_v - opt.PadVg, hi_v + opt.PadVg];
        else
            x_range_von = x_range_ss; % fallback to SS range
        end
        
        % Three-column layout per curve: left=SS (log), middle=Vth extrapolation (log), right=Von (linear)
        tl = tiledlayout(nC, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        for k = 1:nC
            % Left tile: SS fit (log plot)
            nexttile; hold on; grid on;
            Vg = curves(k).Vg(:); absId = abs(curves(k).Id(:));
            % Reapply compliance masking for detail plot consistency
            comp_mask = false(size(absId));
            if isfinite(complianceA) && complianceA > 0
                comp_mask = absId >= opt.ComplianceFrac * complianceA;
            end
            absId(comp_mask) = NaN; % keep NaN to break lines at compliance
            % Do not replace NaNs for plotting; create a plotting copy that preserves NaNs
            plotAbsId = absId; % already has NaNs where comp_mask was true
            floorMask = (~isfinite(plotAbsId) | plotAbsId<=0) & ~comp_mask;
            plotAbsId(floorMask) = opt.YLim(1);
            plot(Vg, plotAbsId, '-', 'LineWidth', 1.0);

            idx_win = PerCurve(k).idx_win;
            if ~isempty(idx_win)
            plot(Vg(idx_win), absId(idx_win), 'o', ...
                'MarkerFaceColor',[.9 .3 .2], 'MarkerEdgeColor','k', 'MarkerSize', 5);

            % fit line
            a = PerCurve(k).slope_dec_per_V; b = PerCurve(k).intercept_dec;
            Vg_win = PerCurve(k).Vg_win;
            Vg_fit = linspace(Vg_win(1)-opt.PadVg, Vg_win(2)+opt.PadVg, 200);
            plot(Vg_fit, 10.^(a*Vg_fit + b), '--', 'LineWidth', 1.2);
            end

            ylim(opt.YLim);
            set(gca,'YScale','log');   % ★ 强制Y轴对数
            xlim(x_range_ss);  % 统一左列X轴范围
            set(gca, 'YTickLabel', []); % hide y tick labels
            % Titles removed; hide x tick labels except last subplot
            if k < nC
                set(gca, 'XTickLabel', []);
            else
                xlabel('V_g (V)');
            end
            % Annotate SS value
            try
                txtSS = sprintf('SS = %.1f mV/dec', PerCurve(k).SS_mV_dec);
                text(0.98, 0.95, txtSS, 'Units','normalized', 'HorizontalAlignment','right', 'VerticalAlignment','top', 'FontSize', 8);
            catch
            end
            hold off;

            % Middle tile: Vth by extrapolation (semi-log)
            nexttile; hold on; grid on;
            Vg = curves(k).Vg(:); absId = abs(curves(k).Id(:));
            comp_mask = false(size(absId));
            if isfinite(complianceA) && complianceA > 0
                comp_mask = absId >= opt.ComplianceFrac * complianceA;
            end
            absId(comp_mask) = NaN;
            plotAbsId2 = absId; plotAbsId2(~isfinite(plotAbsId2) | plotAbsId2<=0) = opt.YLim(1);
            semilogy(Vg, plotAbsId2, '-', 'LineWidth', 1.0);
            set(gca,'YScale','log'); ylim(opt.YLim); xlim(x_range_ss);
            set(gca, 'YTickLabel', []); % hide y tick labels
            % Draw SS fit line used (reuse a,b and Vg_win) and linear fit with intersection
            a = PerCurve(k).slope_dec_per_V; b = PerCurve(k).intercept_dec; Vg_win = PerCurve(k).Vg_win;
            if isfinite(a) && isfinite(b) && all(isfinite(Vg_win))
                Vg_fitE = linspace(Vg_win(1)-opt.PadVg, Vg_win(2)+opt.PadVg, 200);
                plot(Vg_fitE, 10.^(a*Vg_fitE + b), '--', 'LineWidth', 1.2);
                % If we have a valid linear fit from Von stage, overlay it and intersection
                if isfinite(PerCurve(k).Von)
                    % reconstruct linear fit on same high-current region for display
                    Id_lin = curves(k).Id(:); Vg_lin = curves(k).Vg(:);
                    cmask = false(size(abs(Id_lin)));
                    if isfinite(complianceA) && complianceA > 0
                        cmask = abs(Id_lin) >= opt.ComplianceFrac * complianceA;
                    end
                    valid_lin = isfinite(Id_lin) & isfinite(Vg_lin) & ~cmask;
                    if any(valid_lin)
                        Id_valid = Id_lin(valid_lin); Vg_valid = Vg_lin(valid_lin);
                        prc = 80; thr_lin = prctile(abs(Id_valid), prc);
                        sel = abs(Id_valid) >= thr_lin;
                        if nnz(sel) >= opt.MinVonPoints
                            X = [Vg_valid(sel), ones(nnz(sel),1)];
                            coef_lin2 = X \ Id_valid(sel);
                            a_lin2 = coef_lin2(1); b_lin2 = coef_lin2(2);
                            Vg_fitL = linspace(min(Vg_valid(sel)), max(Vg_valid(sel)), 100);
                            plot(Vg_fitL, 10.^(a*Vg_fitL + b), '--', 'LineWidth', 1.0); % SS again to align
                            plot(Vg_fitL, a_lin2*Vg_fitL + b_lin2, ':', 'LineWidth', 1.2);
                            if isfinite(PerCurve(k).Vth_extrap_cross)
                                % mark intersection on this middle plot at Id from either side (choose SS)
                                Id_cross = 10.^(a*PerCurve(k).Vth_extrap_cross + b);
                                plot(PerCurve(k).Vth_extrap_cross, Id_cross, 'o', 'MarkerSize',5, 'MarkerFaceColor',[0.7 0.2 0.9], 'MarkerEdgeColor','k');
                            end
                        end
                    end
                end
                % Draw Iref lines and intersection markers
                Iref1 = clamp(opt.IrefFixed, opt.YLim(1), opt.YLim(2));
                yline(Iref1, ':');
                if isfinite(PerCurve(k).Vth_extrap_Iref)
                    plot(PerCurve(k).Vth_extrap_Iref, Iref1, 's', 'MarkerSize',5,'MarkerFaceColor',[0.2 0.7 0.2], 'MarkerEdgeColor','k');
                end
                Iref2 = clamp(PerCurve(k).Ioff_est*10, opt.YLim(1), opt.YLim(2));
                yline(Iref2, ':');
                if isfinite(PerCurve(k).Vth_extrap_Ioff)
                    plot(PerCurve(k).Vth_extrap_Ioff, Iref2, 'd', 'MarkerSize',5,'MarkerFaceColor',[0.9 0.6 0.1], 'MarkerEdgeColor','k');
                end
            end
            if k < nC
                set(gca, 'XTickLabel', []);
            else
                xlabel('V_g (V)');
            end
            % Annotate Vth values (two lines, three significant digits): extrap-cross and gm-max
            try
                if isfinite(PerCurve(k).Vth_extrap_cross)
                    line1 = sprintf('Vth(extrap)=%.3g V', PerCurve(k).Vth_extrap_cross);
                else
                    line1 = 'Vth(extrap)=N/A';
                end
                if isfinite(PerCurve(k).Vth_gmmax)
                    line2 = sprintf('Vth(gm-max)=%.3g V', PerCurve(k).Vth_gmmax);
                else
                    line2 = 'Vth(gm-max)=N/A';
                end
                text(0.98, 0.95, line1, 'Units','normalized', 'HorizontalAlignment','right', 'VerticalAlignment','top', 'FontSize', 8);
                text(0.98, 0.82, line2, 'Units','normalized', 'HorizontalAlignment','right', 'VerticalAlignment','top', 'FontSize', 8);
            catch
            end
            hold off;

            % Right tile: Von linear fit plot (Id vs Vg in linear scale)
            nexttile; hold on; grid on;
            Vg = curves(k).Vg(:); Id_lin = curves(k).Id(:);
            comp_mask = false(size(abs(Id_lin)));
            if isfinite(complianceA) && complianceA > 0
                comp_mask = abs(Id_lin) >= opt.ComplianceFrac * complianceA;
            end
            Id_plot = Id_lin; Id_plot(comp_mask) = NaN; % break at compliance
            plot(Vg, Id_plot, '-', 'LineWidth', 1.0);
            set(gca, 'YTickLabel', []); % hide y tick labels
            xlim(x_range_von); % 统一右列X轴范围
            % Draw linear fit used for Von with backward extension and intersection (only if quality criteria passed)
            Von = PerCurve(k).Von;
            if isfinite(Von)
                valid_lin = isfinite(Id_lin) & isfinite(Vg) & ~comp_mask;
                Id_valid = Id_lin(valid_lin); Vg_valid = Vg(valid_lin);
                prc = 80; thr_lin = prctile(abs(Id_valid), prc);
                sel = abs(Id_valid) >= thr_lin;
                if nnz(sel) >= opt.MinVonPoints && (max(Vg_valid(sel))-min(Vg_valid(sel))) >= opt.MinVonDeltaVg
                    X = [Vg_valid(sel), ones(nnz(sel),1)];
                    coef_lin = X \ Id_valid(sel);
                    a_lin = coef_lin(1); b_lin = coef_lin(2);
                    if isfinite(a_lin) && abs(a_lin) >= opt.MinVonSlope
                    % Fit segment range
                    Vg_fit2 = linspace(min(Vg_valid(sel)), max(Vg_valid(sel)), 100);
                    plot(Vg_fit2, a_lin*Vg_fit2 + b_lin, '--', 'LineWidth', 1.4);
                    % Backward extension to x-axis
                    Vext = linspace(min([Von, Vg_fit2(1)]), max([Von, Vg_fit2(end)]), 200);
                    plot(Vext, a_lin*Vext + b_lin, ':', 'LineWidth', 1.0);
                    % Mark intersection point (Von, 0)
                    plot(Von, 0, 'o', 'MarkerSize', 5, 'MarkerFaceColor', [0.1 0.5 0.9], 'MarkerEdgeColor', 'k');
                    end
                end
            end
            if k < nC
                set(gca, 'XTickLabel', []);
            else
                xlabel('V_g (V)');
            end
            % Annotate Von value
            try
                if isfinite(PerCurve(k).Von)
                    txtVon = sprintf('Von = %.3f V', PerCurve(k).Von);
                else
                    txtVon = 'Von = N/A';
                end
                text(0.98, 0.95, txtVon, 'Units','normalized', 'HorizontalAlignment','right', 'VerticalAlignment','top', 'FontSize', 8);
            catch
            end
            hold off;
        end

        % Third figure: trends vs Vd
        figs.trends = figure('Name','Trends vs V_d');
        set(figs.trends,'Units','pixels');
        try
            tiledlayout(2,3, 'TileSpacing','compact', 'Padding','compact');
        catch
            tiledlayout(2,3);
        end
        Vd_col = arrayfun(@(x)x.Vd, PerCurve).';
        % 1) SS vs Vd
        nexttile; hold on; grid on;
        plot(Vd_col, arrayfun(@(x)x.SS_mV_dec, PerCurve).', 'o-','LineWidth',1.2);
        xlabel('V_d (V)'); ylabel('SS (mV/dec)'); title('SS vs V_d');
        % 2) Vth (extrap & gmmax) vs Vd
        nexttile; hold on; grid on;
        vth_ex = arrayfun(@(x)x.Vth_extrap_cross, PerCurve).';
        vth_gm = arrayfun(@(x)x.Vth_gmmax, PerCurve).';
        plot(Vd_col, vth_ex, 's-','LineWidth',1.2);
        plot(Vd_col, vth_gm, 'd-','LineWidth',1.2);
        legend({'Vth(extrap)','Vth(gm-max)'}, 'Location','best');
        xlabel('V_d (V)'); ylabel('V_{th} (V)'); title('V_{th} vs V_d');
        % 3) Von vs Vd
        nexttile; hold on; grid on;
        plot(Vd_col, arrayfun(@(x)x.Von, PerCurve).', 'o-','LineWidth',1.2);
        xlabel('V_d (V)'); ylabel('V_{on} (V)'); title('V_{on} vs V_d');
        % 4) gm_{max} vs Vd
        nexttile; hold on; grid on;
        plot(Vd_col, arrayfun(@(x)x.gm_max, PerCurve).', 'o-','LineWidth',1.2);
        xlabel('V_d (V)'); ylabel('g_m^{max} (A/V)'); title('g_m^{max} vs V_d');
        % 5) Ion/Ioff vs Vd (log y)
        nexttile; hold on; grid on;
        yy = arrayfun(@(x)x.Ion_Ioff_ratio, PerCurve).';
        plot(Vd_col, yy, 'o-','LineWidth',1.2);
        set(gca,'YScale','log'); ylabel('Ion/Ioff'); xlabel('V_d (V)'); title('Ion/Ioff vs V_d');
        % 6) Slope (dec/V) vs Vd
        nexttile; hold on; grid on;
        plot(Vd_col, arrayfun(@(x)x.slope_dec_per_V, PerCurve).', 'o-','LineWidth',1.2);
        xlabel('V_d (V)'); ylabel('slope (dec/V)'); title('Slope vs V_d');
        hold off;
    end

    % -------- Summary table --------
    Vd_col = arrayfun(@(x)x.Vd, PerCurve).';
    SS_col = arrayfun(@(x)x.SS_mV_dec, PerCurve).';
    VgL = arrayfun(@(x)x.Vg_win(1), PerCurve).';
    VgR = arrayfun(@(x)x.Vg_win(2), PerCurve).';
    Ioff_col = arrayfun(@(x)x.Ioff_est, PerCurve).';
    Slope_col = arrayfun(@(x)x.slope_dec_per_V, PerCurve).';
    Ion_Ioff_col = arrayfun(@(x)x.Ion_Ioff_ratio, PerCurve).';
    gm_max_col = arrayfun(@(x)x.gm_max, PerCurve).';
    Vth_col = arrayfun(@(x)x.Vth, PerCurve).';
    Vth_Ioff_col = arrayfun(@(x)x.Vth_extrap_Ioff, PerCurve).';
    Vth_Iref_col = arrayfun(@(x)x.Vth_extrap_Iref, PerCurve).';
    Vth_cross_col = arrayfun(@(x)x.Vth_extrap_cross, PerCurve).';
    Vth_gmmax_col = arrayfun(@(x)x.Vth_gmmax, PerCurve).';
    
    Von_col = arrayfun(@(x)x.Von, PerCurve).';
    if isfield(PerCurve, 'SS_fit_R2')
        SS_fit_R2_col = arrayfun(@(x)x.SS_fit_R2, PerCurve).';
    else
        SS_fit_R2_col = NaN(nC, 1);
    end
    if isfield(PerCurve, 'Von_fit_R2')
        Von_fit_R2_col = arrayfun(@(x)x.Von_fit_R2, PerCurve).';
    else
        Von_fit_R2_col = NaN(nC, 1);
    end
    res.PerCurve = PerCurve;
    res.Summary = table(Vd_col, SS_col, SS_fit_R2_col, VgL, VgR, Ioff_col, Slope_col, Ion_Ioff_col, gm_max_col, Vth_col, Vth_Ioff_col, Vth_Iref_col, Vth_cross_col, Vth_gmmax_col, Von_col, Von_fit_R2_col, ...
        'VariableNames', {'Vd','SS_mV_per_dec','SS_fit_R2','Vg_left','Vg_right','Ioff_est','slope_dec_per_V','Ion_Ioff_ratio','gm_max','Vth','Vth_extrap_Ioff','Vth_extrap_Iref','Vth_extrap_cross','Vth_gmmax','Von','Von_fit_R2'});
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
    % Linear fit y = a*x + b. Use robustfit if available; fallback to \ on failure or warning.
    X = [ones(numel(x),1), x(:)];
    y = y(:);
    if exist('robustfit','file')
        try
            warnState = warning('off', 'stats:statrobustfit:IterationLimit');
            coef = robustfit(x, y);
            warning(warnState);
            b = coef(1); a = coef(2);
            if ~isfinite(a) || ~isfinite(b)
                coef = X \ y;
                b = coef(1); a = coef(2);
            end
        catch
            coef = X \ y;
            b = coef(1); a = coef(2);
        end
    else
        coef = X \ y;
        b = coef(1); a = coef(2);
    end
end

function v = clamp(x, lo, hi)
    v = min(max(x, lo), hi);
end
