function res = analyze_output_curves(data, varargin)
% ANALYZE_OUTPUT_CURVES  Analyze Id–Vd output characteristics.
% Accepts:
%   - The struct returned by import_clarius_csv (preferred), or
%   - A table with variables Vd, Id (and optional Vg), or
%   - Numeric arrays (Vd, Id [, VgVector or VgValue])
%
% Name-Value parameters (all optional):
%   "VdLinMax"   : 0.1     % linear-region |Vd| upper bound for Ron fit
%   "VdSatFrac"  : 0.6     % saturation-region start as fraction of max |Vd|
%   "DoPlot"     : true    % draw figures with linear y-axis for output curves
%   "WLCOX"      : []      % [W, L, Cox] (meters, meters, F/m^2), enables quad-fit
%   "UseQuadFit" : false   % enable small-Vd quadratic fit to estimate mu & Vth
%   "MinPtsLin"  : 5       % minimum points required in linear fit
%   "MinPtsSat"  : 5       % minimum points required in saturation fit
%
% Returns struct:
%   .SweepInfo: VgList, hasVg, isMulti
%   .PerVg(i): struct of per-curve metrics:
%       Vg, Ron, dId_dVd_lin, gds_sat, ro, Id_sat, lambda, VA, Vds_knee_est
%       [optional when WLCOX given]: mu, Vth, A=-2b, Vgt=a/A
%   .Summary: table consolidated over Vg
%   .Fits(i): struct with fit details and masks
%   .Figures: figure handles (if DoPlot)
%
% Notes:
%   - Linear region fit: |Vd| <= VdLinMax, robustfit
%   - Saturation region fit: |Vd| >= VdSatFrac * max(|Vd|)
%   - lambda ~ gds_sat / Id_sat (Id_sat 取饱和段末端的平均)
%   - VA ~ 1/lambda
%   - Vds_knee_est: 基于曲率与线性外推差异的启发式估计
%   - Quadratic fit (optional): Id ~ a*Vd + b*Vd^2, with A = -2b = mu*Cox*W/L,
%                               Vgt = a/A, Vth = Vg - Vgt, mu = A/(Cox*W/L)

    %---------------- Parse options ----------------
    p = inputParser;
    addParameter(p, "VdLinMax", 0.1, @(x)isnumeric(x)&&isscalar(x)&&x>0);
    addParameter(p, "VdSatFrac", 0.6, @(x)isnumeric(x)&&isscalar(x)&&x>0&&x<1);
    addParameter(p, "DoPlot", true, @(x)islogical(x)||ismember(x,["auto","true","false"]));
    addParameter(p, "WLCOX", [], @(x)isnumeric(x)&&((isempty(x))||(numel(x)==3)));
    addParameter(p, "UseQuadFit", false, @(x)islogical(x));
    addParameter(p, "MinPtsLin", 5, @(x)isnumeric(x)&&isscalar(x)&&x>=2);
    addParameter(p, "MinPtsSat", 5, @(x)isnumeric(x)&&isscalar(x)&&x>=2);
    parse(p, varargin{:});
    opt = p.Results;

    figs = struct('overview', [], 'fits', [], 'metrics', []);
    per = [];  fits = [];

    %---------------- Normalize input ----------------
    M = []; T = []; hasVg = false;
    if isstruct(data) && isfield(data, 'Data') && isfield(data.Data, 'Matrix')
        M = data.Data.Matrix;
        % Expect fields: Vd, Id, optionally Vg
        if ~isfield(M,'Vd') || ~isfield(M,'Id')
            error('Matrix must contain Vd and Id.');
        end
        Vd = M.Vd; Id = M.Id;
        [N1, N2] = size(Vd); %#ok<ASGLU>
        if isfield(M,'Vg')
            VgMat = M.Vg;
            hasVg = true;
            % 取外层 sweep 作为列索引的恒定值
            if size(VgMat,2)==size(Vd,2)
                VgList = VgMat(1,:);
            else
                % 尝试取每列众数
                VgList = arrayfun(@(j) mode(round(VgMat(:,j),6)), 1:size(Vd,2));
            end
        else
            VgList = 0;  % 单曲线视为一条 VG 标签
        end
        isMulti = size(Vd,2) > 1;
    elseif istable(data)
        T = data;
        mustVars = {'Vd','Id'};
        for k=1:numel(mustVars)
            if ~ismember(mustVars{k}, T.Properties.VariableNames)
                error('Table must contain variable %s.', mustVars{k});
            end
        end
        if ismember('Vg', T.Properties.VariableNames)
            hasVg = true;
            % 将表拆分为不同 Vg 的列
            [VgLevels,~,grp] = unique(T.Vg(:));
            n = numel(VgLevels);
            Vd = []; Id = [];
            for j=1:n
                idx = (grp==j);
                vvd = T.Vd(idx); iid = T.Id(idx);
                % 按 Vd 排序
                [vvd, ord] = sort(vvd); iid = iid(ord);
                if j==1
                    Vd = vvd(:);
                    Id = nan(numel(vvd), n);
                end
                Id(:,j) = interp1(vvd, iid, Vd, 'linear', 'extrap');
            end
            VgList = VgLevels(:).';
            isMulti = n>1;
        else
            % 单曲线
            [Vd, ord] = sort(T.Vd(:)); Id = T.Id(ord);
            Vd = Vd(:); Id = Id(:);
            Vd = Vd(:); Id = Id(:);
            Vd = reshape(Vd, [], 1); Id = reshape(Id, [], 1);
            VgList = 0; isMulti = false;
        end
    else
        error('Unsupported input type. Pass importer struct or a table with Vd, Id [, Vg].');
    end

    % Ensure matrices
    if ~ismatrix(Vd) || ~ismatrix(Id)
        error('Vd and Id must be vectors or 2D matrices.');
    end
    if ~isMulti && size(Id,2)~=1
        % flatten to single column (single curve)
        Id = Id(:); Vd = Vd(:);
    end

    %---------------- Prepare analysis ----------------
    nCurves = size(Id,2);
    per = repmat(struct( ...
        'Vg', NaN, ...
        'Ron', NaN, 'dId_dVd_lin', NaN, ...
        'gds_sat', NaN, 'ro', NaN, ...
        'Id_sat', NaN, 'lambda', NaN, 'VA', NaN, ...
        'Vds_knee_est', NaN, ...
        'mu', NaN, 'Vth', NaN, 'A', NaN, 'Vgt', NaN ...
    ), 1, nCurves);

    fits = repmat(struct( ...
        'idx_lin', [], 'idx_sat', [], ...
        'lin_fit', [], 'sat_fit', [], ...
        'quad_fit', [], 'Vd', [], 'Id', [] ...
    ), 1, nCurves);

    % helper: robust linear fit y = m*x + c
    function [m,c] = robust_line(x,y)
        if exist('robustfit','file')
            b = robustfit(x,y); c = b(1); m = b(2);
        else
            X = [ones(numel(x),1), x(:)];
            b = X\y(:); c = b(1); m = b(2);
        end
    end

    %---------------- Per-curve analysis ----------------
    for j = 1:nCurves
        vd = Vd(:, min(j,size(Vd,2)));
        id = Id(:, j);
        fits(j).Vd = vd; fits(j).Id = id;

        % 排序（以防原始顺序非单调）
        [vd, ord] = sort(vd); id = id(ord);

        % 线性区掩码
        idx_lin = abs(vd) <= opt.VdLinMax;
        if sum(idx_lin) < opt.MinPtsLin
            % 放宽到前/后若干点
            k = min(opt.MinPtsLin, numel(vd));
            idx_lin = false(size(vd)); idx_lin(1:k) = true;
        end
        fits(j).idx_lin = idx_lin;

        % 饱和区掩码
        vd_abs = abs(vd);
        vd_max = max(vd_abs);
        thr = opt.VdSatFrac * vd_max;
        idx_sat = vd_abs >= thr;
        if sum(idx_sat) < opt.MinPtsSat
            % 取末尾 MinPtsSat
            idx_sat = false(size(vd));
            idx_sat(end-opt.MinPtsSat+1:end) = true;
        end
        fits(j).idx_sat = idx_sat;

        % 线性区线性拟合 (Ron, dId/dVd)
        [m_lin, c_lin] = robust_line(vd(idx_lin), id(idx_lin)); %#ok<ASGLU>
        dId_dVd_lin = m_lin;
        Ron = 1 / dId_dVd_lin;
        per(j).dId_dVd_lin = dId_dVd_lin;
        per(j).Ron = Ron;

        % 饱和区线性拟合 (gds, ro, lambda, VA)
        [m_sat, c_sat] = robust_line(vd(idx_sat), id(idx_sat));
        gds_sat = m_sat;
        ro = 1 / gds_sat;
        Id_sat = mean(id(idx_sat), 'omitnan');
        lambda = gds_sat / max(abs(Id_sat), eps);
        VA = 1 / max(lambda, eps);
        per(j).gds_sat = gds_sat;
        per(j).ro = ro;
        per(j).Id_sat = Id_sat;
        per(j).lambda = lambda;
        per(j).VA = VA;
        fits(j).lin_fit = struct('m', m_lin, 'c', c_lin);
        fits(j).sat_fit = struct('m', m_sat, 'c', c_sat);

        % 拐点/估计 Vdsat：用线性外推与真实曲线差的最大处
        y_lin_pred = m_lin .* vd + c_lin;
        [~, iknee] = max(abs(id - y_lin_pred));
        per(j).Vds_knee_est = vd(iknee);

        % 小Vd二次拟合（可选，需 WLCOX）
        if opt.UseQuadFit && ~isempty(opt.WLCOX)
            idx_q = idx_lin;  % 在小Vd区做二次拟合
            x = vd(idx_q); y = id(idx_q);
            if numel(x) >= 3
                % 拟合 y = a*x + b*x^2
                A2 = [x, x.^2];
                coef = A2 \ y;
                a = coef(1); b = coef(2);
                A = -2*b;  % A = mu*Cox*W/L
                W = opt.WLCOX(1); L = opt.WLCOX(2); Cox = opt.WLCOX(3);
                if A > 0 && Cox>0 && W>0 && L>0
                    mu = A * L / (Cox * W);
                    Vgt = a / A;
                    % 若有 Vg 标签：
                    if isvector_hasVg(hasVg, VgList, j)
                        Vg_val = VgList(min(j, numel(VgList)));
                        Vth = Vg_val - Vgt;
                    else
                        Vth = NaN;
                    end
                    per(j).mu = mu;
                    per(j).Vth = Vth;
                    per(j).A   = A;
                    per(j).Vgt = Vgt;
                end
                fits(j).quad_fit = struct('a',a,'b',b);
            end
        end

        % 记录 Vg
        if isvector_hasVg(hasVg, VgList, j)
            per(j).Vg = VgList(min(j, numel(VgList)));
        else
            per(j).Vg = NaN;
        end
    end

    % 汇总表
    Summary = struct2table(per);
    res = struct();
    res.SweepInfo = struct('VgList', VgList(:).', 'hasVg', hasVg, 'isMulti', isMulti);
    res.PerVg = per;
    res.Summary = Summary;
    res.Fits = fits;

    %---------------- Plotting (linear y-axis) ----------------
    if istrue(opt.DoPlot)
        try
            figs.overview = figure('Name','Output Curves (Id–Vd, linear y)');
            if isMulti
                plot(Vd, Id, 'LineWidth', 1.1); grid on;
                xlabel('V_d (V)'); ylabel('I_d (A)');
                if isvector_hasVg(hasVg, VgList, 1)
                    leg = arrayfun(@(x) sprintf('V_g=%.3g V', x), VgList, 'UniformOutput', false);
                    legend(leg, 'Location','best');
                end
                title('Output Curves');
            else
                plot(Vd(:), Id(:), 'LineWidth',1.3); grid on;
                xlabel('V_d (V)'); ylabel('I_d (A)');
                title('Output Curve (single)');
            end

            % 拟合展示
            figs.fits = figure('Name','Fits: linear & saturation regions');
            tiledlayout('flow');
            for j=1:nCurves
                nexttile;
                vd = fits(j).Vd; id = fits(j).Id;
                idx_lin = fits(j).idx_lin; idx_sat = fits(j).idx_sat;
                plot(vd, id, '.', 'MarkerSize',8); hold on;
                % 线性拟合线
                m = fits(j).lin_fit.m; c = fits(j).lin_fit.c;
                x1 = linspace(min(vd(idx_lin)), max(vd(idx_lin)), 100);
                plot(x1, m*x1 + c, '-', 'LineWidth',1.2);
                % 饱和拟合线
                m2 = fits(j).sat_fit.m; c2 = fits(j).sat_fit.c;
                x2 = linspace(min(vd(idx_sat)), max(vd(idx_sat)), 100);
                plot(x2, m2*x2 + c2, '-', 'LineWidth',1.2);
                % knee
                xk = per(j).Vds_knee_est;
                yk = m*xk + c;
                xline(xk, '--');
                grid on; xlabel('V_d (V)'); ylabel('I_d (A)');
                ttl = sprintf('Curve %d', j);
                if isvector_hasVg(hasVg, VgList, j)
                    ttl = sprintf('V_g=%.3g V', VgList(min(j,numel(VgList))));
                end
                title(ttl); hold off;
            end

            % 指标随 Vg
            if isMulti && isvector_hasVg(hasVg, VgList, 1)
                figs.metrics = figure('Name','Metrics vs Vg');
                tl = tiledlayout(2,2,"TileSpacing","compact"); %#ok<NASGU>
                nexttile; plot(Summary.Vg, Summary.Ron, 'o-'); grid on;
                xlabel('V_g (V)'); ylabel('R_{on} (\Omega)'); title('R_{on} vs V_g');
                nexttile; plot(Summary.Vg, Summary.gds_sat, 'o-'); grid on;
                xlabel('V_g (V)'); ylabel('g_{ds} (S)'); title('g_{ds} vs V_g');
                nexttile; plot(Summary.Vg, Summary.ro, 'o-'); grid on;
                xlabel('V_g (V)'); ylabel('r_o (\Omega)'); title('r_o vs V_g');
                nexttile; plot(Summary.Vg, Summary.VA, 'o-'); grid on;
                xlabel('V_g (V)'); ylabel('V_A (V)'); title('V_A vs V_g');
            end
        catch
            % ignore plotting errors (e.g., headless)
        end
    end

    res.Figures = figs;
end

function tf = isvector_hasVg(hasVg, VgList, j)
    tf = hasVg && ~isempty(VgList) && numel(VgList)>=j;
end

function tf = istrue(x)
    if islogical(x), tf = x; return; end
    if isstring(x)
        tf = any(x==["auto","true"]);  % treat "auto" as true
    else
        tf = logical(x);
    end
end
