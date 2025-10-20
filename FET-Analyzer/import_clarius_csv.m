function out = import_clarius_csv(filename, varargin)
% IMPORT_CLARIUS_CSV  Import Keysight B1500/Clarius CSV (transfer/output/single).
%
% out = import_clarius_csv(file, Name,Value,...)
%
% Name-Value options:
%   FileType   : "auto" (default) | "transfer" | "output_series" | "output_single"
%   Plot       : true | false | "auto" (default: true unless headless)
%   ParseHeaders : true (default) | false
%   InnerVar   : "auto" (default) | "Vg" | "Vd"    % force inner sweep variable
%   OuterVar   : "auto" (default) | "Vg" | "Vd"    % force outer sweep variable
%
% Returns struct with:
%   .SetupTitle, .PrimitiveTest, .TestParameter, .ParameterSetup, .AnalysisSetup
%   .Dimensions.N1 (inner), .N2 (outer)
%   .Sweep.FileType, .Sweep.InnerVar, .Sweep.OuterVar
%   .Data.Table (raw), .Data.Matrix (reshaped)
%   .FileInfo (file, rows, cols, timestamp)
%
% Plots:
%   transfer      : semilogy Id vs Vg per Vd
%   output_series : semilogy Id vs Vd per Vg
%   output_single : semilogy Id vs Vd (single curve)

    %---------------- Args ----------------
    ip = inputParser;
    addRequired(ip, 'filename', @(s)ischar(s) || (isstring(s)&&isscalar(s)));
    addParameter(ip, 'FileType', "auto", @(s)isstring(s)||ischar(s));
    addParameter(ip, 'Plot', "auto", @(x)islogical(x) || (isstring(x)&&isscalar(x)));
    addParameter(ip, 'ParseHeaders', true, @islogical);
    addParameter(ip, 'InnerVar', "auto", @(s)isstring(s)||ischar(s));
    addParameter(ip, 'OuterVar', "auto", @(s)isstring(s)||ischar(s));
    parse(ip, filename, varargin{:});
    opt = ip.Results;
    opt.FileType = string(lower(opt.FileType));
    opt.InnerVar = string(upper(opt.InnerVar));
    opt.OuterVar = string(upper(opt.OuterVar));

    % Decide plotting default
    doPlot = true;
    if isstring(opt.Plot) && opt.Plot=="auto"
        % Heuristic: if in deployed/headless, skip plotting
        doPlot = usejava('desktop');
    elseif islogical(opt.Plot)
        doPlot = opt.Plot;
    end

    %---------------- Read file ----------------
    raw = fileread(filename);
    raw = strrep(raw, sprintf('\r\n'), sprintf('\n'));
    raw = strrep(raw, sprintf('\r'), sprintf('\n'));
    lines = splitlines(raw);

    makeField = @(s) matlab.lang.makeValidName(strtrim(s));

    % nested assignment utilities
    function S = assignTop(S, parts, value)
        if numel(parts)==1
            S.(parts{1}) = value; return;
        end
        p = parts{1};
        if ~isfield(S,p) || ~isstruct(S.(p)), S.(p)=struct(); end
        S.(p) = assignTop(S.(p), parts(2:end), value);
    end
    function S = setNested(S, keyPath, value)
        if isempty(keyPath), return; end
        parts = strsplit(strtrim(keyPath), '.');
        for k = 1:numel(parts), parts{k}=makeField(parts{k}); end
        S = assignTop(S, parts, value);
    end
    function v = parseValues(tokens)
        tokens = strtrim(tokens);
        if isempty(tokens) || (numel(tokens)==1 && tokens{1}=="")
            v = []; return;
        end
        vnum = nan(size(tokens)); isnum = false(size(tokens));
        for iTok = 1:numel(tokens)
            d = str2double(tokens{iTok});
            if ~isnan(d), vnum(iTok)=d; isnum(iTok)=true; end
        end
        if all(isnum), v = vnum(:).'; else, v = cellstr(tokens); end
    end

    %---------------- Parse header blocks ----------------
    SetupTitle = ""; PrimitiveTest = "";
    TestParameter=struct(); ParameterSetup=struct(); AnalysisSetup=struct();
    Dim1=[]; Dim2=[];
    dataNames = [];
    i = 1; n = numel(lines);
    while i<=n
        line = strtrim(lines{i});
        if line=="", i=i+1; continue; end
        toks = split(line, ','); toks = strip(toks);
        tag = string(toks{1});
        if tag=="DataName"
            dataNames = strip(toks(2:end)); i=i+1; break;
        end
        switch tag
            case "SetupTitle"
                if numel(toks)>=2, SetupTitle = string(strjoin(strip(toks(2:end)), ", ")); end
            case "PrimitiveTest"
                if numel(toks)>=2, PrimitiveTest = string(strjoin(strip(toks(2:end)), ", ")); end
            case "TestParameter"
                if numel(toks)>=3
                    TestParameter = setNested(TestParameter, string(toks{2}), ...
                        parseValues(cellstr(toks(3:end))));
                end
            case "ParameterSetup"
                if numel(toks)>=3
                    ParameterSetup = setNested(ParameterSetup, string(toks{2}), ...
                        parseValues(cellstr(toks(3:end))));
                end
            case "AnalysisSetup"
                if numel(toks)>=3
                    AnalysisSetup = setNested(AnalysisSetup, string(toks{2}), ...
                        parseValues(cellstr(toks(3:end))));
                end
            case "Dimension1"
                if numel(toks)>=2, Dim1 = str2double(toks(2:end)); Dim1=Dim1(~isnan(Dim1)); end
            case "Dimension2"
                if numel(toks)>=2, Dim2 = str2double(toks(2:end)); Dim2=Dim2(~isnan(Dim2)); end
            otherwise
                % ignore
        end
        i=i+1;
    end
    if isempty(dataNames)
        error('DataName header not found — file is not a valid Clarius CSV.');
    end

    %---------------- Read DataValue rows ----------------
    vals=[];
    while i<=n
        line = strtrim(lines{i});
        if startsWith(line,"DataValue")
            toks = split(line, ','); toks=strip(toks);
            row = str2double(toks(2:end));
            vals = [vals; row.']; %#ok<AGROW>
        end
        i=i+1;
    end
    T = array2table(vals, 'VariableNames', cellstr(dataNames));

    %---------------- Canonical variable detection ----------------
    % Case-insensitive map to canonical names (Vg,Vd,Id,Ig,absId,absIg)
    varMap = struct('VG',"", 'VD',"", 'ID',"", 'IG',"", 'ABSID',"", 'ABSIG',"");
    lowerNames = lower(T.Properties.VariableNames);
    for k=1:numel(lowerNames)
        nm = lowerNames{k};
        if     any(strcmp(nm, ["vg","gate","vg_v","v_g","vgs","v_gs"])), varMap.VG = T.Properties.VariableNames{k};
        elseif any(strcmp(nm, ["vd","drain","vd_v","v_d","vds","v_ds"])), varMap.VD = T.Properties.VariableNames{k};
        elseif any(strcmp(nm, ["id","ids","i_d","id_a"])),               varMap.ID = T.Properties.VariableNames{k};
        elseif any(strcmp(nm, ["ig","igs","i_g"])),                       varMap.IG = T.Properties.VariableNames{k};
        elseif any(strcmp(nm, ["absid","|id|","abs(id)","abs_id"])),      varMap.ABSID = T.Properties.VariableNames{k};
        elseif any(strcmp(nm, ["absig","|ig|","abs(ig)","abs_ig"])),      varMap.ABSIG = T.Properties.VariableNames{k};
        end
    end

    % convenience handles (if missing, return [])
    function v = getVar(name)
        if name=="VG" && varMap.VG~="", v = T.(varMap.VG); else
        if name=="VD" && varMap.VD~="", v = T.(varMap.VD); else
        if name=="ID" && varMap.ID~="", v = T.(varMap.ID); else
        if name=="IG" && varMap.IG~="", v = T.(varMap.IG); else
        if name=="ABSID" && varMap.ABSID~="", v = T.(varMap.ABSID); else
        if name=="ABSIG" && varMap.ABSIG~="", v = T.(varMap.ABSIG); else
            v = [];
        end,end,end,end,end,end
    end

    vg = getVar("VG"); vd = getVar("VD"); id = getVar("ID");

    %---------------- Dim inference ----------------
    N1=[]; N2=[];
    if ~isempty(Dim1) && ~isempty(Dim2) && isfinite(Dim1(1)) && isfinite(Dim2(1))
        if Dim1(1)*Dim2(1)==height(T)
            N1=Dim1(1); N2=Dim2(1);
        end
    end

    % Heuristics for sweep order
    function [innerName, outerName, typeGuess] = guessType()
        % Returns canonical names "VG"/"VD" for inner/outer and a type string
        % Strategy: check runs of constant VG and constant VD
        typeGuess = "unknown"; innerName=""; outerName="";
        % single curve?
        if ~isempty(vd) && numel(unique(vd))>1 && (isempty(vg) || numel(unique(vg))==1)
            innerName="VD"; outerName="VG"; typeGuess="output_single";
            return;
        end
        if ~isempty(vg) && ~isempty(vd)
            % compute longest constant run lengths for each variable
            runlen = @(x) max(diff([find([true; diff(x)~=0]); numel(x)+1]));
            lenConstVD = runlen(vd);
            lenConstVG = runlen(vg);
            % If VD is constant over long runs, inner likely VG (transfer)
            if lenConstVD >= lenConstVG
                innerName="VG"; outerName="VD"; typeGuess="transfer";
            else
                innerName="VD"; outerName="VG"; typeGuess="output_series";
            end
        elseif ~isempty(vd)
            innerName="VD"; outerName="VG"; typeGuess="output_single";
        elseif ~isempty(vg)
            innerName="VG"; outerName="VD"; typeGuess="transfer";
        end
    end

    [innerAuto, outerAuto, typeAuto] = guessType();

    % Respect forced type/vars if provided
    fileType = opt.FileType;
    if fileType=="auto" || fileType==""
        fileType = typeAuto;
        if fileType=="unknown", fileType="transfer"; end
    end
    innerVar = opt.InnerVar; outerVar = opt.OuterVar;
    if innerVar=="auto" || innerVar==""
        innerVar = innerAuto;
    end
    if outerVar=="auto" || outerVar==""
        outerVar = outerAuto;
    end

    % If still missing, default by fileType
    if innerVar=="" || outerVar==""
        switch fileType
            case "transfer",      innerVar="VG"; outerVar="VD";
            case "output_series", innerVar="VD"; outerVar="VG";
            case "output_single", innerVar="VD"; outerVar="VG";
            otherwise,            innerVar="VG"; outerVar="VD";
        end
    end

    % Infer N1,N2 if needed by counting constant-outer runs
    function [n1,n2] = inferDims(inner, outer)
        xOuter = getVar(outer);
        if isempty(xOuter)
            n1 = height(T); n2 = 1;
        else
            brk = [true; diff(xOuter)~=0];
            runStarts = find(brk);
            runLens = [runStarts(2:end)-runStarts(1:end-1); numel(xOuter)-runStarts(end)+1];
            n1 = mode(runLens);
            n2 = height(T)/n1;
            if n1*n2 ~= height(T)
                % fallback single column
                n1 = height(T); n2 = 1;
            end
        end
    end

    if isempty(N1) || isempty(N2)
        [N1,N2] = inferDims(innerVar, outerVar);
    end

    %---------------- Reshape matrices ----------------
    reshapeTo = @(v) reshape(v, [N1, N2]);
    M = struct();
    vars = T.Properties.VariableNames;
    for k=1:numel(vars)
        v = T.(vars{k});
        if isnumeric(v)
            M.(makeField(vars{k})) = reshapeTo(v);
        end
    end

    %---------------- Pack output ----------------
    out = struct();
    out.SetupTitle     = SetupTitle;
    out.PrimitiveTest  = PrimitiveTest;
    if opt.ParseHeaders
        out.TestParameter  = TestParameter;
        out.ParameterSetup = ParameterSetup;
        out.AnalysisSetup  = AnalysisSetup;
    else
        out.TestParameter  = struct(); out.ParameterSetup = struct(); out.AnalysisSetup = struct();
    end
    out.Dimensions     = struct('N1', N1, 'N2', N2);
    out.Sweep          = struct('FileType', string(fileType), ...
                                'InnerVar', string(innerVar), ...
                                'OuterVar', string(outerVar));
    out.Data           = struct('Table', T, 'Matrix', M);
    out.FileInfo       = struct('File', string(filename), ...
                                'NumRows', height(T), ...
                                'NumCols', width(T), ...
                                'ParsedOn', string(datetime('now')));

    %---------------- Optional plotting ----------------
        %---------------- Optional plotting ----------------
    if doPlot
        try
            switch string(fileType)
                case "transfer"
                    if isfield(M,'Vg') && isfield(M,'Id')
                        figure('Name','Transfer: Id(Vg) per Vd');
                        semilogy(M.Vg, abs(M.Id));  % log scale for Id
                        grid on;
                        xlabel('V_g (V)'); ylabel('|I_d| (A)');
                        if isfield(M,'Vd')
                            leg = arrayfun(@(x) sprintf('V_d=%.3g V', M.Vd(1,x)), 1:N2, 'UniformOutput', false);
                            legend(leg, 'Location','best');
                        end
                        title('B1500 — Transfer Curves');
                    end

                case "output_series"
                    if isfield(M,'Vd') && isfield(M,'Id')
                        figure('Name','Output: Id(Vd) per Vg');
                        plot(M.Vd, M.Id, 'LineWidth',1.2);   % *** linear y-axis ***
                        grid on;
                        xlabel('V_d (V)'); ylabel('I_d (A)');
                        if isfield(M,'Vg')
                            leg = arrayfun(@(x) sprintf('V_g=%.3g V', M.Vg(1,x)), 1:N2, 'UniformOutput', false);
                            legend(leg, 'Location','best');
                        end
                        title('B1500 — Output Curves (series)');
                    end

                case "output_single"
                    if isfield(M,'Vd') && isfield(M,'Id')
                        figure('Name','Output: single Id(Vd)');
                        plot(M.Vd(:), M.Id(:), 'LineWidth',1.5);   % *** linear y-axis ***
                        grid on;
                        xlabel('V_d (V)'); ylabel('I_d (A)');
                        title('B1500 — Output Curve (single)');
                    end
            end
        catch
            % Ignore plotting errors
        end
    end

end
