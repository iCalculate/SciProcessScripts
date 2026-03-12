% IMPORT_DATAMATRIX_AND_ANALYZE  从 dataMatrix.xlsx 导入数据并提取各器件转移曲线参数
%
% Excel 格式：
%   - 第一列：Vg（栅压）
%   - 第二列起：每列为一个器件的 Id 数据（与第一列 Vg 一一对应）
%
% 数据处理与参数提取直接调用 analyze_transfer_curves.m。
%
% 用法：
%   res = import_dataMatrix_and_analyze();
%   res = import_dataMatrix_and_analyze('dataMatrix.xlsx');
%   res = import_dataMatrix_and_analyze('dataMatrix.xlsx', 'DoPlot', false);

function res = import_dataMatrix_and_analyze(excelFile, varargin)

    % 默认 Excel 文件名（与脚本同目录或当前目录）
    if nargin < 1 || isempty(excelFile)
        excelFile = 'dataMatrix.xlsx';
    end
    excelFile = char(excelFile);

    % 若为相对路径，优先在脚本所在目录查找
    if isempty(fileparts(excelFile))
        scriptDir = fileparts(mfilename('fullpath'));
        candidate = fullfile(scriptDir, excelFile);
        if isfile(candidate)
            excelFile = candidate;
        end
    end

    if ~isfile(excelFile)
        error('找不到文件: %s', excelFile);
    end

    % 确保能调用同目录下的 analyze_transfer_curves
    scriptDir = fileparts(mfilename('fullpath'));
    if isempty(which('analyze_transfer_curves'))
        addpath(scriptDir);
    end

    % -------- 导入 Excel --------
    % 第一列 -> Vg，其余列 -> 每列一个器件的 Id
    T = readtable(excelFile, 'VariableNamingRule', 'preserve');

    if size(T, 2) < 2
        error('Excel 至少需要两列：第一列 Vg，第二列起为各器件 Id。');
    end

    % 第一列 -> Vg（readtable 默认第一行为表头，数据从第二行起）
    Vg = T{:, 1};
    Vg = Vg(:);
    % 若表中含非数值（如表头被读入），只保留数值行
    validRow = isfinite(Vg);
    if ~all(validRow)
        Vg = Vg(validRow);
        T = T(validRow, :);
    end

    % 第二列到最后一列：每列一个器件的 Id
    Id_matrix = T{:, 2:end};
    if isvector(Id_matrix)
        Id_matrix = Id_matrix(:);
    end

    nPts = numel(Vg);
    if size(Id_matrix, 1) ~= nPts
        error('Vg 行数 (%d) 与 Id 数据行数 (%d) 不一致。', nPts, size(Id_matrix, 1));
    end
    nDevices = size(Id_matrix, 2);

    % -------- 构造成 analyze_transfer_curves 可接受的 struct 格式 --------
    % 格式与 import_clarius_csv(FileType="transfer") 的 .Data.Matrix 一致：
    %   Vg: nPts x 1（所有器件共用同一 Vg）
    %   Id: nPts x nDevices（每列一个器件）
    %   Vd: 无测量时用 NaN
    data = struct();
    data.Data = struct();
    data.Data.Matrix = struct();
    data.Data.Matrix.Vg = Vg;
    data.Data.Matrix.Id = Id_matrix;
    data.Data.Matrix.Vd = NaN(nPts, nDevices);

    % -------- 调用转移曲线分析与参数提取 --------
    res = analyze_transfer_curves(data, varargin{:});

    % -------- 简要输出：各器件参数汇总表 --------
    if nargout < 1
        disp('===== 各器件参数汇总 (res.Summary) =====');
        disp(res.Summary);
    end
end
