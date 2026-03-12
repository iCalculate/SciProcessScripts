% 拉曼功率依赖性分析脚本 - 快速参考
% power_dependence.m Quick Reference Guide

% ========== 基本使用 ==========

% 1. 直接运行（使用预设的MoS₂数据和340-450 cm⁻¹范围）
power_dependence.m

% 2. 修改绘图范围（340-450 cm⁻¹）
ramanPlotRange = [340 450];

% 3. 修改文件路径
ramanFilePath = 'your_raman_data.mat';
powerFilePath = 'your_power_data.mat';

% ========== 输出图形 ==========

% 图1: 等高线图 (功率 vs 拉曼位移)
%   - 显示全范围光谱强度分布
%   - X轴: 激光功率 (mW)
%   - Y轴: 拉曼位移 (cm⁻¹)

% 图2: 堆叠谱线演变 (340-450 cm⁻¹)
%   - 不同功率下的谱线堆叠
%   - 聚焦于指定范围
%   - 便于观察峰形和强度变化

% 图3: 功率依赖性分析
%   左图: 主峰强度 vs 功率 (对数-对数坐标)
%         - 幂律拟合: I = a·P^n
%         - 基于 ramanPlotRange 范围内的峰
%   
%   右图: 规范化谱线演变 (340-450 cm⁻¹)
%         - 8条或更少的谱线
%         - 颜色梯度显示功率变化

% 图4: 详细堆叠图 (340-450 cm⁻¹)
%   - 每3条功率点显示
%   - 高分辨率细节展示

% ========== 关键输出信息 ==========

% Step 1: 数据加载
% 拉曼数据文件: [filepath]
% 功率数据文件: [filepath]
% 数据大小: N × M (N=功率点数, M=拉曼点数)
% 功率范围: X - Y mW
% 拉曼位移范围: A - B cm⁻¹

% Step 2-5: 各分析步骤完成

% 检测光谱主峰:
% 搜索范围: 340 - 450 cm⁻¹
% 主峰位移: X.X cm⁻¹
% 拟合范围: X.X - Y.Y cm⁻¹

% 功率依赖性拟合结果:
% 主峰: I = a × P^n

% ========== 常见调整 ==========

% 改变拉曼绘图范围
% 编辑第17行: ramanPlotRange = [340 450];
% 例: [300 500] - 更宽的范围
%     [380 400] - 更窄的范围

% 改变峰检测宽度 (第112行)
% peakWidth = 30;  % ±30 cm⁻¹
% 例: peakWidth = 20;  % 更窄的峰

% 改变显示谱线数 (第159行)
% numPlots = min(8, ceil(numPowers / 4));
% 例: numPlots = 5;  % 显示5条谱线

% ========== 数据格式示例 ==========

% 拉曼数据文件应包含 (以下任一形式):
% 1. RamanData.data (99×2000 double)
%    RamanData.axisscale{1,1} (99×1 double)
%    RamanData.axisscale{2,1} (2000×1 double)
% 
% 2. 或保存为结构体并在文件中为第一个变量

% 功率数据文件应包含:
% 1. powers / power / laser_power (99×1 double)
% 2. 或任何结构体的 .data 字段

% ========== 拟合公式 ==========

% 功率依赖性: I = a·P^n
% 其中:
%   I - 峰强度 (任意单位)
%   P - 激光功率 (mW)
%   a - 系数 (与光学常数相关)
%   n - 幂指数 (物理含义)
%       n ≈ 1.0: 线性响应 (单光子过程)
%       n ≈ 1.5: 经典预期值
%       n ≈ 2.0: 二阶过程

% ========== 故障排除 ==========

% 拟合失败?
% - 检查数据点数是否 >= 10
% - 检查功率值是否都为正
% - 检查 ramanPlotRange 是否正确

% 峰检测错误?
% - 调整 ramanPlotRange 确保包含感兴趣的峰
% - 或调整 peakWidth 参数

% 范围内无数据?
% - 脚本会自动切换到全范围
% - 检查指定范围是否超出实际数据范围
% - 第一步的输出会显示实际拉曼位移范围

% ========== 相关函数 ==========

% 核心函数 (位于 ../core/data/):
% load_raman_power_data.m - 加载并合并拉曼数据和功率数据
% import_raman_data.m - 导入拉曼数据文件
% extract_laser_power.m - 提取激光功率数据

% 绘图函数 (位于 ../core/visualization/):
% plot_raman_contour.m - 绘制等高线图
% plot_raman_stack.m - 绘制堆叠谱线
% plot_single_raman.m - 绘制单条谱线

% ========== 示例 ==========

% 例1: 使用默认参数分析MoS₂数据
% 直接运行 power_dependence.m

% 例2: 修改范围分析其他材料
% 将第17行改为: ramanPlotRange = [200 300];
% 然后运行脚本

% 例3: 使用自己的数据
% 将第13-14行改为自己的数据路径
% 并根据需要调整 ramanPlotRange
% 然后运行脚本
