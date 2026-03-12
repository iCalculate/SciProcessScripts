function [fitresult, gof] = shg_power_fitting(power, intensity, varargin)
%SHG_POWER_FITTING SHG强度功率依赖性拟合
%   [fitresult, gof] = shg_power_fitting(power, intensity, Name, Value)
%
%   对SHG强度随功率变化的规律进行幂律拟合: I ∝ P^n
%
%   输入:
%       power - 功率数据向量 (mW)
%       intensity - SHG强度数据向量 (a.u.)
%
%   可选参数:
%       'ModelType' - 模型类型: 'power' (默认) 或 'exponential'
%       'InitialGuess' - 初始参数猜测
%       'Plot' - 是否绘制拟合结果，默认 false
%
%   输出:
%       fitresult - 拟合结果结构体
%           .coeff - 拟合系数
%           .exponent - 幂指数(仅对power模型)
%           .rsquare - 拟合优度 R²
%       gof - 拟合统计信息
%
%   示例:
%       power = 0.1:0.1:2.0;
%       intensity = power.^2;
%       [fitresult, gof] = shg_power_fitting(power, intensity, 'Plot', true);

p = inputParser;
addRequired(p, 'power', @isvector);
addRequired(p, 'intensity', @isvector);
addParameter(p, 'ModelType', 'power', @(x) ismember(x, {'power', 'exponential'}));
addParameter(p, 'InitialGuess', [], @(x) isempty(x) || isvector(x));
addParameter(p, 'Plot', false, @islogical);
parse(p, power, intensity, varargin{:});

power = power(:);
intensity = intensity(:);

% 移除负值和零值
valid_idx = power > 0 & intensity > 0;
power_fit = power(valid_idx);
intensity_fit = intensity(valid_idx);

fitresult = struct();
gof = struct();

switch p.Results.ModelType
    case 'power'
        % 幂律模型: I = a * P^n
        % 对数化: log(I) = log(a) + n*log(P)
        
        % 线性回归
        log_power = log(power_fit);
        log_intensity = log(intensity_fit);
        
        coeffs = polyfit(log_power, log_intensity, 1);
        n_exponent = coeffs(1);
        log_a = coeffs(2);
        a_coeff = exp(log_a);
        
        % 计算拟合值
        intensity_fitted = a_coeff * power_fit.^n_exponent;
        
        % 计算R²
        ss_res = sum((intensity_fit - intensity_fitted).^2);
        ss_tot = sum((intensity_fit - mean(intensity_fit)).^2);
        r_squared = 1 - ss_res/ss_tot;
        
        fitresult.coeff = a_coeff;
        fitresult.exponent = n_exponent;
        fitresult.model = sprintf('I = %.4f × P^%.4f', a_coeff, n_exponent);
        
        gof.rsquare = r_squared;
        gof.adjrsquare = r_squared;  % 简化处理
        gof.rmse = sqrt(ss_res / length(power_fit));
        
        if p.Results.Plot
            figure('Color', 'w');
            ax1 = subplot(1, 2, 1);
            loglog(ax1, power, intensity, 'bo', 'MarkerSize', 8, 'LineWidth', 2);
            hold(ax1, 'on');
            loglog(ax1, power_fit, intensity_fitted, 'r-', 'LineWidth', 2.5);
            hold(ax1, 'off');
            
            xlabel(ax1, 'Power (mW)', 'FontSize', 12);
            ylabel(ax1, 'SHG Intensity (a.u.)', 'FontSize', 12);
            title(ax1, 'Log-Log Plot', 'FontSize', 13);
            legend(ax1, 'Data', fitresult.model, 'FontSize', 11);
            grid(ax1, 'on');
            
            ax2 = subplot(1, 2, 2);
            residuals = intensity_fit - intensity_fitted;
            plot(ax2, power_fit, residuals, 'go', 'MarkerSize', 8, 'LineWidth', 2);
            hold(ax2, 'on');
            axline(ax2, [0 0], [1 0], 'LineStyle', '--', 'Color', 'r', 'LineWidth', 1.5);
            hold(ax2, 'off');
            
            xlabel(ax2, 'Power (mW)', 'FontSize', 12);
            ylabel(ax2, 'Residuals', 'FontSize', 12);
            title(ax2, sprintf('Residual Plot (R² = %.4f)', r_squared), 'FontSize', 13);
            grid(ax2, 'on');
        end
        
    case 'exponential'
        % 指数模型: I = a * exp(b*P)
        
        try
            % 初始参数猜测
            if isempty(p.Results.InitialGuess)
                initial_a = intensity_fit(1);
                initial_b = 0.1;
            else
                initial_a = p.Results.InitialGuess(1);
                initial_b = p.Results.InitialGuess(2);
            end
            
            % 非线性拟合
            ft = fittype('a*exp(b*x)', 'independent', 'x', 'dependent', 'y');
            opts = fitoptions(ft);
            opts.StartPoint = [initial_a, initial_b];
            
            [fit_obj, fit_gof] = fit(power_fit, intensity_fit, ft, opts);
            
            intensity_fitted = fit_obj(power_fit);
            
            ss_res = sum((intensity_fit - intensity_fitted).^2);
            ss_tot = sum((intensity_fit - mean(intensity_fit)).^2);
            r_squared = 1 - ss_res/ss_tot;
            
            fitresult.coeff_a = fit_obj.a;
            fitresult.coeff_b = fit_obj.b;
            fitresult.model = sprintf('I = %.4f × exp(%.4f × P)', fit_obj.a, fit_obj.b);
            
            gof.rsquare = r_squared;
            gof.rmse = sqrt(ss_res / length(power_fit));
            
        catch
            warning('Exponential fitting failed, returning NaN');
            fitresult.model = 'Fitting failed';
            gof.rsquare = NaN;
            gof.rmse = NaN;
        end
end

fprintf('SHG功率拟合结果:\n');
fprintf('  模型: %s\n', fitresult.model);
fprintf('  R² = %.6f\n', gof.rsquare);
fprintf('  RMSE = %.6f\n\n', gof.rmse);

end
