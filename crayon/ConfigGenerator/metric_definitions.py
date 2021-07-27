import yaml

GLUONTS_METRICS = yaml.safe_load(
    """
    Coverage[0.1]: 'Coverage\[0\.1\]\): (\d+\.\d+)'
    Coverage[0.5]: 'Coverage\[0\.5\]\): (\d+\.\d+)'
    Coverage[0.9]: 'Coverage\[0\.9\]\): (\d+\.\d+)'
    MAE_Coverage: 'MAE_Coverage\): (\d+\.\d+)'
    MAPE: 'MAPE\): (\d+\.\d+)'
    MASE: 'MASE\): (\d+\.\d+)'
    MSE: 'MSE\): (\d+\.\d+)'
    MSIS: 'MSIS\): (\d+\.\d+)'
    ND: 'ND\): (\d+\.\d+)'
    NRMSE: 'NRMSE\): (\d+\.\d+)'
    OWA: 'OWA\): (\d+\.\d+)'
    QuantileLoss[0.1]: 'QuantileLoss\[0\.1\]\): (\d+\.\d+)'
    QuantileLoss[0.5]: 'QuantileLoss\[0\.5\]\): (\d+\.\d+)'
    QuantileLoss[0.9]: 'QuantileLoss\[0\.9\]\): (\d+\.\d+)'
    RMSE: 'RMSE\): (\d+\.\d+)'
    abs_error: 'abs_error\): (\d+\.\d+)'
    abs_target_mean: 'abs_target_mean\): (\d+\.\d+)'
    abs_target_sum: 'abs_target_sum\): (\d+\.\d+)'
    mean_wQuantileLoss: 'mean_wQuantileLoss\): (\d+\.\d+)'
    sMAPE: 'sMAPE\): (\d+\.\d+)'
    seasonal_error: 'seasonal_error\): (\d+\.\d+)'
    wQuantileLoss[0.1]: 'wQuantileLoss\[0\.1\]\): (\d+\.\d+)'
    wQuantileLoss[0.5]: 'wQuantileLoss\[0\.5\]\): (\d+\.\d+)'
    wQuantileLoss[0.9]: 'wQuantileLoss\[0\.9\]\): (\d+\.\d+)'
    """
)
