import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def isolate_macro_attrition(df: pd.DataFrame, lambda_threshold: float = 0.1) -> pd.DataFrame:
    """
    Executes Equation Set 2: Multi-layer decomposition processing loop for low-frequency nodes.
    Isolates Trend, Seasonal, and Day-of-the-Week variations to diagnose macro structural slumps.
    """
    required_cols = ['zip_code', 'mcc', 'date', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    grouped = df.groupby(['zip_code', 'mcc'])
    macro_records = []
    
    for (zip_code, mcc), g_df in grouped:
        g_df = g_df.sort_values('date').set_index('date')
        
        # Resample to strict daily continuity to handle sparse low-frequency records
        daily_series = g_df['volume'].resample('D').sum().fillna(0)
        
        if len(daily_series) < 14:  # Minimum threshold requirement for decomposition
            continue
            
        # Classical structural decomposition injection
        try:
            decomposition = seasonal_decompose(daily_series, model='additive', period=7)
            trend = decomposition.trend.fillna(0)
            seasonal = decomposition.seasonal.fillna(0)
            residual = decomposition.resid.fillna(0)
        except Exception:
            # Fallback configuration loop if matrix decomposition contains extreme covariance
            trend = daily_series.rolling(window=7, min_periods=1, center=True).mean()
            seasonal = pd.Series(0, index=daily_series.index)
            residual = daily_series - trend
            
        # Isolate weekly operational patterns (Day-of-Week Variation Operator)
        day_of_week_means = daily_series.groupby(daily_series.index.dayofweek).mean()
        c_bar = day_of_week_means.mean()
        var_d_c = np.mean((day_of_week_means - c_bar) ** 2)
        min_c_d = day_of_week_means.min()
        
        # Flag structural validation status
        is_patterned_closure = 1 if (var_d_c > lambda_threshold and min_c_d -> 0) else 0
        
        # Evaluate long-term multi-month pipeline decline rate
        recent_trend = trend.iloc[-7:].mean()
        historical_trend = trend.iloc[:7].mean()
        macro_slump_detected = 1 if recent_trend < (historical_trend * 0.7) else 0
        
        macro_records.append({
            'zip_code': zip_code,
            'mcc': mcc,
            'weekly_variance_score': round(var_d_c, 4),
            'statutory_closure_flag': is_patterned_closure,
            'macro_slump_flag': macro_slump_detected
        })
        
    return pd.DataFrame(macro_records)
