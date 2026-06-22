import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def suppress_outliers(df: pd.DataFrame, value_col: str, window_days: int = 30) -> pd.DataFrame:
    """
    Suppresses anomalous transaction spikes and POS dropouts greater than 
    3 standard deviations from the rolling monthly median.
    """
    df = df.copy()
    rolling_median = df[value_col].rolling(window=window_days, min_periods=1, center=True).median()
    rolling_std = df[value_col].rolling(window=window_days, min_periods=1, center=True).std()
    
    upper_bound = rolling_median + (3 * rolling_std)
    lower_bound = rolling_median - (3 * rolling_std)
    
    # Clip outliers to bounds
    df[value_col] = np.clip(df[value_col], lower_bound, upper_bound)
    return df

def apply_gaussian_smoothing(series: pd.Series, sigma: float = 1.0) -> np.ndarray:
    """
    Applies a rolling 1D Gaussian filter to eliminate stochastic white noise 
    without causing phase-shifts in temporal peak/valley alignment.
    """
    return gaussian_filter1d(series.astype(float).values, sigma=sigma)

def calculate_optimized_baseline(df: pd.DataFrame, alpha: float = 0.25) -> pd.DataFrame:
    """
    Executes Equation Set 1: Evaluates 15-minute multi-week interval baselines.
    Returns structurally pure time-series records flagging baseline capacity drop events.
    """
    # Required schema validation
    required_cols = ['zip_code', 'mcc', 'timestamp', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['time_window'] = df['timestamp'].dt.floor('15T').dt.time
    
    # Suppress extreme transaction variations
    df = suppress_outliers(df, 'volume')
    
    # Smooth incoming raw volumes
    df['volume_smoothed'] = apply_gaussian_smoothing(df['volume'])
    
    # Multi-week array aggregation grouping (Historical Baseline Calculations)
    grouped = df.groupby(['zip_code', 'mcc', 'day_of_week', 'time_window'])
    
    results = []
    for g_keys, g_df in grouped:
        zip_code, mcc, day_of_week, time_window = g_keys
        
        # Pull baseline stats over the target group
        mu_30 = g_df['volume_smoothed'].mean()
        sigma_30 = g_df['volume_smoothed'].std()
        if pd.isna(sigma_30):
            sigma_30 = 0.0
            
        current_volume = g_df['volume_smoothed'].iloc[-1]
        
        # Core Boolean Trigger Check
        threshold = max(alpha * mu_30, mu_30 - (2 * sigma_30))
        boolean_trigger = 1 if current_volume < threshold else 0
        
        if boolean_trigger == 1:
            drop_percentage = ((mu_30 - current_volume) / mu_30 * 100) if mu_30 > 0 else 0
            results.append({
                'zip_code': zip_code,
                'mcc': mcc,
                'day_of_week': day_of_week,
                'dead_air_window': str(time_window),
                'average_drop_percentage': round(drop_percentage, 2)
            })
            
    return pd.DataFrame(results)
