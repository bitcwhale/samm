import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import datetime
import os
from sklearn.covariance import LedoitWolf
import cvxpy as cp
from sklearn.decomposition import PCA

# Define base URL for GitHub raw data files
base_url = "https://raw.githubusercontent.com/bitcwhale/samm/main/"

# [STEP 1 to STEP 5 remain unchanged - data loading, cleaning, and MVP weight function]

# ------------------------------
# STEP 6 – ROLLING OPTIMIZATION
# ------------------------------

# Adjust start_year to 2012 to include 2013 returns
start_year = 2012  # Weights for 2013 returns
end_year = 2023    # Weights for 2024 returns
methods = ['lw', 'pinv', 'factor_1', 'factor_3', 'identity']
mvp_weights_all = {}

for method in methods:
    mvp_weights = {}
    prev_weights = None
    for year in range(start_year, end_year + 1):
        window_start = pd.Timestamp(f"{year - 10}-01-01")
        window_end = pd.Timestamp(f"{year - 1}-12-31")
        
        eligible_assets = first_available[first_available <= window_start].index
        returns_window = simple_returns[(simple_returns.index >= window_start) &
                                        (simple_returns.index <= window_end)][eligible_assets]
        mkt_caps_window = mkt_caps[(mkt_caps.index >= window_start) &
                                   (mkt_caps.index <= window_end)][eligible_assets]
        revenues_window = revenues[(revenues.index >= window_start) &
                                   (revenues.index <= window_end)][eligible_assets]
        rf_window = rf_series[(rf_series.index >= window_start) &
                              (rf_series.index <= window_end)]

        weights = compute_mvp_weights(returns_window, mkt_caps_window, revenues_window, rf_window,
                                      method=method, max_weight=0.05,
                                      prev_weights=prev_weights, turnover_limit=0.3)
        mvp_weights[year] = weights.dropna()
        prev_weights = weights

    mvp_weights_all[method] = mvp_weights

# ------------------------------
# STEP 7 – EX-POST RETURNS
# ------------------------------

def compute_portfolio_returns(returns_df, weights_dict=None, mkt_caps_df=None, mode='mvp', start_date=None, end_date=None):
    """Compute portfolio returns for MVP or VW portfolios within a specified date range."""
    if start_date is None:
        start_date = returns_df.index.min()
    if end_date is None:
        end_date = returns_df.index.max()
    
    R = returns_df[(returns_df.index >= start_date) & (returns_df.index <= end_date)]
    returns = []
    dates = []

    if mode == 'mvp':
        if weights_dict is None:
            raise ValueError("weights_dict must be provided for mode='mvp'")
        years = sorted(weights_dict.keys())
        for year in years:
            alpha = weights_dict.get(year)
            if alpha is None or alpha.empty:
                continue
            alpha = alpha / alpha.sum()
            next_year_start = pd.Timestamp(f"{year + 1}-01-01")
            next_year_end = pd.Timestamp(f"{year + 1}-12-31")
            R_year = R[(R.index >= next_year_start) & (R.index <= next_year_end)]
            for date in R_year.index:
                r_t = R_year.loc[date]
                common_assets = alpha.index.intersection(r_t.index)
                if len(common_assets) < 2:
                    continue
                r_t = r_t[common_assets].fillna(0)
                w = alpha[common_assets] / alpha[common_assets].sum()
                ret_p = np.dot(w, r_t)
                returns.append(ret_p)
                alpha = w * (1 + r_t)
                if ret_p != -1:
                    alpha = alpha / (1 + ret_p)
                dates.append(date)
    
    elif mode == 'vw':
        if mkt_caps_df is None:
            raise ValueError("mkt_caps_df must be provided for mode='vw'")
        for date in R.index:
            r_t = R.loc[date]
            mc_t_shifted = mkt_caps_df.shift(1).loc[date]
            if mc_t_shifted.isna().all():
                continue
            common_assets = r_t.index.intersection(mc_t_shifted.index)
            if len(common_assets) < 2:
                continue
            r_t = r_t[common_assets].fillna(0)
            mc_weights = mc_t_shifted[common_assets] / mc_t_shifted[common_assets].sum()
            ret_p = np.dot(mc_weights.fillna(0), r_t)
            returns.append(ret_p)
            dates.append(date)
    
    else:
        raise ValueError("Mode must be 'mvp' or 'vw'")
    
    return pd.Series(returns, index=pd.to_datetime(dates))

# Compute VW portfolio returns from 2013 to 2024
vw_start_date = pd.Timestamp("2013-01-01")
vw_end_date = pd.Timestamp("2024-12-31")
vw_series = compute_portfolio_returns(simple_returns, mkt_caps_df=mkt_caps, mode='vw', 
                                      start_date=vw_start_date, end_date=vw_end_date)

# Compute MVP returns from 2013 to 2024 (weights from 2012 to 2023)
mvp_series_all = {method: compute_portfolio_returns(simple_returns, mvp_weights_all[method], mode='mvp',
                                                    start_date=vw_start_date, end_date=vw_end_date)
                  for method in methods}

# ------------------------------
# STEP 9 – PLOT CUMULATIVE RETURNS
# ------------------------------

plt.figure(figsize=(12, 6))
for method, series in mvp_series_all.items():
    if not series.dropna().empty:
        (1 + series).cumprod().plot(label=f"MVP ({method})")
(1 + vw_series).cumprod().plot(label="VW Portfolio")

# Dynamic title based on data range
min_year = min([series.index.min().year for series in mvp_series_all.values() if not series.empty] + [vw_series.index.min().year])
max_year = max([series.index.max().year for series in mvp_series_all.values() if not series.empty] + [vw_series.index.max().year])
plt.title(f"Cumulative Returns ({min_year}–{max_year})")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# STEP 11 – CARBON METRICS FOR P_mv_oos
# ------------------------------

# Adjust range to 2013–2023 (carbon data availability)
for Y in range(2013, 2024):  # Changed from (2013, 2025) since carbon data ends in 2023
    weights_Y_minus_1 = mvp_weights_all[method].get(Y - 1, pd.Series())
    if weights_Y_minus_1.empty:
        print(f"No weights available for year {Y-1}")
        continue
    firms_Y = weights_Y_minus_1.index
    emissions_Y = emissions_long[emissions_long['Year'] == Y].set_index('Firm_Name')['E'].reindex(firms_Y).dropna()
    mkt_cap_Y = mkt_caps_annual.loc[Y].reindex(firms_Y).dropna() if Y in mkt_caps_annual.index else pd.Series()
    revenue_Y = revenues_annual.loc[Y].reindex(firms_Y).dropna() if Y in revenues_annual.index else pd.Series()
    
    if mkt_cap_Y.empty or revenue_Y.empty:
        print(f"No market cap or revenue data for year {Y}")
        continue
    
    common_firms = weights_Y_minus_1.index.intersection(emissions_Y.index).intersection(mkt_cap_Y.index).intersection(revenue_Y.index)
    if common_firms.empty:
        print(f"No common firms for year {Y}")
        continue

    weights_Y_minus_1 = weights_Y_minus_1.reindex(common_firms)
    emissions_Y = emissions_Y.reindex(common_firms)
    mkt_cap_Y = mkt_cap_Y.reindex(common_firms)
    revenue_Y = revenue_Y.reindex(common_firms)

    emissions_over_mkt_cap = emissions_Y / mkt_cap_Y
    cf_Y = (weights_Y_minus_1 * emissions_over_mkt_cap).sum()
    carbon_footprints[Y] = cf_Y

    revenue_Y_millions = revenue_Y / 1_000
    ci_Y = emissions_Y / revenue_Y_millions
    waci_Y = (weights_Y_minus_1 * ci_Y).sum()
    waci_values[Y] = waci_Y

# Plot CF and WACI with dynamic title
plt.figure(figsize=(10, 6))
years_cf = list(carbon_footprints.keys())
plt.plot(years_cf, list(carbon_footprints.values()), marker='o', label='Carbon Footprint (tons CO2e per million USD invested)')
plt.plot(years_cf, list(waci_values.values()), marker='s', label='WACI (tons CO2e per million USD of revenue)')
min_year_cf = min(years_cf)
max_year_cf = max(years_cf)
plt.title(f"Carbon Footprint and WACI for P_mv_oos ({min_year_cf}–{max_year_cf})")
plt.xlabel("Year")
plt.ylabel("Metrics (tons CO2e per million USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# [STEP 12 to STEP 15 - Adjust optimization ranges and plot titles similarly]

# ------------------------------
# STEP 12 – PORTFOLIO WITH 50% CARBON FOOTPRINT REDUCTION (2.2)
# ------------------------------

# Adjust start_year to 2012 to include weights for 2013 returns
start_year = 2012
end_year = 2023

# Compute c_Y vectors for each year (E_{i,Y} / Cap_{i,Y})
c_vectors = {}
for Y in range(2013, 2024):
    emissions_Y = emissions_long[emissions_long['Year'] == Y].set_index('Firm_Name')['E']
    if Y not in mkt_caps_annual.index:
        continue
    mkt_cap_Y = mkt_caps_annual.loc[Y]
    common_firms = emissions_Y.index.intersection(mkt_cap_Y.index)
    c_Y = emissions_Y.reindex(common_firms) / mkt_cap_Y.reindex(common_firms)
    c_vectors[Y] = c_Y.dropna()

# Assume carbon_footprints from original portfolio (STEP 10) is available
carbon_footprints_original = carbon_footprints  # From STEP 10

# Rolling optimization for constrained portfolio
mvp_weights_constrained = {}
prev_weights = None
method = 'lw'  # Ledoit-Wolf covariance estimation
for year in range(start_year, end_year + 1):
    window_start = pd.Timestamp(f"{year - 10}-01-01")
    window_end = pd.Timestamp(f"{year - 1}-12-31")
    eligible_assets = first_available[first_available <= window_start].index
    returns_window = simple_returns[(simple_returns.index >= window_start) &
                                    (simple_returns.index <= window_end)][eligible_assets]
    mkt_caps_window = mkt_caps[(mkt_caps.index >= window_start) &
                               (mkt_caps.index <= window_end)][eligible_assets]
    revenues_window = revenues[(revenues.index >= window_start) &
                               (revenues.index <= window_end)][eligible_assets]
    rf_window = rf_series[(rf_series.index >= window_start) &
                          (rf_series.index <= window_end)]

    Y = year + 1
    if Y in c_vectors and Y in carbon_footprints_original:
        c_Y = c_vectors[Y]
        threshold = 0.5 * carbon_footprints_original[Y]
        carbon_constraint = (c_Y, threshold)
    else:
        carbon_constraint = None
        print(f"Warning: No carbon constraint data for year {Y}")

    weights = compute_mvp_weights(returns_window, mkt_caps_window, revenues_window, rf_window,
                                  method=method, max_weight=0.05,
                                  prev_weights=prev_weights, turnover_limit=0.3,
                                  carbon_constraint=carbon_constraint)
    mvp_weights_constrained[year] = weights.dropna()
    prev_weights = weights

# Compute ex-post returns for constrained portfolio
mvp_series_constrained = compute_portfolio_returns(simple_returns, mvp_weights_constrained, mode='mvp')

# Compute performance metrics
metrics['constrained'] = compute_metrics(mvp_series_constrained, rf_aligned)

# Compute carbon footprints for constrained portfolio (verification)
carbon_footprints_constrained = {}
for Y in range(2013, 2024):
    weights_Y_minus_1 = mvp_weights_constrained.get(Y - 1, pd.Series())
    if weights_Y_minus_1.empty:
        continue
    firms_Y = weights_Y_minus_1.index
    c_Y = c_vectors.get(Y, pd.Series())
    common_firms = firms_Y.intersection(c_Y.index)
    if common_firms.empty:
        continue
    weights_Y_minus_1 = weights_Y_minus_1.reindex(common_firms)
    c_Y = c_Y.reindex(common_firms)
    cf_Y = (weights_Y_minus_1 * c_Y).sum()
    carbon_footprints_constrained[Y] = cf_Y

# Display carbon footprints comparison
print("\n=== Carbon Footprint Comparison ===")
print("Year | Original CF | Constrained CF | Ratio (Constrained/Original)")
for Y in carbon_footprints_constrained.keys():
    cf_orig = carbon_footprints_original.get(Y, np.nan)
    cf_constr = carbon_footprints_constrained[Y]
    ratio = cf_constr / cf_orig if cf_orig != 0 else np.nan
    print(f"{Y}   | {cf_orig:.2f}       | {cf_constr:.2f}         | {ratio:.2f}")

# Update and display metrics table
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=metric_names)
print("\n=== Updated Portfolio Metrics Table ===")
print(metrics_df.to_string(formatters={k: "{:.4f}".format for k in metric_names}))

# Plot cumulative returns
plt.figure(figsize=(12, 6))
(1 + mvp_series_all['lw']).cumprod().plot(label="Original MVP (lw)")
(1 + mvp_series_constrained).cumprod().plot(label="Constrained MVP (lw, 0.5 CF)")
(1 + vw_series).cumprod().plot(label="VW Portfolio")
plt.title("Cumulative Returns Comparison (2013–2024)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot carbon footprints
plt.figure(figsize=(10, 6))
plt.plot(list(carbon_footprints_original.keys()), list(carbon_footprints_original.values()), marker='o', label='Original MVP CF')
plt.plot(list(carbon_footprints_constrained.keys()), list(carbon_footprints_constrained.values()), marker='s', label='Constrained MVP CF (0.5)')
plt.title("Carbon Footprint Comparison (2013–2023)")
plt.xlabel("Year")
plt.ylabel("Carbon Footprint (tons CO2e per million USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# STEP 13 – PORTFOLIO TRACKING WITH 50% CARBON FOOTPRINT REDUCTION (2.3)
# ------------------------------

# Function to compute value-weighted benchmark weights
def compute_vw_weights(mkt_caps_annual, year):
    if year not in mkt_caps_annual.index:
        print(f"No market cap data for year {year}")
        return pd.Series()
    mkt_cap_Y = mkt_caps_annual.loc[year]
    total_cap = mkt_cap_Y.sum()
    if total_cap == 0:
        print(f"Total market cap is zero for year {year}")
        return pd.Series()
    return mkt_cap_Y / total_cap

# Compute benchmark carbon footprint
benchmark_cf = {}
for Y in range(2013, 2024):
    emissions_Y = emissions_long[emissions_long['Year'] == Y].set_index('Firm_Name')['E']
    if Y not in mkt_caps_annual.index:
        continue
    mkt_cap_Y = mkt_caps_annual.loc[Y]
    common_firms = emissions_Y.index.intersection(mkt_cap_Y.index)
    emissions_Y = emissions_Y.reindex(common_firms)
    total_cap_Y = mkt_cap_Y.reindex(common_firms).sum()
    cf_vw_Y = emissions_Y.sum() / total_cap_Y if total_cap_Y != 0 else np.nan
    benchmark_cf[Y] = cf_vw_Y

# Rolling optimization for tracking portfolio
start_year = 2012
end_year = 2023
tracking_weights = {}
method = 'lw'

for year in range(start_year, end_year + 1):
    window_start = pd.Timestamp(f"{year - 10}-01-01")
    window_end = pd.Timestamp(f"{year - 1}-12-31")
    eligible_assets = first_available[first_available <= window_start].index
    returns_window = simple_returns[(simple_returns.index >= window_start) &
                                   (simple_returns.index <= window_end)][eligible_assets]
    mkt_caps_window = mkt_caps[(mkt_caps.index >= window_start) &
                              (mkt_caps.index <= window_end)][eligible_assets]
    revenues_window = revenues[(revenues.index >= window_start) &
                              (revenues.index <= window_end)][eligible_assets]
    rf_window = rf_series[(rf_series.index >= window_start) &
                         (rf_series.index <= window_end)]
    
    vw_weights_Y = compute_vw_weights(mkt_caps_annual, year)
    if vw_weights_Y.empty:
        continue
    
    c_Y = c_vectors.get(year, pd.Series())
    if c_Y.empty:
        continue
    
    cf_vw_Y = benchmark_cf.get(year, np.nan)
    if np.isnan(cf_vw_Y):
        continue
    cf_threshold = 0.5 * cf_vw_Y
    
    weights = compute_tracking_weights(returns_window, mkt_caps_window, revenues_window, rf_window,
                                      vw_weights_Y, c_Y, cf_threshold, method=method, max_weight=0.05)
    tracking_weights[year] = weights.dropna()

# Compute ex-post portfolio returns
tracking_series = compute_portfolio_returns(simple_returns, tracking_weights, mode='mvp')

# Compute performance metrics
metrics['tracking_0.5'] = compute_metrics(tracking_series, rf_aligned)

# Compute carbon footprints for verification
carbon_footprints_tracking = {}
for Y in range(2013, 2024):
    weights_Y_minus_1 = tracking_weights.get(Y - 1, pd.Series())
    if weights_Y_minus_1.empty:
        continue
    firms_Y = weights_Y_minus_1.index
    c_Y = c_vectors.get(Y, pd.Series())
    common_firms = firms_Y.intersection(c_Y.index)
    if common_firms.empty:
        continue
    weights_Y_minus_1 = weights_Y_minus_1.reindex(common_firms)
    c_Y = c_Y.reindex(common_firms)
    cf_Y = (weights_Y_minus_1 * c_Y).sum()
    carbon_footprints_tracking[Y] = cf_Y

# Display carbon footprint comparison
print("\n=== Carbon Footprint Comparison for Tracking Portfolio ===")
print("Year | Benchmark CF | Tracking CF | Ratio (Tracking/Benchmark)")
for Y in carbon_footprints_tracking.keys():
    cf_vw = benchmark_cf.get(Y, np.nan)
    cf_track = carbon_footprints_tracking[Y]
    ratio = cf_track / cf_vw if cf_vw != 0 else np.nan
    print(f"{Y}   | {cf_vw:.2f}       | {cf_track:.2f}         | {ratio:.2f}")

# Update and display metrics table
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=metric_names)
print("\n=== Updated Portfolio Metrics Table ===")
print(metrics_df.to_string(formatters={k: "{:.4f}".format for k in metric_names}))

# Plot cumulative returns
plt.figure(figsize=(12, 6))
(1 + vw_series).cumprod().plot(label="VW Portfolio")
(1 + tracking_series).cumprod().plot(label="Tracking Portfolio (0.5 CF)")
plt.title("Cumulative Returns Comparison (2013–2024)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot carbon footprints
plt.figure(figsize=(10, 6))
plt.plot(list(benchmark_cf.keys()), list(benchmark_cf.values()), marker='o', label='Benchmark CF')
plt.plot(list(carbon_footprints_tracking.keys()), list(carbon_footprints_tracking.values()), marker='s', label='Tracking Portfolio CF (0.5)')
plt.title("Carbon Footprint Comparison (2013–2023)")
plt.xlabel("Year")
plt.ylabel("Carbon Footprint (tons CO2e per million USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# STEP 14 – COMPARISON ANALYSIS 50% CARBON FOOTPRINT REDUCTION (2.4)
# ------------------------------

# Align risk-free rate with portfolio returns
rf_aligned_mv = rf_series.reindex(mvp_series_all['lw'].index, method="ffill")
rf_aligned_mv_constr = rf_series.reindex(mvp_series_constrained.index, method="ffill")
rf_aligned_vw = rf_series.reindex(vw_series.index, method="ffill")
rf_aligned_track = rf_series.reindex(tracking_series.index, method="ffill")

# Compute metrics for all portfolios
metric_names = ["Annualized Return", "Annualized Volatility", "Cumulative Return",
                "Sharpe Ratio", "Min Monthly Return", "Max Monthly Return", "Max Drawdown"]
metrics_mv = compute_metrics(mvp_series_all['lw'], rf_aligned_mv)
metrics_mv_constr = compute_metrics(mvp_series_constrained, rf_aligned_mv_constr)
metrics_vw = compute_metrics(vw_series, rf_aligned_vw)
metrics_track = compute_metrics(tracking_series, rf_aligned_track)

# Create metrics DataFrame
metrics_dict = {
    "P_oos^(mv)": metrics_mv,
    "P_oos^(mv)(0.5)": metrics_mv_constr,
    "P^(vw)": metrics_vw,
    "P_oos^(vw)(0.5)": metrics_track
}
metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=metric_names)

# Display financial performance comparison
print("\n=== Financial Performance Comparison (2013–2024) ===")
print(metrics_df.to_string(formatters={k: "{:.4f}".format for k in metric_names}))

# Carbon footprint comparison DataFrames
cf_mv_df = pd.DataFrame({
    "P_oos^(mv)": carbon_footprints_original,
    "P_oos^(mv)(0.5)": carbon_footprints_constrained
}).dropna()
cf_mv_df["Ratio"] = cf_mv_df["P_oos^(mv)(0.5)"] / cf_mv_df["P_oos^(mv)"]

cf_vw_df = pd.DataFrame({
    "P^(vw)": benchmark_cf,
    "P_oos^(vw)(0.5)": carbon_footprints_tracking
}).dropna()
cf_vw_df["Ratio"] = cf_vw_df["P_oos^(vw)(0.5)"] / cf_vw_df["P^(vw)"]

# Display carbon footprint comparisons
print("\n=== Carbon Footprint Comparison: P_oos^(mv) vs P_oos^(mv)(0.5) ===")
print(cf_mv_df.to_string(formatters={"P_oos^(mv)": "{:.2f}".format,
                                     "P_oos^(mv)(0.5)": "{:.2f}".format,
                                     "Ratio": "{:.2f}".format}))

print("\n=== Carbon Footprint Comparison: P^(vw) vs P_oos^(vw)(0.5) ===")
print(cf_vw_df.to_string(formatters={"P^(vw)": "{:.2f}".format,
                                     "P_oos^(vw)(0.5)": "{:.2f}".format,
                                     "Ratio": "{:.2f}".format}))

# Plot cumulative returns
plt.figure(figsize=(12, 6))
(1 + mvp_series_all['lw']).cumprod().plot(label="P_oos^(mv)")
(1 + mvp_series_constrained).cumprod().plot(label="P_oos^(mv)(0.5)")
(1 + vw_series).cumprod().plot(label="P^(vw)")
(1 + tracking_series).cumprod().plot(label="P_oos^(vw)(0.5)")
plt.title("Cumulative Returns Comparison (2013–2024)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot carbon footprints
plt.figure(figsize=(12, 6))
plt.plot(cf_mv_df.index, cf_mv_df["P_oos^(mv)"], marker='o', label="P_oos^(mv) CF")
plt.plot(cf_mv_df.index, cf_mv_df["P_oos^(mv)(0.5)"], marker='s', label="P_oos^(mv)(0.5) CF")
plt.plot(cf_vw_df.index, cf_vw_df["P^(vw)"], marker='^', label="P^(vw) CF")
plt.plot(cf_vw_df.index, cf_vw_df["P_oos^(vw)(0.5)"], marker='d', label="P_oos^(vw)(0.5) CF")
plt.title("Carbon Footprint Comparison (2013–2023)")
plt.xlabel("Year")
plt.ylabel("Carbon Footprint (tons CO2e per million USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# STEP 15 – NET ZERO PORTFOLIO (3.1)
# ------------------------------

# Base year and reduction rate
Y0 = 2013
theta = 0.10  # 10% annual reduction

# Compute base year carbon footprint for benchmark
emissions_Y0 = emissions_long[emissions_long['Year'] == Y0].set_index('Firm_Name')['E']
mkt_cap_Y0 = mkt_caps_annual.loc[Y0]
common_firms_Y0 = emissions_Y0.index.intersection(mkt_cap_Y0.index)
emissions_Y0 = emissions_Y0.reindex(common_firms_Y0)
total_cap_Y0 = mkt_cap_Y0.reindex(common_firms_Y0).sum()
CF_Y0_vw = emissions_Y0.sum() / total_cap_Y0
print(f"Base Year (2013) Benchmark Carbon Footprint: {CF_Y0_vw:.2f} tons CO2e per million USD")

# Function to compute annual Net Zero targets
def compute_nz_target(Y, CF_Y0, theta):
    return (1 - theta) ** (Y - Y0 + 1) * CF_Y0

# Rolling optimization for Net Zero portfolio
nz_weights = {}
method = 'lw'
start_year = 2012
end_year = 2023

for year in range(start_year, end_year + 1):
    window_start = pd.Timestamp(f"{year - 10}-01-01")
    window_end = pd.Timestamp(f"{year - 1}-12-31")
    eligible_assets = first_available[first_available <= window_start].index
    returns_window = simple_returns[(simple_returns.index >= window_start) &
                                   (simple_returns.index <= window_end)][eligible_assets]
    if returns_window.empty or returns_window.shape[1] < 2:
        nz_weights[year] = pd.Series(np.nan, index=eligible_assets)
        continue
    
    vw_weights_Y = compute_vw_weights(mkt_caps_annual, year)
    if vw_weights_Y.empty:
        nz_weights[year] = pd.Series(np.nan, index=eligible_assets)
    c_Y = c_vectors.get(year, pd.Series())
    if c_Y.empty:
        nz_weights[year] = pd.Series(np.nan, index=eligible_assets)
        continue
    
    cf_target_Y = compute_nz_target(year, CF_Y0_vw, theta)
    
    assets = returns_window.columns
    vw_weights_aligned = vw_weights_Y.reindex(assets).fillna(0).values
    c_aligned = c_Y.reindex(assets).fillna(0).values
    
    lw = LedoitWolf()
    lw.fit(returns_window)
    cov_matrix = lw.covariance_
    
    w = cp.Variable(len(assets))
    diff = w - vw_weights_aligned
    objective = cp.Minimize(cp.quad_form(diff, cp.psd_wrap(cov_matrix)))
    constraints = [cp.sum(w) == 1, w >= 0, w @ c_aligned <= cf_target_Y, w <= 0.05]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    nz_weights[year] = pd.Series(w.value, index=assets) if w.value is not None else pd.Series(np.nan, index=assets)

# Compute ex-post returns
nz_series = compute_portfolio_returns(simple_returns, nz_weights, mode='mvp')

# Compute performance metrics
metrics['nz'] = compute_metrics(nz_series, rf_series.reindex(nz_series.index, method='ffill'))

# Compute carbon footprints for verification
carbon_footprints_nz = {}
for Y in range(2013, 2024):
    weights_Y_minus_1 = nz_weights.get(Y - 1, pd.Series())
    if weights_Y_minus_1.empty or weights_Y_minus_1.isna().all():
        continue
    firms_Y = weights_Y_minus_1.index
    c_Y = c_vectors.get(Y, pd.Series())
    common_firms = firms_Y.intersection(c_Y.index)
    if common_firms.empty:
        continue
    weights_Y_minus_1 = weights_Y_minus_1.reindex(common_firms)
    c_Y = c_Y.reindex(common_firms)
    cf_Y = (weights_Y_minus_1 * c_Y).sum()
    carbon_footprints_nz[Y] = cf_Y

# Display metrics and carbon footprints
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=metric_names)
print("\n=== Net Zero Portfolio Metrics ===")
print(metrics_df.loc['nz'].to_string(formatters={k: "{:.4f}".format for k in metric_names}))

print("\n=== Carbon Footprint for NZ Portfolio ===")
print("Year | Target CF | Actual CF | Ratio (Actual/Target)")
for Y in carbon_footprints_nz.keys():
    cf_target = compute_nz_target(Y, CF_Y0_vw, theta)
    cf_actual = carbon_footprints_nz[Y]
    ratio = cf_actual / cf_target if cf_target != 0 else np.nan
    print(f"{Y}   | {cf_target:.2f}       | {cf_actual:.2f}         | {ratio:.2f}")

# Plot cumulative returns
plt.figure(figsize=(12, 6))
(1 + vw_series).cumprod().plot(label="VW Portfolio")
(1 + nz_series).cumprod().plot(label="NZ Portfolio")
plt.title("Cumulative Returns: VW vs NZ Portfolio (2013–2024)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot carbon footprints with targets
years = list(carbon_footprints_nz.keys())
targets = [compute_nz_target(Y, CF_Y0_vw, theta) for Y in years]
actual_cf = list(carbon_footprints_nz.values())
plt.figure(figsize=(10, 6))
plt.plot(years, targets, marker='o', label='NZ Target CF')
plt.plot(years, actual_cf, marker='s', label='NZ Portfolio CF')
plt.title("NZ Portfolio Carbon Footprint vs Targets (2013–2023)")
plt.xlabel("Year")
plt.ylabel("Carbon Footprint (tons CO2e per million USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
