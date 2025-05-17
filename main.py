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

# ------------------------------
# STEP 1 – LOAD STATIC & ESG DATA
# ------------------------------

# Load static info and filter for AMER region
static_df = pd.read_excel(base_url + "Static.xlsx")
static_df.columns = static_df.columns.str.strip()  # Remove leading/trailing spaces

# Create a mapping from ISIN to firm name
isin_to_name = static_df.set_index('ISIN')['Name'].to_dict()

# Filter for AMER region using firm names instead of ISINs
amer_df = static_df[static_df["Region"] == "AMER"]
amer_names = set(amer_df["Name"].unique())  # Get unique firm names in AMER region

# Load Scope 1 and Scope 2 emissions
scope1_df = pd.read_excel(base_url + "Scope_1.xlsx")
scope2_df = pd.read_excel(base_url + "Scope_2.xlsx")

# Add 'Firm_Name' column to ESG data using the ISIN-to-name mapping
scope1_df['Firm_Name'] = scope1_df['ISIN'].map(isin_to_name)
scope2_df['Firm_Name'] = scope2_df['ISIN'].map(isin_to_name)

# Filter ESG data for AMER firms using firm names
scope1_us = scope1_df[scope1_df["Firm_Name"].isin(amer_names)]
scope2_us = scope2_df[scope2_df["Firm_Name"].isin(amer_names)]

# Define ESG coverage check: ≥7 years & ≥5 consecutive years
def check_scope(row):
    # Exclude 'ISIN' and 'Firm_Name' columns to focus on yearly data
    values = row.drop(['ISIN', 'Firm_Name'], errors='ignore').notna().astype(int)
    valid_years = values.sum()
    max_consecutive = values.groupby((values != values.shift()).cumsum()).transform('size') * values
    consecutive_years = max_consecutive.max()
    return (valid_years >= 7) and (consecutive_years >= 5)

# Filter firm names that meet both Scope 1 and Scope 2 criteria
scope1_ok = set(scope1_us[scope1_us.apply(check_scope, axis=1)]["Firm_Name"])
scope2_ok = set(scope2_us[scope2_us.apply(check_scope, axis=1)]["Firm_Name"])
solid_names = scope1_ok & scope2_ok  # Intersection of firm names meeting ESG criteria

# ------------------------------
# STEP 2 – LOAD FINANCIAL DATA
# ------------------------------

# Load price data
raw_prices = pd.read_excel(base_url + "DS_RI_T_USD_M.xlsx")
prices = raw_prices.set_index("ISIN").transpose()
prices.index = pd.to_datetime(prices.index, format="%Y-%m-%d", errors="coerce")

# Replace ISIN columns with firm names
prices.columns = prices.columns.map(isin_to_name)

# Load market cap data
raw_mkt_caps = pd.read_excel(base_url + "DS_MV_T_USD_M.xlsx")
mkt_caps = raw_mkt_caps.set_index("ISIN").transpose()
mkt_caps.index = pd.to_datetime(mkt_caps.index, format="%Y-%m-%d", errors="coerce")

# Replace ISIN columns with firm names
mkt_caps.columns = mkt_caps.columns.map(isin_to_name)

# Load and transform revenue data
raw_revenues = pd.read_excel(base_url + "DS_REV_USD_Y.xlsx")
year_cols = [col for col in raw_revenues.columns if str(col).isdigit()]
revenues_long = raw_revenues.melt(
    id_vars=["ISIN"],
    value_vars=year_cols,
    var_name="Year",
    value_name="Revenue"
)
revenues_long["Date"] = pd.to_datetime(revenues_long["Year"], format="%Y") + pd.offsets.YearEnd(0)
revenues = revenues_long.pivot(index="Date", columns="ISIN", values="Revenue")
# Using backward-filling to assign annual revenue to months within the year
revenues = revenues.resample("M").bfill()

# Replace ISIN columns with firm names
revenues.columns = revenues.columns.map(isin_to_name)

# Load risk-free rate
rf_df = pd.read_excel(base_url + "Risk_Free_Rate.xlsx")
rf_df.columns = ["DateRaw", "RF"]
rf_df["Date"] = pd.to_datetime(rf_df["DateRaw"].astype(str), format="%Y%m")
rf_df.set_index("Date", inplace=True)
rf_series = rf_df["RF"] / 100

# ------------------------------
# STEP 3 – ALIGN AND CLEAN DATA
# ------------------------------

# Filter data to companies that passed ESG screen using firm names
common_names = solid_names & set(prices.columns) & set(mkt_caps.columns) & set(revenues.columns)
prices = prices[list(common_names)]
mkt_caps = mkt_caps[list(common_names)]
revenues = revenues[list(common_names)]

# Clean: interpolate linearly, internally only
prices = prices.replace(0, np.nan).apply(pd.to_numeric, errors="coerce")
prices = prices.interpolate(method="linear", axis=0, limit_area="inside")

mkt_caps = mkt_caps.replace(0, np.nan).apply(pd.to_numeric, errors="coerce")
mkt_caps = mkt_caps.interpolate(method="linear", axis=0, limit_area="inside")

revenues = revenues.replace(0, np.nan).apply(pd.to_numeric, errors="coerce")
revenues = revenues.interpolate(method="linear", axis=0, limit_area="inside")

# Diagnostics after ESG filtering
print("✅ Data loading and ESG filtering complete.")
print(f"Remaining firms after ESG filter: {len(common_names)}")

# Additional filtering: Drop firms with >10% missing data after interpolation
max_missing_ratio = 0.1
for df in [prices, mkt_caps, revenues]:
    missing_ratios = df.isna().mean()
    firms_to_drop = missing_ratios[missing_ratios > max_missing_ratio].index
    df.drop(columns=firms_to_drop, inplace=True)

# Update common_names to reflect dropped firms
common_names = set(prices.columns) & set(mkt_caps.columns) & set(revenues.columns)
prices = prices[list(common_names)]
mkt_caps = mkt_caps[list(common_names)]
revenues = revenues[list(common_names)]

# Diagnostics after missing value filtering
print(f"Remaining firms after missing value filtering: {len(common_names)}")

# Calculate simple returns and handle NaNs
simple_returns = prices.pct_change(fill_method=None)
simple_returns = simple_returns.dropna()   # Drop rows with NaN returns

# Check if risk-free rate covers the period of returns
if not simple_returns.index.isin(rf_series.index).all():
    print("Warning: Risk-free rate does not cover the entire period of returns.")

# Retain original first_available calculation
first_available = simple_returns.notna().apply(lambda x: x[x].index.min())

# ------------------------------
# STEP 4 – NEW FACTOR MODEL FUNCTIONS
# ------------------------------

def construct_factors(returns, market_caps, revenues, risk_free_rate):
    try:
        # 🔧 Backward-fill revenue data (since it's annual) to align with monthly data
        revenues = revenues.bfill(limit=11).ffill(limit=11)

        # === Market (Mkt-RF) Factor ===
        market_weights = market_caps.div(market_caps.sum(axis=1), axis=0)
        market_return = (returns * market_weights).sum(axis=1)
        excess_market_return = market_return - risk_free_rate

        # === Size (SMB) Factor ===
        market_cap_median = market_caps.median(axis=1)
        small = returns[market_caps.lt(market_cap_median, axis=0)]
        big = returns[market_caps.ge(market_cap_median, axis=0)]
        smb = small.mean(axis=1) - big.mean(axis=1)

        # === Value (HML) Factor ===
        # Avoid divide-by-zero issues in revenue-to-market
        safe_mkt_caps = market_caps.copy()
        safe_mkt_caps[safe_mkt_caps <= 0] = np.nan
        revenue_to_market = revenues.div(safe_mkt_caps)

        rm_median = revenue_to_market.median(axis=1)
        high_mask = revenue_to_market.gt(rm_median, axis=0)
        low_mask = revenue_to_market.le(rm_median, axis=0)

        # Require enough valid companies per group (e.g., ≥ 20)
        high_valid = high_mask.sum(axis=1)
        low_valid = low_mask.sum(axis=1)
        valid_dates = (high_valid >= 20) & (low_valid >= 20)

        if valid_dates.sum() == 0:
            print("⚠️ No valid dates with enough high/low revenue-to-market splits for HML.")
            hml = pd.Series(index=returns.index, data=np.nan)
        else:
            high_rm = returns.where(high_mask)
            low_rm = returns.where(low_mask)
            hml_raw = high_rm.mean(axis=1) - low_rm.mean(axis=1)
            hml = hml_raw.where(valid_dates)

        # === Combine all factors ===
        factors = pd.DataFrame({
            'Mkt-RF': excess_market_return,
            'SMB': smb,
            'HML': hml
        }, index=returns.index)

        # Fill gaps to avoid dropped rows later
        factors = factors.ffill().bfill()

        # === Diagnostics ===
        print("Factor Construction Diagnostics:")
        print("Revenue-to-Market: % NaN =", revenue_to_market.isna().mean().mean())
        print("SMB - % NaN:", smb.isna().mean(), " | Std Dev:", smb.std())
        print("HML - % NaN:", hml.isna().mean(), " | Std Dev:", hml.std())
        print("Excess Mkt-RF - % NaN:", excess_market_return.isna().mean())
        print("Factors constructed. Date range:", factors.index.min(), "to", factors.index.max())

        return factors

    except Exception as e:
        print("Error in construct_factors:", e)
        print("returns shape:", returns.shape)
        print("market_caps shape:", market_caps.shape)
        print("revenues shape:", revenues.shape)
        print("risk_free_rate shape:", risk_free_rate.shape)
        return pd.DataFrame()

def estimate_factor_loadings(returns, factors):
    betas = {}
    residuals = {}
    for company in returns.columns:
        y = returns[company].dropna()
        X = factors.loc[y.index]
        common_index = X.dropna().index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        y = y.loc[X.index]
        if len(y) < 60:
            betas[company] = np.full(len(factors.columns), np.nan)
            residuals[company] = np.nan
            continue
        X = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        betas[company] = beta[1:]
        residuals[company] = y - X @ beta
    return betas, residuals

def compute_factor_cov_matrix(returns, factors):
    betas, residuals = estimate_factor_loadings(returns, factors)
    companies = [c for c in betas if not np.isnan(betas[c]).any()]

    if not companies:
        print("⚠️ No valid companies after filtering for betas.")
        return np.array([]), []

    B = np.array([betas[c] for c in companies])
    print("B shape:", B.shape)
    print("Factors shape:", factors.shape)

    factor_sub = factors.loc[returns.index].dropna()
    if len(factor_sub) < 2:
        print("❌ Not enough data to compute factor covariance matrix.")
        return np.array([]), []

    F = np.cov(factor_sub.T, ddof=1)

    # Fix for single-factor model: ensure F is 2D
    if F.ndim == 0:
        F = np.array([[F]])
    elif F.ndim == 1:
        F = F.reshape((1, 1))

    print("Corrected F shape:", F.shape)

    D = np.diag([np.var(residuals[c], ddof=1) for c in companies])

    cov_matrix = B @ F @ B.T + D
    return cov_matrix, companies

# ------------------------------
# STEP 5 – MVP WEIGHT FUNCTION
# ------------------------------

def compute_mvp_weights(returns_window, mkt_caps_window, revenues_window, rf_window, method='lw',
                        max_weight=0.05, prev_weights=None, turnover_limit=None, transaction_cost=None):
    """Compute Minimum Variance Portfolio weights using specified method."""
    sufficient_data = returns_window.count() >= 120
    returns_window = returns_window.loc[:, sufficient_data].dropna(axis=1, how="any")
    mkt_caps_window = mkt_caps_window.loc[:, sufficient_data].reindex(returns_window.index, method='ffill')
    revenues_window = revenues_window.loc[:, sufficient_data].reindex(returns_window.index, method='ffill')
    rf_window = rf_window.reindex(returns_window.index, method='ffill').fillna(0)

    if returns_window.shape[1] < 2:
        return pd.Series(np.nan)

    assets = returns_window.columns
    n = len(assets)

    # Covariance matrix estimation
    if method == 'lw':
        lw = LedoitWolf()
        lw.fit(returns_window)
        cov_matrix = lw.covariance_

    elif method == 'pinv':
        cov_matrix = np.cov(returns_window.values, rowvar=False)
        cov_pinv = np.linalg.pinv(cov_matrix)
        ones = np.ones(n)
        weights = cov_pinv @ ones / (ones.T @ cov_pinv @ ones)
        return pd.Series(weights, index=assets)

    elif method == 'factor_1':
        factors = construct_factors(returns_window, mkt_caps_window, revenues_window, rf_window)[['Mkt-RF']]
        cov_matrix, companies = compute_factor_cov_matrix(returns_window[assets], factors)
        if not companies:
            print("⚠️ Skipping optimization: No valid assets for factor_1")
            return pd.Series(np.nan)
        assets = companies

    elif method == 'factor_3':
        factors = construct_factors(returns_window, mkt_caps_window, revenues_window, rf_window)
        cov_matrix, companies = compute_factor_cov_matrix(returns_window[assets], factors)
        if not companies:
            print("⚠️ Skipping optimization: No valid assets for factor_3")
            return pd.Series(np.nan)
        assets = companies

    elif method == 'identity':
        sample_cov = np.cov(returns_window.values, rowvar=False)
        avg_var = np.trace(sample_cov) / n
        identity = np.eye(n) * avg_var
        shrinkage_intensity = 0.2
        cov_matrix = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * identity

    else:
        raise ValueError("Method must be 'lw', 'pinv', 'factor_1', 'factor_3', or 'identity'.")

    # Optimization
    w = cp.Variable(len(assets))
    constraints = [cp.sum(w) == 1, w >= 0]
    if max_weight:
        constraints.append(w <= max_weight)
    if turnover_limit and prev_weights is not None:
        aligned_prev = prev_weights.reindex(assets).fillna(0).values
        constraints.append(cp.norm1(w - aligned_prev) <= turnover_limit)

    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov_matrix)))
    if transaction_cost and prev_weights is not None:
        aligned_prev = prev_weights.reindex(assets).fillna(0).values
        trade_cost = transaction_cost * cp.norm1(w - aligned_prev)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov_matrix)) + trade_cost)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return pd.Series(w.value, index=assets) if w.value is not None else pd.Series(np.nan, index=assets)

# ------------------------------
# STEP 6 – ROLLING OPTIMIZATION
# ------------------------------

start_year = 2013
end_year = 2023
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

def compute_portfolio_returns(returns_df, weights_dict=None, mkt_caps_df=None, mode='mvp'):
    """Compute portfolio returns for MVP or VW portfolios."""
    returns = []
    dates = []

    for year in range(start_year, end_year + 1):
        start = pd.Timestamp(f"{year + 1}-01-01")
        end = pd.Timestamp(f"{year + 1}-12-31")
        R = returns_df[(returns_df.index >= start) & (returns_df.index <= end)]

        if mode == 'mvp':
            alpha = weights_dict.get(year)
            if alpha is None or alpha.empty:
                continue
            alpha = alpha / alpha.sum()

        for date in R.index:
            r_t = R.loc[date]

            if mode == 'mvp':
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

            elif mode == 'vw':
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

    return pd.Series(returns, index=pd.to_datetime(dates))

# Compute returns for all portfolios
mvp_series_all = {method: compute_portfolio_returns(simple_returns, mvp_weights_all[method], mode='mvp')
                  for method in methods}
vw_series = compute_portfolio_returns(simple_returns, mkt_caps_df=mkt_caps, mode='vw')

# ------------------------------
# STEP 8 – PERFORMANCE METRICS
# ------------------------------

rf_aligned = rf_series.reindex(vw_series.index, method="ffill")

def compute_metrics(r, rf):
    """Calculate portfolio performance metrics."""
    r, rf = r.dropna(), rf.dropna()
    aligned_dates = r.index.intersection(rf.index)
    r, rf = r.loc[aligned_dates], rf.loc[aligned_dates]

    if len(r) == 0:
        return [np.nan] * 7

    excess_r = r - rf
    n_months = len(r)

    ann_avg_ret = (1 + r).prod() ** (12 / n_months) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = excess_r.mean() / r.std() * np.sqrt(12) if r.std() != 0 else np.nan
    rmin, rmax = r.min(), r.max()
    cumulative_return = (1 + r).prod() - 1

    cum_series = (1 + r).cumprod()
    drawdown = (cum_series - cum_series.cummax()) / cum_series.cummax()
    max_drawdown = drawdown.min()

    return ann_avg_ret, ann_vol, cumulative_return, sharpe, rmin, rmax, max_drawdown

# Compute metrics for all portfolios
metrics = {method: compute_metrics(mvp_series_all[method], rf_aligned) for method in methods}
metrics['VW'] = compute_metrics(vw_series, rf_aligned)

# Print metrics in a formatted way
print("\n=== Portfolio Performance Metrics ===")
for name in metrics.keys():
    ann_avg_ret, ann_vol, cum_ret, sharpe, rmin, rmax, max_dd = metrics[name]
    print(f"\n{name} Portfolio:")
    print(f"{'Annualized Average Return':<30}: {ann_avg_ret:>8.4f}")
    print(f"{'Annualized Volatility':<30}: {ann_vol:>8.4f}")
    print(f"{'Cumulative Total Return':<30}: {cum_ret:>8.4f}")
    print(f"{'Sharpe Ratio':<30}: {sharpe:>8.4f}")
    print(f"{'Minimum Monthly Return':<30}: {rmin:>8.4f}")
    print(f"{'Maximum Monthly Return':<30}: {rmax:>8.4f}")
    print(f"{'Maximum Drawdown':<30}: {max_dd:>8.4f}")

# ------------------------------
# STEP 9 – PLOT CUMULATIVE RETURNS
# ------------------------------

plt.figure(figsize=(12, 6))
for method, series in mvp_series_all.items():
    if not series.dropna().empty:
        (1 + series).cumprod().plot(label=f"MVP ({method})")
(1 + vw_series).cumprod().plot(label="VW Portfolio")
plt.title("Cumulative Returns (2013–2023)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# STEP 10 – DISPLAY PORTFOLIO METRICS AS A TABLE
# ------------------------------

# Create a DataFrame for the metrics
metric_names = [
    "Annualized Average Return",
    "Annualized Volatility",
    "Cumulative Total Return",
    "Sharpe Ratio",
    "Minimum Monthly Return",
    "Maximum Monthly Return",
    "Maximum Drawdown"
]
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=metric_names)

# Option 1: Display the DataFrame as a formatted text table
print("\n=== Portfolio Metrics Table (Text Format) ===")
print(metrics_df.to_string(formatters={
    "Annualized Average Return": "{:.4f}".format,
    "Annualized Volatility": "{:.4f}".format,
    "Cumulative Total Return": "{:.4f}".format,
    "Sharpe Ratio": "{:.4f}".format,
    "Minimum Monthly Return": "{:.4f}".format,
    "Maximum Monthly Return": "{:.4f}".format,
    "Maximum Drawdown": "{:.4f}".format
}))

# Option 2: Plot the metrics as a table using matplotlib
plt.figure(figsize=(12, 4))
plt.axis('off')  # Hide axes
table = plt.table(
    cellText=metrics_df.round(4).values,
    colLabels=metrics_df.columns,
    rowLabels=metrics_df.index,
    loc='center',
    cellLoc='center',
    colWidths=[0.15] * len(metrics_df.columns)
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)  # Adjust table size
plt.title("Portfolio Performance Metrics Table", fontsize=12, pad=20)
plt.show()

# ------------------------------
# STEP 11 – CARBON METRICS FOR P_mv_oos
# ------------------------------

# Create long-format emissions data
scope1_long = scope1_us.melt(
    id_vars=['Firm_Name'],
    value_vars=[col for col in scope1_us.columns if str(col).isdigit()],
    var_name='Year',
    value_name='Scope1'
)
scope2_long = scope2_us.melt(
    id_vars=['Firm_Name'],
    value_vars=[col for col in scope2_us.columns if str(col).isdigit()],
    var_name='Year',
    value_name='Scope2'
)

# Merge Scope 1 and Scope 2 emissions
emissions_long = pd.merge(scope1_long, scope2_long, on=['Firm_Name', 'Year'], how='inner')
emissions_long['E'] = emissions_long['Scope1'] + emissions_long['Scope2']
emissions_long['Year'] = emissions_long['Year'].astype(int)

# Create long-format revenue data (already pivoted and resampled monthly in STEP 2)
# We need annual revenues, so take the last value of each year
revenues_annual = revenues.groupby(revenues.index.year).last()

# Create annual market caps (last value of each year)
mkt_caps_annual = mkt_caps.groupby(mkt_caps.index.year).last()

# Compute WACI and Carbon Footprint for P_mv_oos using 'lw' method
method = 'lw'
carbon_footprints = {}
waci_values = {}

for Y in range(2013, 2023):  # From 2013 to 2023
    # Get weights from previous year (weights for year Y are set at end of Y-1)
    weights_Y_minus_1 = mvp_weights_all[method].get(Y - 1, pd.Series())
    if weights_Y_minus_1.empty:
        print(f"No weights available for year {Y-1}")
        continue
    firms_Y = weights_Y_minus_1.index

    # Get emissions for year Y
    emissions_Y = emissions_long[emissions_long['Year'] == Y]
    emissions_Y = emissions_Y.set_index('Firm_Name')['E'].reindex(firms_Y).dropna()

    # Get market cap for year Y (end of year)
    if Y not in mkt_caps_annual.index:
        print(f"No market cap data for year {Y}")
        continue
    mkt_cap_Y = mkt_caps_annual.loc[Y].reindex(firms_Y).dropna()

    # Get revenues for year Y (end of year)
    if Y not in revenues_annual.index:
        print(f"No revenue data for year {Y}")
        continue
    revenue_Y = revenues_annual.loc[Y].reindex(firms_Y).dropna()

    # Align data: keep only firms with weights, emissions, market cap, and revenue
    common_firms = weights_Y_minus_1.index.intersection(emissions_Y.index).intersection(mkt_cap_Y.index).intersection(revenue_Y.index)
    if common_firms.empty:
        print(f"No common firms for year {Y}")
        continue

    weights_Y_minus_1 = weights_Y_minus_1.reindex(common_firms)
    emissions_Y = emissions_Y.reindex(common_firms)
    mkt_cap_Y = mkt_cap_Y.reindex(common_firms)
    revenue_Y = revenue_Y.reindex(common_firms)

    # Compute Carbon Footprint (tons CO2e per million USD invested)
    emissions_over_mkt_cap = emissions_Y / mkt_cap_Y  # mkt_caps in millions USD
    cf_Y = (weights_Y_minus_1 * emissions_over_mkt_cap).sum()
    carbon_footprints[Y] = cf_Y

    # Compute Carbon Intensity (CI_i,Y) = E_i,Y / Revenue_i,Y (tons CO2e per million USD of revenue)
    # Revenues are in thousands USD, so convert to millions: divide by 1,000
    revenue_Y_millions = revenue_Y / 1_000  # thousands to millions
    ci_Y = emissions_Y / revenue_Y_millions  # tons CO2e per million USD

    # Compute WACI (tons CO2e per million USD of revenue)
    waci_Y = (weights_Y_minus_1 * ci_Y).sum()
    waci_values[Y] = waci_Y

# Display the results
print("\n=== Carbon Footprint for P_mv_oos (tons CO2e per million USD invested) ===")
for year, cf in carbon_footprints.items():
    print(f"Year {year}: {cf:.2f}")

print("\n=== Weighted-Average Carbon Intensity for P_mv_oos (tons CO2e per million USD of revenue) ===")
for year, waci in waci_values.items():
    print(f"Year {year}: {waci:.2f}")

# Plot CF and WACI over time
plt.figure(figsize=(10, 6))
plt.plot(list(carbon_footprints.keys()), list(carbon_footprints.values()), marker='o', label='Carbon Footprint (tons CO2e per million USD invested)')
plt.plot(list(waci_values.keys()), list(waci_values.values()), marker='s', label='WACI (tons CO2e per million USD of revenue)')
plt.title("Carbon Footprint and WACI for P_mv_oos (2013–2023)")
plt.xlabel("Year")
plt.ylabel("Metrics (tons CO2e per million USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------
# STEP 12 – PORTFOLIO WITH 50% CARBON FOOTPRINT REDUCTION (2.2)
# ------------------------------

# Compute c_Y vectors for each year (E_{i,Y} / Cap_{i,Y})
c_vectors = {}
for Y in range(2013, 2023):
    emissions_Y = emissions_long[emissions_long['Year'] == Y].set_index('Firm_Name')['E']
    if Y not in mkt_caps_annual.index:
        continue
    mkt_cap_Y = mkt_caps_annual.loc[Y]
    common_firms = emissions_Y.index.intersection(mkt_cap_Y.index)
    c_Y = emissions_Y.reindex(common_firms) / mkt_cap_Y.reindex(common_firms)
    c_vectors[Y] = c_Y.dropna()

# Compute carbon footprints for the original portfolio (already done in STEP 10, adjust if needed)
# Ensure carbon_footprints is available from STEP 10
carbon_footprints_original = carbon_footprints  # From STEP 10

# Modify compute_mvp_weights to include carbon constraint
def compute_mvp_weights(returns_window, mkt_caps_window, revenues_window, rf_window, method='lw',
                        max_weight=0.05, prev_weights=None, turnover_limit=None, transaction_cost=None,
                        carbon_constraint=None):
    """Compute Minimum Variance Portfolio weights with optional carbon footprint constraint."""
    sufficient_data = returns_window.count() >= 120
    returns_window = returns_window.loc[:, sufficient_data].dropna(axis=1, how="any")
    mkt_caps_window = mkt_caps_window.loc[:, sufficient_data].reindex(returns_window.index, method='ffill')
    revenues_window = revenues_window.loc[:, sufficient_data].reindex(returns_window.index, method='ffill')
    rf_window = rf_window.reindex(returns_window.index, method='ffill').fillna(0)

    if returns_window.shape[1] < 2:
        return pd.Series(np.nan)

    assets = returns_window.columns
    n = len(assets)

    # Covariance matrix estimation (Ledoit-Wolf)
    if method == 'lw':
        lw = LedoitWolf()
        lw.fit(returns_window)
        cov_matrix = lw.covariance_
    else:
        raise ValueError("Only 'lw' method is implemented for 2.2 in this example.")

    # Optimization with carbon constraint
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    if turnover_limit and prev_weights is not None:
        aligned_prev = prev_weights.reindex(assets).fillna(0).values
        constraints.append(cp.norm1(w - aligned_prev) <= turnover_limit)
    
    if carbon_constraint is not None:
        c, threshold = carbon_constraint
        c_aligned = c.reindex(assets).fillna(0).values  # Align with assets, assume 0 if missing
        constraints.append(w @ c_aligned <= threshold)

    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov_matrix)))
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        if w.value is None:
            print(f"Optimization failed for method {method}")
            return pd.Series(np.nan, index=assets)
    except Exception as e:
        print(f"Optimization error: {e}")
        return pd.Series(np.nan, index=assets)

    return pd.Series(w.value, index=assets)

# Rolling optimization for constrained portfolio
mvp_weights_constrained = {}
prev_weights = None
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

    # Set carbon constraint for year Y (weights for Y+1)
    Y = year
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

# Compute carbon footprints for constrained portfolio (for verification)
carbon_footprints_constrained = {}
for Y in range(2013, 2023):
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

# Update metrics table
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=metric_names)
print("\n=== Updated Portfolio Metrics Table ===")
print(metrics_df.to_string(formatters={
    k: "{:.4f}".format for k in metric_names
}))

# Plot cumulative returns
plt.figure(figsize=(12, 6))
(1 + mvp_series_all['lw']).cumprod().plot(label="Original MVP (lw)")
(1 + mvp_series_constrained).cumprod().plot(label="Constrained MVP (lw, 0.5 CF)")
(1 + vw_series).cumprod().plot(label="VW Portfolio")
plt.title("Cumulative Returns Comparison (2013–2023)")
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

# Function to compute value-weighted benchmark weights for a given year
def compute_vw_weights(mkt_caps_annual, year):
    """
    Compute value-weighted weights based on market capitalization for a given year.
    
    Args:
        mkt_caps_annual (pd.DataFrame): Annual market cap data with years as index
        year (int): Year for which to compute weights
    
    Returns:
        pd.Series: Value-weighted weights, or empty Series if no data
    """
    if year not in mkt_caps_annual.index:
        print(f"No market cap data for year {year}")
        return pd.Series()
    mkt_cap_Y = mkt_caps_annual.loc[year]
    total_cap = mkt_cap_Y.sum()
    if total_cap == 0:
        print(f"Total market cap is zero for year {year}")
        return pd.Series()
    vw_weights = mkt_cap_Y / total_cap
    return vw_weights

# Compute benchmark carbon footprint for each year
benchmark_cf = {}
for Y in range(2013, 2023):
    emissions_Y = emissions_long[emissions_long['Year'] == Y].set_index('Firm_Name')['E']
    if Y not in mkt_caps_annual.index:
        print(f"Skipping year {Y}: No market cap data")
        continue
    mkt_cap_Y = mkt_caps_annual.loc[Y]
    common_firms = emissions_Y.index.intersection(mkt_cap_Y.index)
    emissions_Y = emissions_Y.reindex(common_firms)
    total_cap_Y = mkt_cap_Y.reindex(common_firms).sum()
    cf_vw_Y = emissions_Y.sum() / total_cap_Y if total_cap_Y != 0 else np.nan
    benchmark_cf[Y] = cf_vw_Y

# Define optimization function for tracking portfolio with carbon constraint
def compute_tracking_weights(returns_window, mkt_caps_window, revenues_window, rf_window, vw_weights,
                            c_vector, cf_threshold, method='lw', max_weight=0.05):
    """
    Compute portfolio weights that minimize tracking error with a carbon footprint constraint.
    
    Args:
        returns_window (pd.DataFrame): Historical returns for the window
        mkt_caps_window (pd.DataFrame): Market caps for the window (unused here but kept for consistency)
        revenues_window (pd.DataFrame): Revenues for the window (unused here but kept for consistency)
        rf_window (pd.Series): Risk-free rate for the window (unused here but kept for consistency)
        vw_weights (pd.Series): Value-weighted benchmark weights
        c_vector (pd.Series): Carbon intensity vector for assets
        cf_threshold (float): Maximum allowed carbon footprint
        method (str): Covariance estimation method ('lw' for Ledoit-Wolf)
        max_weight (float): Maximum weight per asset (default 0.05)
    
    Returns:
        pd.Series: Optimized weights, or NaN-filled Series if optimization fails
    """
    # Filter for assets with sufficient data (at least 120 observations) and no NaNs
    sufficient_data = returns_window.count() >= 120
    returns_window = returns_window.loc[:, sufficient_data].dropna(axis=1, how="any")
    
    # Check if there's enough data to proceed
    if returns_window.empty or returns_window.shape[1] < 2:
        print(f"Insufficient data for optimization: returns_window shape = {returns_window.shape}")
        return pd.Series(np.nan, index=vw_weights.index)
    
    assets = returns_window.columns
    n = len(assets)
    
    # Align benchmark weights and carbon vector with available assets
    vw_weights_aligned = vw_weights.reindex(assets).fillna(0).values
    c_aligned = c_vector.reindex(assets).fillna(0).values
    
    # Estimate covariance matrix using Ledoit-Wolf
    if method == 'lw':
        lw = LedoitWolf()
        try:
            lw.fit(returns_window)
            cov_matrix = lw.covariance_
        except ValueError as e:
            print(f"LedoitWolf fit failed: {e}")
            return pd.Series(np.nan, index=assets)
    else:
        raise ValueError("Only 'lw' method is supported in this implementation")
    
    # Define and solve optimization problem
    w = cp.Variable(n)
    diff = w - vw_weights_aligned
    objective = cp.Minimize(cp.quad_form(diff, cp.psd_wrap(cov_matrix)))
    constraints = [
        cp.sum(w) == 1,          # Weights sum to 1
        w >= 0,                  # No short-selling
        w @ c_aligned <= cf_threshold  # Carbon footprint constraint
    ]
    if max_weight:
        constraints.append(w <= max_weight)  # Maximum weight constraint
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        if w.value is None:
            print("Optimization failed to find a solution")
            return pd.Series(np.nan, index=assets)
    except Exception as e:
        print(f"Optimization error: {e}")
        return pd.Series(np.nan, index=assets)
    
    return pd.Series(w.value, index=assets)

# Rolling optimization for tracking portfolio
start_year = 2013  # Adjust based on your data availability
end_year = 2023    # Adjust based on your data availability
tracking_weights = {}
method = 'lw'

for year in range(start_year, end_year + 1):
    # Define 10-year lookback window
    window_start = pd.Timestamp(f"{year - 10}-01-01")
    window_end = pd.Timestamp(f"{year - 1}-12-31")
    
    # Filter for assets available at window_start
    eligible_assets = first_available[first_available <= window_start].index
    returns_window = simple_returns[(simple_returns.index >= window_start) &
                                   (simple_returns.index <= window_end)][eligible_assets]
    
    # Placeholder for unused inputs (align with your data structure)
    mkt_caps_window = mkt_caps[(mkt_caps.index >= window_start) &
                              (mkt_caps.index <= window_end)][eligible_assets]
    revenues_window = revenues[(revenues.index >= window_start) &
                              (revenues.index <= window_end)][eligible_assets]
    rf_window = rf_series[(rf_series.index >= window_start) &
                         (rf_series.index <= window_end)]
    
    # Compute benchmark weights
    vw_weights_Y = compute_vw_weights(mkt_caps_annual, year)
    if vw_weights_Y.empty:
        print(f"Skipping year {year}: No benchmark weights")
        continue
    
    # Get carbon intensity vector
    c_Y = c_vectors.get(year, pd.Series())
    if c_Y.empty:
        print(f"Skipping year {year}: No carbon intensity data")
        continue
    
    # Set carbon footprint threshold (50% of benchmark)
    cf_vw_Y = benchmark_cf.get(year, np.nan)
    if np.isnan(cf_vw_Y):
        print(f"Skipping year {year}: No benchmark carbon footprint")
        continue
    cf_threshold = 0.5 * cf_vw_Y
    
    # Compute optimized weights
    weights = compute_tracking_weights(
        returns_window, mkt_caps_window, revenues_window, rf_window,
        vw_weights_Y, c_Y, cf_threshold, method=method, max_weight=0.05
    )
    tracking_weights[year] = weights.dropna()

# Compute ex-post portfolio returns
tracking_series = compute_portfolio_returns(simple_returns, tracking_weights, mode='mvp')

# Compute performance metrics
metrics['tracking_0.5'] = compute_metrics(tracking_series, rf_aligned)

# Compute carbon footprints for verification
carbon_footprints_tracking = {}
for Y in range(2013, 2023):
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
plt.title("Cumulative Returns Comparison (2013–2023)")
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
# STEP 14 – COMPARAISON ANALYSIS 50% CARBON FOOTPRINT REDUCTION (2.4)
# ------------------------------


# Align risk-free rate with portfolio returns
rf_aligned_mv = rf_series.reindex(mvp_series_all['lw'].index, method="ffill")
rf_aligned_mv_constr = rf_series.reindex(mvp_series_constrained.index, method="ffill")
rf_aligned_vw = rf_series.reindex(vw_series.index, method="ffill")
rf_aligned_track = rf_series.reindex(tracking_series.index, method="ffill")

# Function to compute performance metrics (reused from previous code)
def compute_metrics(r, rf):
    r, rf = r.dropna(), rf.dropna()
    aligned_dates = r.index.intersection(rf.index)
    r, rf = r.loc[aligned_dates], rf.loc[aligned_dates]
    if len(r) == 0:
        return [np.nan] * 7
    excess_r = r - rf
    n_months = len(r)
    ann_avg_ret = (1 + r).prod() ** (12 / n_months) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = excess_r.mean() / r.std() * np.sqrt(12) if r.std() != 0 else np.nan
    rmin, rmax = r.min(), r.max()
    cumulative_return = (1 + r).prod() - 1
    cum_series = (1 + r).cumprod()
    drawdown = (cum_series - cum_series.cummax()) / cum_series.cummax()
    max_drawdown = drawdown.min()
    return ann_avg_ret, ann_vol, cumulative_return, sharpe, rmin, rmax, max_drawdown

# Compute metrics for all four portfolios
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
print("\n=== Financial Performance Comparison (2013–2023) ===")
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

# Plot cumulative returns for both pairs
plt.figure(figsize=(12, 6))
(1 + mvp_series_all['lw']).cumprod().plot(label="P_oos^(mv)")
(1 + mvp_series_constrained).cumprod().plot(label="P_oos^(mv)(0.5)")
(1 + vw_series).cumprod().plot(label="P^(vw)")
(1 + tracking_series).cumprod().plot(label="P_oos^(vw)(0.5)")
plt.title("Cumulative Returns Comparison (2013–2023)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot carbon footprints for both pairs
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

# Compute base year carbon footprint for the benchmark (P^(vw))
emissions_Y0 = emissions_long[emissions_long['Year'] == Y0].set_index('Firm_Name')['E']
if Y0 not in mkt_caps_annual.index:
    raise ValueError(f"No market cap data for base year {Y0}")
mkt_cap_Y0 = mkt_caps_annual.loc[Y0]
common_firms_Y0 = emissions_Y0.index.intersection(mkt_cap_Y0.index)
emissions_Y0 = emissions_Y0.reindex(common_firms_Y0)
total_cap_Y0 = mkt_cap_Y0.reindex(common_firms_Y0).sum()
CF_Y0_vw = emissions_Y0.sum() / total_cap_Y0 if total_cap_Y0 != 0 else np.nan
print(f"Base Year (2013) Benchmark Carbon Footprint: {CF_Y0_vw:.2f} tons CO2e per million USD")

# Function to compute annual Net Zero carbon footprint targets
def compute_nz_target(Y, CF_Y0, theta):
    return (1 - theta) ** (Y - Y0 + 1) * CF_Y0

# Initialize metrics dictionary if not already defined
metrics = {} if 'metrics' not in globals() else metrics

# Rolling optimization for Net Zero portfolio
nz_weights = {}
method = 'lw'  # Using Ledoit-Wolf covariance estimation
start_year = 2013
end_year = 2023

for year in range(start_year, end_year + 1):
    # Define 10-year lookback window for covariance estimation
    window_start = pd.Timestamp(f"{year - 10}-01-01")
    window_end = pd.Timestamp(f"{year - 1}-12-31")
    eligible_assets = first_available[first_available <= window_start].index
    returns_window = simple_returns[(simple_returns.index >= window_start) &
                                   (simple_returns.index <= window_end)][eligible_assets]
    
    # Skip if insufficient data
    if returns_window.empty or returns_window.shape[1] < 2:
        print(f"Skipping year {year}: Insufficient data for optimization")
        nz_weights[year] = pd.Series(np.nan, index=eligible_assets)
        continue
    
    # Compute value-weighted benchmark weights for year Y
    vw_weights_Y = compute_vw_weights(mkt_caps_annual, year)
    if vw_weights_Y.empty:
        print(f"Skipping year {year}: No benchmark weights")
        nz_weights[year] = pd.Series(np.nan, index=eligible_assets)
        continue
    
    # Get carbon intensity vector for year Y
    c_Y = c_vectors.get(year, pd.Series())
    if c_Y.empty:
        print(f"Skipping year {year}: No carbon intensity data")
        nz_weights[year] = pd.Series(np.nan, index=eligible_assets)
        continue
    
    # Compute Net Zero carbon footprint target for year Y
    cf_target_Y = compute_nz_target(year, CF_Y0_vw, theta)
    
    # Align data
    assets = returns_window.columns
    vw_weights_aligned = vw_weights_Y.reindex(assets).fillna(0).values
    c_aligned = c_Y.reindex(assets).fillna(0).values
    
    # Estimate covariance matrix using Ledoit-Wolf
    lw = LedoitWolf()
    try:
        lw.fit(returns_window)
        cov_matrix = lw.covariance_
    except ValueError as e:
        print(f"LedoitWolf fit failed for year {year}: {e}")
        nz_weights[year] = pd.Series(np.nan, index=assets)
        continue
    
    # Optimization: Minimize tracking error with NZ carbon constraint
    w = cp.Variable(len(assets))
    diff = w - vw_weights_aligned
    objective = cp.Minimize(cp.quad_form(diff, cp.psd_wrap(cov_matrix)))
    constraints = [
        cp.sum(w) == 1,          # Fully invested
        w >= 0,                  # Long-only
        w @ c_aligned <= cf_target_Y,  # Carbon footprint constraint
        w <= 0.05                # Maximum weight per asset
    ]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        if w.value is None:
            print(f"Optimization failed for year {year}")
            nz_weights[year] = pd.Series(np.nan, index=assets)
        else:
            print(f"Optimization successful for year {year}")
            nz_weights[year] = pd.Series(w.value, index=assets)
    except Exception as e:
        print(f"Optimization error for year {year}: {e}")
        nz_weights[year] = pd.Series(np.nan, index=assets)

# Check if any optimization succeeded
if not any(w.notna().any() for w in nz_weights.values()):
    print("Warning: No successful optimizations for NZ portfolio. NZ series will be empty.")
    nz_series = pd.Series()
else:
    # Compute ex-post returns for NZ portfolio
    nz_series = compute_portfolio_returns(simple_returns, nz_weights, mode='mvp')

# Compute performance metrics for NZ portfolio if series is valid
if not nz_series.empty:
    metrics['nz'] = compute_metrics(nz_series, rf_series.reindex(nz_series.index, method='ffill'))
else:
    print("NZ series is empty; skipping metrics computation for NZ portfolio.")

# Rebuild metrics_df with all portfolios
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=metric_names)

# Compute carbon footprints for NZ portfolio (verification)
carbon_footprints_nz = {}
for Y in range(2013, 2023):  # From 2013 to 2023, using weights from Y-1 for year Y
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

# Display NZ portfolio metrics if available
print("\n=== Net Zero Portfolio Metrics ===")
if 'nz' in metrics_df.index:
    formatted_metrics = metrics_df.loc['nz'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    print(formatted_metrics.to_string())
else:
    print("NZ portfolio metrics not available.")

# Display carbon footprint verification
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
if not nz_series.empty:
    (1 + nz_series).cumprod().plot(label="NZ Portfolio")
plt.title("Cumulative Returns: VW vs NZ Portfolio (2013–2023)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot carbon footprints with targets
if carbon_footprints_nz:
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
