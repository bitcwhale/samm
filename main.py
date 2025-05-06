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
amer_df = static_df[static_df["Region"] == "AMER"]
amer_isins = set(amer_df["ISIN"].unique())

# Load Scope 1 and Scope 2 emissions
scope1_df = pd.read_excel(base_url + "Scope_1.xlsx")
scope2_df = pd.read_excel(base_url + "Scope_2.xlsx")
scope1_us = scope1_df[scope1_df["ISIN"].isin(amer_isins)]
scope2_us = scope2_df[scope2_df["ISIN"].isin(amer_isins)]

# Define ESG coverage check: ≥7 years & ≥5 consecutive years
def check_scope(row):
    values = row.iloc[2:].notna().astype(int)
    valid_years = values.sum()
    max_consecutive = values.groupby((values != values.shift()).cumsum()).transform('size') * values
    consecutive_years = max_consecutive.max()
    return (valid_years >= 7) and (consecutive_years >= 5)

# Filter ISINs that meet both Scope 1 and Scope 2 criteria
scope1_ok = set(scope1_us[scope1_us.apply(check_scope, axis=1)]["ISIN"])
scope2_ok = set(scope2_us[scope2_us.apply(check_scope, axis=1)]["ISIN"])
solid_isins = amer_isins & scope1_ok & scope2_ok

# ------------------------------
# STEP 2 – LOAD FINANCIAL DATA
# ------------------------------

# Load price data
raw_prices = pd.read_excel(base_url + "DS_RI_T_USD_M.xlsx")
prices = raw_prices.set_index("ISIN").transpose()
prices.index = pd.to_datetime(prices.index, format="%Y-%m-%d", errors="coerce")

# Load market cap data
raw_mkt_caps = pd.read_excel(base_url + "DS_MV_T_USD_M.xlsx")
mkt_caps = raw_mkt_caps.set_index("ISIN").transpose()
mkt_caps.index = pd.to_datetime(mkt_caps.index, format="%Y-%m-%d", errors="coerce")

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
revenues = revenues.resample("M").ffill()

# Load risk-free rate
rf_df = pd.read_excel(base_url + "Risk_Free_Rate.xlsx")
rf_df.columns = ["DateRaw", "RF"]
rf_df["Date"] = pd.to_datetime(rf_df["DateRaw"].astype(str), format="%Y%m")
rf_df.set_index("Date", inplace=True)
rf_series = rf_df["RF"] / 100

# ------------------------------
# STEP 3 – ALIGN AND CLEAN DATA
# ------------------------------

# Filter data to companies that passed ESG screen
common_isins = solid_isins & set(prices.columns) & set(mkt_caps.columns) & set(revenues.columns)
prices = prices[list(common_isins)]
mkt_caps = mkt_caps[list(common_isins)]
revenues = revenues[list(common_isins)]

# Clean: interpolate linearly, internally only
prices = prices.replace(0, np.nan).apply(pd.to_numeric, errors="coerce")
prices = prices.interpolate(method="linear", axis=0, limit_area="inside")

mkt_caps = mkt_caps.replace(0, np.nan).apply(pd.to_numeric, errors="coerce")
mkt_caps = mkt_caps.interpolate(method="linear", axis=0, limit_area="inside")

revenues = revenues.replace(0, np.nan).apply(pd.to_numeric, errors="coerce")
revenues = revenues.interpolate(method="linear", axis=0, limit_area="inside")

# Diagnostics
print("✅ Data loading and ESG filtering complete.")
print(f"Remaining ISINs after ESG filter: {len(common_isins)}")
# Calculate simple returns
simple_returns = prices.pct_change(fill_method=None)
first_available = simple_returns.notna().apply(lambda x: x[x].index.min())


# ------------------------------
# **NEW FACTOR MODEL FUNCTIONS**
# ------------------------------

def construct_factors(returns, market_caps, revenues, risk_free_rate):
    try:
        # 🔧 Forward-fill revenue data (since it's annual) to fill monthly gaps
        revenues = revenues.ffill(limit=11).bfill(limit=11)

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
        #print("Revenues coverage:")
        #print(revenues.notna().sum().sort_values().tail(10))
        #print("Market caps coverage:")
        #print(market_caps.notna().sum().sort_values().tail(10))
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
# **STEP 2 – MVP WEIGHT FUNCTION**
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
# **STEP 3 – ROLLING OPTIMIZATION**
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


# [Vorheriger Code bleibt unverändert bis STEP 3]

# ------------------------------
# **STEP 4 – EX-POST RETURNS**
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
# **STEP 5 – PERFORMANCE METRICS**
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
# **STEP 6 – PLOT CUMULATIVE RETURNS**
# ------------------------------

plt.figure(figsize=(12, 6))
for method, series in mvp_series_all.items():
    if not series.dropna().empty:
        (1 + series).cumprod().plot(label=f"MVP ({method})")
(1 + vw_series).cumprod().plot(label="VW Portfolio")
plt.title("Cumulative Returns (2014–2024)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Assuming `metrics` is a dict with tuples
df = pd.DataFrame.from_dict(metrics, orient='index', columns=[
    "Annualized Average Return", "Annualized Volatility", "Cumulative Total Return",
    "Sharpe Ratio", "Minimum Monthly Return", "Maximum Monthly Return", "Maximum Drawdown"
])

# Set up a basic matplotlib figure
fig, ax = plt.subplots(figsize=(12, 0.5 * len(df)))
ax.axis('off')

# Plot the DataFrame as a table
table = ax.table(cellText=df.round(4).values,
                 colLabels=df.columns,
                 rowLabels=df.index,
                 loc='center',
                 cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.title("Portfolio Performance Metrics", fontsize=14, pad=20)
plt.tight_layout()
plt.show()
plt.show()

