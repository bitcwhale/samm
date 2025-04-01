import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Define base URL for GitHub raw data files
base_url = "https://raw.githubusercontent.com/bitcwhale/samm/main/"

# ------------------------------
# **STEP 1 – LOAD AND CLEAN DATA**
# ------------------------------

# Load static info and filter for AMER region
static_df = pd.read_excel(base_url + "Static.xlsx")
amer_df = static_df[static_df["Region"] == "AMER"]
amer_isins = set(amer_df["ISIN"].unique())

# Load price data and transform
raw_prices = pd.read_excel(base_url + "DS_RI_T_USD_M.xlsx")
prices = raw_prices.set_index("ISIN").transpose()
prices.index = pd.to_datetime(prices.index, format="%Y-%m-%d", errors="coerce")

# Load market cap data and transform
raw_mkt_caps = pd.read_excel(base_url + "DS_MV_T_USD_M.xlsx")
mkt_caps = raw_mkt_caps.set_index("ISIN").transpose()
mkt_caps.index = pd.to_datetime(mkt_caps.index, format="%Y-%m-%d", errors="coerce")

# Align data to common ISINs
common_isins = amer_isins & set(prices.columns) & set(mkt_caps.columns)
prices = prices[list(common_isins)]
mkt_caps = mkt_caps[list(common_isins)]

# Clean prices and market caps
prices = prices.replace(0, np.nan).apply(pd.to_numeric, errors="coerce").interpolate(method="linear", axis=0, limit_direction="both")
mkt_caps = mkt_caps.apply(pd.to_numeric, errors="coerce").interpolate(method="linear", axis=0, limit_direction="both")

# Calculate simple returns
simple_returns = prices.pct_change()
first_available = simple_returns.notna().apply(lambda x: x[x].index.min())

# ------------------------------
# **STEP 2 – MVP WEIGHT FUNCTION**
# ------------------------------

def compute_mvp_weights(returns_window, method='lw', max_weight=0.05, prev_weights=None, turnover_limit=None, transaction_cost=None):
    """Compute Minimum Variance Portfolio weights using specified method."""
    # Filter for sufficient data (at least 120 observations)
    sufficient_data = returns_window.count() >= 120
    returns_window = returns_window.loc[:, sufficient_data].dropna(axis=1, how="any")
    
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
    
    elif method == 'factor':
        X = returns_window.values - returns_window.values.mean(axis=0)
        pca = PCA(n_components=min(5, n))
        factors = pca.fit_transform(X)
        loadings = pca.components_.T
        F = np.cov(factors, rowvar=False)
        sigma_specific = np.var(X - factors @ loadings.T, axis=0)
        D = np.diag(sigma_specific)
        cov_matrix = loadings @ F @ loadings.T + D
    
    else:
        raise ValueError("Method must be 'lw', 'pinv', or 'factor'.")
    
    # Optimization
    w = cp.Variable(n)
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
methods = ['lw', 'pinv', 'factor']
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
        
        weights = compute_mvp_weights(returns_window, method=method, max_weight=0.05, 
                                    prev_weights=prev_weights, turnover_limit=0.3)
        mvp_weights[year] = weights.dropna()
        prev_weights = weights
    
    mvp_weights_all[method] = mvp_weights

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

# Load and prepare risk-free rate
rf_df = pd.read_excel("Risk_Free_Rate.xlsx")
rf_df.columns = ["DateRaw", "RF"]
rf_df["Date"] = pd.to_datetime(rf_df["DateRaw"].astype(str), format="%Y%m")
rf_df.set_index("Date", inplace=True)
rf_series = rf_df["RF"] / 100
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
    
    ann_avg_ret = (1 + r).prod()**(12 / n_months) - 1
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

# Print metrics
for name, (ann_avg_ret, ann_vol, cum_ret, sharpe, rmin, rmax, max_dd) in metrics.items():
    print(f"\n{name} Portfolio:")
    print(f"Annualized Average Return: {ann_avg_ret:.4f}")
    print(f"Annualized Volatility: {ann_vol:.4f}")
    print(f"Cumulative Total Return: {cum_ret:.4f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Minimum Monthly Return: {rmin:.4f}")
    print(f"Maximum Monthly Return: {rmax:.4f}")
    print(f"Maximum Drawdown: {max_dd:.4f}")

# ------------------------------
# **STEP 6 – PLOT CUMULATIVE RETURNS**
# ------------------------------

plt.figure(figsize=(12, 6))
for method, series in mvp_series_all.items():
    (1 + series).cumprod().plot(label=f"MVP ({method})")
(1 + vw_series).cumprod().plot(label="VW Portfolio")
plt.title("Cumulative Returns (2014–2024)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()