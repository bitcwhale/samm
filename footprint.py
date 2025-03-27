import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import datetime
import os
from sklearn.covariance import LedoitWolf

# Define the data path in order to save the end result properly
 data_path = r"C:\put\your\path\here\to\save\result\in\excell\file"

base_url = "https://raw.githubusercontent.com/bitcwhale/samm/main/"

# ------------------------------
# Step 1: Load and Clean Data
# ------------------------------

# === Load data from GitHub with error handling ===
try:
    stock_prices = pd.read_excel(base_url + "DS_RI_T_USD_M.xlsx")
    company_data = pd.read_excel(base_url + "Static.xlsx")
    market_cap = pd.read_excel(base_url + "DS_MV_T_USD_M.xlsx")
    scope1 = pd.read_excel(base_url + "Scope_1.xlsx")
    scope2 = pd.read_excel(base_url + "Scope_2.xlsx")
    revenues = pd.read_excel(base_url + "DS_REV_USD_Y.xlsx")
    annual_market_cap = pd.read_excel(base_url + "DS_MV_T_USD_Y.xlsx")
except Exception as e:
    print(f"Error loading files from GitHub: {e}")
    exit()

# === Filter for US companies ===

us_companies = company_data[company_data['Country'] == 'US']['Name'].tolist()
us_stock_prices = stock_prices[stock_prices['NAME'].isin(us_companies)]
us_market_cap = market_cap[market_cap['NAME'].isin(us_companies)]
scope1_us = scope1[scope1['NAME'].isin(us_companies)]
scope2_us = scope2[scope2['NAME'].isin(us_companies)]
revenues_us = revenues[revenues['NAME'].isin(us_companies)]
annual_market_cap_us = annual_market_cap[annual_market_cap['NAME'].isin(us_companies)]

# === Identify date columns between 2003 and 2024 ===

date_columns = [col for col in us_stock_prices.columns
                if isinstance(col, datetime.datetime) and 2003 <= col.year <= 2024]

# === Clean stock prices and market cap data ===

us_stock_prices_cleaned = us_stock_prices.copy()
us_market_cap_cleaned = us_market_cap.copy()

# === Replace zeros with NaN and interpolate missing values===

us_stock_prices_cleaned[date_columns] = us_stock_prices_cleaned[date_columns].replace(0, np.nan)
us_market_cap_cleaned[date_columns] = us_market_cap_cleaned[date_columns].replace(0, np.nan)
us_stock_prices_cleaned[date_columns] = us_stock_prices_cleaned[date_columns].interpolate(
    method='linear', axis=1, limit_direction='both'
)
us_market_cap_cleaned[date_columns] = us_market_cap_cleaned[date_columns].interpolate(
    method='linear', axis=1, limit_direction='both'
)

# === Calculate monthly log returns ===

monthly_log_returns = np.log(us_stock_prices_cleaned[date_columns] / 
                             us_stock_prices_cleaned[date_columns].shift(1, axis=1))
monthly_log_returns['Company_Name'] = us_stock_prices_cleaned['NAME']
monthly_log_returns = monthly_log_returns[['Company_Name'] + date_columns]

# === Replace NaN returns with row-wise averages ===

return_cols = [col for col in monthly_log_returns.columns if col != 'Company_Name']
row_means = monthly_log_returns[return_cols].apply(
    lambda row: row.dropna().mean() if len(row.dropna()) > 0 else 0, axis=1
)
for idx in monthly_log_returns.index:
    monthly_log_returns.loc[idx, return_cols] = monthly_log_returns.loc[idx, return_cols].fillna(row_means[idx])

# === Set index for emissions, revenues, and annual market cap ===

scope1_us.set_index('NAME', inplace=True)
scope2_us.set_index('NAME', inplace=True)
revenues_us.set_index('NAME', inplace=True)
annual_market_cap_us.set_index('NAME', inplace=True)

# === Define function to fill NaNs with row mean ===

def fill_with_row_mean(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    row_mean = df.mean(axis=1)
    df_filled = df.apply(lambda row: row.fillna(row_mean[row.name]), axis=1)
    return df_filled.fillna(0)

# === Apply fill_with_row_mean to emissions, revenues, and annual market cap ===

scope1_us = fill_with_row_mean(scope1_us)
scope2_us = fill_with_row_mean(scope2_us)
revenues_us = fill_with_row_mean(revenues_us)
annual_market_cap_us = fill_with_row_mean(annual_market_cap_us)

# === Compute total emissions (Scope 1 + Scope 2) ===
E_df = scope1_us + scope2_us

# === Compute carbon intensity (CI_{i,Y} = E_{i,Y} / Rev_{i,Y}) ===

CI_df = E_df / revenues_us
CI_df = CI_df.replace([np.inf, -np.inf], 0).fillna(0)

# === Ensure columns are strings for consistency ===

E_df.columns = E_df.columns.astype(str)
revenues_us.columns = revenues_us.columns.astype(str)
annual_market_cap_us.columns = annual_market_cap_us.columns.astype(str)
CI_df.columns = CI_df.columns.astype(str)

# ------------------------------
# Step 2: Prepare Data for Optimization
# ------------------------------

monthly_log_returns.set_index('Company_Name', inplace=True)
returns_transposed = monthly_log_returns.T
returns_transposed.index = pd.to_datetime(returns_transposed.index)

us_market_cap_cleaned.set_index('NAME', inplace=True)
market_caps_transposed = us_market_cap_cleaned[date_columns].T
market_caps_transposed.index = pd.to_datetime(market_caps_transposed.index)

# === Align data by common dates and companies ===

common_dates = returns_transposed.index.intersection(market_caps_transposed.index)
common_companies = returns_transposed.columns.intersection(market_caps_transposed.columns)
returns_transposed = returns_transposed.loc[common_dates, common_companies]
market_caps_transposed = market_caps_transposed.loc[common_dates, common_companies]

# ------------------------------
# Step 3: Define Portfolio Functions (Optimized)
# ------------------------------

def compute_mvp_weights(returns_window, min_obs=60):
    """Compute minimum variance portfolio weights with pairwise covariance and Ledoit-Wolf shrinkage."""
    cov_matrix_pairwise = returns_window.cov(min_periods=min_obs)
    valid_assets = cov_matrix_pairwise.columns[cov_matrix_pairwise.notna().all(axis=0)]
    if len(valid_assets) < 2:
        return None
    returns_valid = returns_window[valid_assets]
    lw = LedoitWolf()
    lw.fit(returns_valid)
    cov_matrix_shrunk = lw.covariance_
    n_assets = len(valid_assets)
    
    def portfolio_variance(weights):
        return weights.T @ cov_matrix_shrunk @ weights
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, None)] * n_assets
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=valid_assets) if result.success else None

def compute_carbon_reduced_mvp_weights(returns_window, cf_target, E_Y, Cap_Y, min_obs=60):
    """Compute MVP weights with carbon footprint constraint using pairwise covariance and Ledoit-Wolf shrinkage."""
    cov_matrix_pairwise = returns_window.cov(min_periods=min_obs)
    valid_assets = cov_matrix_pairwise.columns[cov_matrix_pairwise.notna().all(axis=0)]
    if len(valid_assets) < 2:
        return None
    returns_valid = returns_window[valid_assets]
    E_Y_valid = E_Y[valid_assets]
    Cap_Y_valid = Cap_Y[valid_assets]
    
    lw = LedoitWolf()
    lw.fit(returns_valid)
    cov_matrix_shrunk = lw.covariance_
    n_assets = len(valid_assets)
    
    def portfolio_variance(weights):
        return weights.T @ cov_matrix_shrunk @ weights
    
    def carbon_constraint(weights):
        cf_p = np.sum(weights * E_Y_valid / Cap_Y_valid)
        return cf_target - cf_p
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': carbon_constraint}
    ]
    bounds = [(0, None)] * n_assets
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=valid_assets) if result.success else None

def compute_tracking_error_weights(returns_window, vw_weights, cf_target, E_Y, Cap_Y, min_obs=60):
    """Minimize tracking error with carbon footprint constraint using pairwise covariance and Ledoit-Wolf shrinkage."""
    cov_matrix_pairwise = returns_window.cov(min_periods=min_obs)
    valid_assets = cov_matrix_pairwise.columns[cov_matrix_pairwise.notna().all(axis=0)]
    if len(valid_assets) < 2:
        return None
    returns_valid = returns_window[valid_assets]
    E_Y_valid = E_Y[valid_assets]
    Cap_Y_valid = Cap_Y[valid_assets]
    vw_weights_valid = vw_weights[valid_assets]
    
    lw = LedoitWolf()
    lw.fit(returns_valid)
    cov_matrix_shrunk = lw.covariance_
    n_assets = len(valid_assets)
    
    def tracking_error(weights):
        diff = weights - vw_weights_valid
        return diff.T @ cov_matrix_shrunk @ diff
    
    def carbon_constraint(weights):
        cf_p = np.sum(weights * E_Y_valid / Cap_Y_valid)
        return cf_target - cf_p
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': carbon_constraint}
    ]
    bounds = [(0, None)] * n_assets
    initial_weights = vw_weights_valid.copy()
    
    result = minimize(tracking_error, initial_weights, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=valid_assets) if result.success else None

def adjust_weights(weights, monthly_returns):
    """Adjust weights monthly based on returns and normalize."""
    new_weights = weights * (1 + monthly_returns)
    return new_weights / new_weights.sum()

# ------------------------------
# Step 4: Portfolio Optimization and Returns
# ------------------------------

portfolio_returns = []  # MVP
portfolio_returns_50 = []  # MVP with 50% CF reduction
portfolio_returns_te = []  # Tracking error minimized with 50% CF reduction
value_weighted_returns = []
carbon_footprints = {'MVP': [], 'MVP_50': [], 'TE_50': [], 'VWP': []}
actual_dates = []

# === Compute 2013 baseline for carbon footprint ===

Y_2013 = '2013'
if Y_2013 in E_df.columns and Y_2013 in annual_market_cap_us.columns:
    E_2013 = E_df[Y_2013]
    Cap_2013 = annual_market_cap_us[Y_2013]
    common_firms_2013 = E_2013.index.intersection(Cap_2013.index)
    vw_weights_2013 = (Cap_2013[common_firms_2013] / Cap_2013[common_firms_2013].sum())
    CF_2013_vw = np.dot(vw_weights_2013, E_2013[common_firms_2013] / Cap_2013[common_firms_2013])
else:
    print("Error: Unable to compute 2013 baseline for carbon footprint. Exiting.")
    exit()

for year in range(2013, 2023):
    start_date = pd.Timestamp(f'{year-9}-01-01')
    end_date = pd.Timestamp(f'{year}-12-31')
    window_log_returns = returns_transposed.loc[start_date:end_date].dropna(axis=1, how='any')
    
    if window_log_returns.empty:
        print(f"Skipping year {year} due to no data")
        continue
    
    Y_str = str(year)
    if Y_str not in E_df.columns or Y_str not in annual_market_cap_us.columns:
        print(f"Skipping year {year} due to missing data")
        continue
    E_Y = E_df[Y_str]
    Cap_Y = annual_market_cap_us[Y_str]
    
    valid_firms = Cap_Y > 0
    valid_firms_aligned = valid_firms.reindex(window_log_returns.columns, fill_value=False)
    included_firms = window_log_returns.columns[valid_firms_aligned]
    
    if len(included_firms) == 0:
        print(f"No common firms with complete data for year {year}")
        continue
    
    window_log_returns_selected = window_log_returns[included_firms]
    E_Y_selected = E_Y[included_firms]
    Cap_Y_selected = Cap_Y[included_firms]
    
# === Compute MVP weights ===

    weights_mvp = compute_mvp_weights(window_log_returns_selected)
    if weights_mvp is None:
        print(f"MVP optimization failed for year {year}")
        continue
    CF_mvp = np.dot(weights_mvp, E_Y_selected / Cap_Y_selected)
    
# === Compute MVP with 50% CF reduction ===

    cf_target_50 = 0.5 * CF_mvp
    weights_mvp_50 = compute_carbon_reduced_mvp_weights(window_log_returns_selected, cf_target_50, E_Y_selected, Cap_Y_selected)
    if weights_mvp_50 is None:
        print(f"MVP 50% CF optimization failed for year {year}")
        continue
    CF_mvp_50 = np.dot(weights_mvp_50, E_Y_selected / Cap_Y_selected)
    
# === Compute value-weighted weights ===

    vw_weights = Cap_Y_selected / Cap_Y_selected.sum()
    CF_vw = np.dot(vw_weights, E_Y_selected / Cap_Y_selected)
    
    # Compute tracking error minimized portfolio with 50% CF reduction
    cf_target_te = 0.5 * CF_vw
    weights_te = compute_tracking_error_weights(window_log_returns_selected, vw_weights, cf_target_te, E_Y_selected, Cap_Y_selected)
    if weights_te is None:
        print(f"Tracking error optimization failed for year {year}")
        continue
    CF_te = np.dot(weights_te, E_Y_selected / Cap_Y_selected)
    
# === Store carbon footprints ===

    carbon_footprints['MVP'].append(CF_mvp)
    carbon_footprints['MVP_50'].append(CF_mvp_50)
    carbon_footprints['TE_50'].append(CF_te)
    carbon_footprints['VWP'].append(CF_vw)
    
# === Compute returns for the next year ===

    next_year_start = pd.Timestamp(f'{year+1}-01-01')
    next_year_end = pd.Timestamp(f'{year+1}-12-31')
    next_year_log_returns = returns_transposed.loc[next_year_start:next_year_end]
    next_year_market_caps = market_caps_transposed.loc[next_year_start:next_year_end]
    
    for date in next_year_log_returns.index:
        monthly_log_returns_date = next_year_log_returns.loc[date].dropna()
        if monthly_log_returns_date.empty:
            continue
        
        monthly_simple_returns = np.exp(monthly_log_returns_date) - 1
        common_companies = monthly_simple_returns.index.intersection(included_firms).intersection(next_year_market_caps.columns)
        monthly_simple_returns = monthly_simple_returns[common_companies]
        monthly_market_caps = next_year_market_caps.loc[date, common_companies]
        
        if len(common_companies) == 0:
            continue
        
# === MVP return ===

        weights_mvp_month = weights_mvp[common_companies]
        portfolio_return_mvp = np.sum(weights_mvp_month * monthly_simple_returns)
        portfolio_returns.append(portfolio_return_mvp)
        weights_mvp = adjust_weights(weights_mvp_month, monthly_simple_returns)
        
# === MVP 50% CF return ===

        weights_mvp_50_month = weights_mvp_50[common_companies]
        portfolio_return_mvp_50 = np.sum(weights_mvp_50_month * monthly_simple_returns)
        portfolio_returns_50.append(portfolio_return_mvp_50)
        weights_mvp_50 = adjust_weights(weights_mvp_50_month, monthly_simple_returns)
        
# === TE 50% CF return ===

        weights_te_month = weights_te[common_companies]
        portfolio_return_te = np.sum(weights_te_month * monthly_simple_returns)
        portfolio_returns_te.append(portfolio_return_te)
        weights_te = adjust_weights(weights_te_month, monthly_simple_returns)
        
# === Value-weighted return ===

        mc_weights = monthly_market_caps / monthly_market_caps.sum()
        vw_return = np.sum(mc_weights * monthly_simple_returns)
        value_weighted_returns.append(vw_return)
        actual_dates.append(date)

if not portfolio_returns:
    print("No portfolio returns calculated. Check data availability.")
else:
    # Convert to DataFrame
    portfolio_returns_df = pd.Series(portfolio_returns, index=actual_dates[:len(portfolio_returns)], name='MVP_Return')
    portfolio_returns_50_df = pd.Series(portfolio_returns_50, index=actual_dates[:len(portfolio_returns_50)], name='MVP_50_Return')
    portfolio_returns_te_df = pd.Series(portfolio_returns_te, index=actual_dates[:len(portfolio_returns_te)], name='TE_50_Return')
    value_weighted_returns_df = pd.Series(value_weighted_returns, index=actual_dates[:len(value_weighted_returns)], name='VWP_Return')
    carbon_footprints_df = pd.DataFrame(carbon_footprints, index=range(2013, 2023))

# --------------------------------
# --- Step 5 : Compute Metrics ---
# --------------------------------

    def compute_metrics(returns_series, rf_rate=0.0):
        """Calculate portfolio performance metrics."""
        if len(returns_series) == 0:
            return {
                'Annualized Average Return': 0,
                'Annualized Volatility': 0,
                'Sharpe Ratio': 0
            }
        annualized_return = ((1 + returns_series.mean()) ** 12 - 1)
        annualized_volatility = returns_series.std() * np.sqrt(12)
        sharpe_ratio = (annualized_return - rf_rate) / annualized_volatility if annualized_volatility != 0 else 0
        return {
            'Annualized Average Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio
        }

    mvp_metrics = compute_metrics(portfolio_returns_df)
    mvp_50_metrics = compute_metrics(portfolio_returns_50_df)
    te_50_metrics = compute_metrics(portfolio_returns_te_df)
    vwp_metrics = compute_metrics(value_weighted_returns_df)

# ------------------------------
# --- step 6 : Visualization ---
# ------------------------------

    cumulative_mvp = (1 + portfolio_returns_df).cumprod()
    cumulative_mvp_50 = (1 + portfolio_returns_50_df).cumprod()
    cumulative_te_50 = (1 + portfolio_returns_te_df).cumprod()
    cumulative_vwp = (1 + value_weighted_returns_df).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_mvp, label='Minimum Variance Portfolio')
    plt.plot(cumulative_mvp_50, label='MVP with 50% CF Reduction')
    plt.plot(cumulative_te_50, label='Tracking Error Minimized with 50% CF Reduction')
    plt.plot(cumulative_vwp, label='Value-Weighted Portfolio')
    plt.title('Cumulative Returns (2014-2023)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid()
    plt.show()

# ------------------------------------
# --- step 7: Save Results to Excel---
# ------------------------------------

    output_file = os.path.join(data_path, 'portfolio_resultsCO2.xlsx')
    with pd.ExcelWriter(output_file) as writer:
        monthly_returns_df = pd.DataFrame({
            'MVP_Return': portfolio_returns_df,
            'MVP_50_Return': portfolio_returns_50_df,
            'TE_50_Return': portfolio_returns_te_df,
            'VWP_Return': value_weighted_returns_df
        })
        monthly_returns_df.to_excel(writer, sheet_name='Monthly Returns')
        
        metrics_df = pd.DataFrame([mvp_metrics, mvp_50_metrics, te_50_metrics, vwp_metrics], 
                                 index=['MVP', 'MVP_50', 'TE_50', 'VWP'])
        metrics_df.to_excel(writer, sheet_name='Metrics')
        
        carbon_footprints_df.to_excel(writer, sheet_name='Carbon Footprints')
        
        CI_df.to_excel(writer, sheet_name='Carbon Intensity')

# === Print Results ===

    print("\n--- Portfolio Metrics ---")
    for portfolio, metrics in zip(['MVP', 'MVP_50', 'TE_50', 'VWP'], 
                                 [mvp_metrics, mvp_50_metrics, te_50_metrics, vwp_metrics]):
        print(f"\n{portfolio}:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

    print("\n--- Trade-Off Analysis ---")
    print("MVP vs MVP_50:")
    print(f"Sharpe Ratio Difference: {mvp_metrics['Sharpe Ratio'] - mvp_50_metrics['Sharpe Ratio']:.4f}")
    print(f"Carbon Footprint (2013) - MVP: {carbon_footprints['MVP'][0]:.4f}, MVP_50: {carbon_footprints['MVP_50'][0]:.4f}")
    print("VWP vs TE_50:")
    print(f"Sharpe Ratio Difference: {vwp_metrics['Sharpe Ratio'] - te_50_metrics['Sharpe Ratio']:.4f}")
    print(f"Carbon Footprint (2013) - VWP: {carbon_footprints['VWP'][0]:.4f}, TE_50: {carbon_footprints['TE_50'][0]:.4f}")
