"""
Scenario analysis module for mortgage vs cash purchase Monte Carlo simulations.
Provides functionality to run and analyze different property investment scenarios.

NOTE ON ROI:
-----------------
We do NOT report or plot ROI (Return on Investment) for these scenarios. This is because, for property purchase scenarios (especially Buy_Mortgage), the initial investment is constant across all simulations, while the net gain has limited variability. This results in an ROI distribution that is nearly a vertical line, which is not informative or meaningful in a Monte Carlo context. For rent scenarios, ROI can be misleadingly high if the denominator is not the total cash available. Instead, we focus on Net Gain, IRR, and Total Net Worth, which provide a more realistic and comparable assessment of outcomes.
-----------------
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy_financial as npf
from scipy.stats import ttest_ind
from mortgage_mc import get_base_config, run_monte_carlo_simulation
from mc_analysis import process_mc_results, display_summary_stats, display_probability_analysis, plot_mc_distributions
from investment_options_mc import get_base_config_buy_vs_rent, run_buy_vs_rent_mc_simulation

def process_buy_vs_rent_mc_results(mc_results: list) -> pd.DataFrame:
    """
    Process Monte Carlo simulation results for buy vs rent comparison.
    
    Args:
        mc_results (list): List of dictionaries containing simulation results
        
    Returns:
        pd.DataFrame: Processed results with Net Gain, IRR, and Total Net Worth calculations
    """
    processed_results = []
    
    # Process results
    for res in mc_results:
        cash_flows = res['cash_flows']
        
        # Calculate net gain
        net_gain = cash_flows[-1] - cash_flows[0]
        
        # Calculate IRR with proper handling of cash flow timing
        try:
            # Convert cash flows to numpy array for better handling
            cf_array = np.array(cash_flows)
            # Calculate IRR only if we have valid cash flows
            if len(cf_array) > 1 and np.any(cf_array > 0) and np.any(cf_array < 0):
                irr = npf.irr(cf_array) * 100
            else:
                irr = 0
        except Exception:
            irr = 0
            
        total_net_worth = res['total_net_worth']
        
        processed_results.append({
            'scenario_type': res['scenario_type'],
            'net_gain': net_gain,
            'irr': irr,
            'total_net_worth': total_net_worth,
            'initial_investment': abs(cash_flows[0])
        })
    
    df = pd.DataFrame(processed_results)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def run_and_analyze_scenario(
    scenario_name: str, 
    config_overrides: dict, 
    dist_assumptions: dict, 
    num_simulations: int = 10000,
    show_plots: bool = True
) -> pd.DataFrame:
    """
    Runs Monte Carlo simulation for a scenario, displays analysis, and optionally plots distributions.

    Args:
        scenario_name (str): Name for the scenario (used in prints/titles)
        config_overrides (dict): Dictionary of base_config values to override
        dist_assumptions (dict): Dictionary of distribution parameters containing:
            - prop_app_mean (float): Mean property appreciation rate
            - prop_app_std_dev (float): Standard deviation of property appreciation
            - alt_inv_mean (float): Mean alternative investment return rate
            - alt_inv_std_dev (float): Standard deviation of alternative investment returns
            - sc_inf_mean (float): Mean service charge inflation rate
            - sc_inf_std_dev (float): Standard deviation of service charge inflation
            - remort_rate_mean (float): Mean remortgage rate
            - remort_rate_std_dev (float): Standard deviation of remortgage rates
        num_simulations (int): Number of Monte Carlo iterations
        show_plots (bool): Whether to display distribution plots

    Returns:
        pd.DataFrame: Processed DataFrame with Net Gain and IRR results, or None if failed
        Contains columns:
            - Cash Net Gain: Net gains for cash purchase strategy
            - Mortgage Net Gain: Net gains for mortgage strategy
            - Cash IRR (%): IRR for cash strategy
            - Mortgage IRR (%): IRR for mortgage strategy

    Example:
        >>> config_overrides = {
        ...     "property_value_initial": 550000,
        ...     "holding_period_years": 15,
        ...     "deposit_percentage": 0.2
        ... }
        >>> dist_assumptions = {
        ...     "prop_app_mean": 0.03,
        ...     "prop_app_std_dev": 0.05,
        ...     "alt_inv_mean": 0.05,
        ...     "alt_inv_std_dev": 0.08,
        ...     "sc_inf_mean": 0.035,
        ...     "sc_inf_std_dev": 0.025,
        ...     "remort_rate_mean": 0.048,
        ...     "remort_rate_std_dev": 0.015
        ... }
        >>> df = run_and_analyze_scenario("Base Case", config_overrides, dist_assumptions)
    """
    print(f"\n{'='*20} SCENARIO: {scenario_name.upper()} {'='*20}")

    # Create config for this scenario
    scenario_config = get_base_config()
    scenario_config.update(config_overrides)

    # Print configuration
    print("\n--- Running with Configuration ---")
    print(f"  Number of simulations: {num_simulations:,}")
    print(f"  Property Value Initial: £{scenario_config['property_value_initial']:,.0f}")
    print(f"  Holding Period:         {scenario_config['holding_period_years']} years")
    print(f"  Fixed Term Length:      {scenario_config.get('fixed_term_length_years', 5)} years")
    print(f"  Initial Mortgage Rate:  {scenario_config['initial_mortgage_interest_rate_annual']:.2%}")
    print(f"  Deposit Percentage:     {scenario_config['deposit_percentage']:.0%}")
    print(f"  Initial Service Charge: £{scenario_config['service_charge_annual_initial']:,.0f} p.a.")
    print(f"  Remortgage Fee:         £{scenario_config.get('remortgage_fee', 0):,.0f}")

    # Print distribution assumptions
    print("\nDistribution Assumptions (Mean / Std Dev):")
    print(f"  Property Apprec.:       {dist_assumptions['prop_app_mean']:.1%} / {dist_assumptions['prop_app_std_dev']:.1%}")
    print(f"  Alt. Investment Ret.:    {dist_assumptions['alt_inv_mean']:.1%} / {dist_assumptions['alt_inv_std_dev']:.1%}")
    print(f"  Service Chg Infl.:      {dist_assumptions['sc_inf_mean']:.1%} / {dist_assumptions['sc_inf_std_dev']:.1%}")
    print(f"  Remortgage Rate:       {dist_assumptions['remort_rate_mean']:.1%} / {dist_assumptions['remort_rate_std_dev']:.1%}")

    # Run simulation
    print(f"\n--- Running {num_simulations:,} Simulations ---")
    start_time = time.time()
    try:
        cash_results, mortgage_results = run_monte_carlo_simulation(
            base_config=scenario_config,
            num_simulations=num_simulations,
            **dist_assumptions
        )
        end_time = time.time()
        print(f"Duration: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        return None

    # Process results
    print("\n--- Processing Results ---")
    try:
        df_processed = process_mc_results(cash_results, mortgage_results, scenario_config['holding_period_years'])
    except Exception as e:
        print(f"Error processing results: {str(e)}")
        return None

    # Display analysis
    if not df_processed.empty:
        print(f"\n--- Analysis for: {scenario_name} ---")
        display_summary_stats(df_processed)
        display_probability_analysis(df_processed)
        
        # Optional plotting
        if show_plots:
            plot_mc_distributions(df_processed, num_simulations, scenario_config, dist_assumptions)
    else:
        print("Error: No results processed for this scenario.")
        return None

    print(f"{'='*20} END SCENARIO: {scenario_name.upper()} {'='*20}\n")
    return df_processed

def compare_scenarios(scenarios_results: dict, show_plots: bool = True):
    """
    Compares results across multiple scenarios (new long-format version), with CIs.
    """
    if not scenarios_results:
        print("No scenarios to compare.")
        return
    print("\n=== Scenario Comparison ===")
    summary_rows = []
    for scenario_name, df in scenarios_results.items():
        if df is not None and not df.empty:
            for scen_type in df['scenario_type'].unique():
                subdf = df[df['scenario_type'] == scen_type]
                summary_rows.append({
                    'Scenario': scenario_name,
                    'Strategy': scen_type,
                    'Net Gain Mean': subdf['net_gain'].mean(),
                    'Net Gain Std': subdf['net_gain'].std(),
                    'Net Gain 95% CI': bootstrap_ci(subdf['net_gain'].values, np.mean),
                    'IRR Mean': subdf['irr'].mean(),
                    'IRR Std': subdf['irr'].std(),
                    'IRR 95% CI': bootstrap_ci(subdf['irr'].values, np.mean),
                    'Total Net Worth Mean': subdf['total_net_worth'].mean(),
                    'Total Net Worth Std': subdf['total_net_worth'].std(),
                    'Total Net Worth 95% CI': bootstrap_ci(subdf['total_net_worth'].values, np.mean),
                })
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\nScenario Comparison Summary:")
        print(summary_df.round(2).to_string(index=False))
    else:
        print("No valid scenario data for comparison.")
        return
    # Optional: plot distributions for each scenario_type across all scenarios
    if show_plots:
        plt.style.use('seaborn-v0_8-whitegrid')
        metrics = ['net_gain', 'irr', 'total_net_worth']
        metric_titles = ['Net Gain (£)', 'IRR (%)', 'Total Net Worth (£)']
        for metric, title in zip(metrics, metric_titles):
            plt.figure(figsize=(12, 6))
            plot_data = []
            for scenario_name, df in scenarios_results.items():
                if df is not None and not df.empty:
                    for scen_type in df['scenario_type'].unique():
                        vals = df[df['scenario_type'] == scen_type][metric]
                        for v in vals:
                            plot_data.append({'Scenario': scenario_name, 'Strategy': scen_type, 'Value': v})
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                sns.boxplot(data=plot_df, x='Scenario', y='Value', hue='Strategy')
                plt.title(f'Distribution of {title} by Scenario and Strategy')
                plt.ylabel(title)
                plt.xlabel('Scenario')
                plt.legend(title='Strategy')
                plt.tight_layout()
                plt.show()

def run_and_analyze_buy_vs_rent_scenario(
    scenario_name: str,
    config_overrides: dict,
    dist_assumptions: dict,
    num_simulations: int = 10000,
    show_plots: bool = True,
    cash_available: float = None,
    use_mortgage: bool = True,
) -> pd.DataFrame:
    """
    Runs Monte Carlo simulation for a buy vs rent scenario, displays analysis, and optionally plots distributions.

    Args:
        scenario_name (str): Name for the scenario (used in prints/titles)
        config_overrides (dict): Dictionary of base_config values to override
        dist_assumptions (dict): Dictionary of distribution parameters containing:
            - prop_app_mean (float): Mean property appreciation rate
            - prop_app_std_dev (float): Standard deviation of property appreciation
            - alt_inv_mean (float): Mean alternative investment return rate
            - alt_inv_std_dev (float): Standard deviation of alternative investment returns
            - sc_inf_mean (float): Mean service charge inflation rate
            - sc_inf_std_dev (float): Standard deviation of service charge inflation
            - rent_inf_mean (float): Mean rent inflation rate
            - rent_inf_std_dev (float): Standard deviation of rent inflation
            - mortgage_rate_mean (float): Mean mortgage rate
            - mortgage_rate_std_dev (float): Standard deviation of mortgage rates
        num_simulations (int): Number of Monte Carlo iterations
        show_plots (bool): Whether to display distribution plots
        cash_available (float): Amount of cash available for investment
        use_mortgage (bool): Whether to use mortgage for buying scenario

    Returns:
        pd.DataFrame: Processed DataFrame with Net Gain, IRR, and Total Net Worth results
    """
    print(f"\n{'='*20} SCENARIO: {scenario_name.upper()} {'='*20}")

    # Create config for this scenario
    scenario_config = get_base_config_buy_vs_rent()
    scenario_config.update(config_overrides)

    # Print configuration
    print("\n--- Running with Configuration ---")
    print(f"  Number of simulations: {num_simulations:,}")
    print(f"  Property Value Initial: £{scenario_config['property_value_initial']:,.0f}")
    print(f"  Initial Annual Rent:    £{scenario_config['initial_annual_rent']:,.0f}")
    print(f"  Holding Period:         {scenario_config['holding_period_years']} years")
    print(f"  Initial Service Charge: £{scenario_config['service_charge_annual_initial']:,.0f} p.a.")
    if use_mortgage:
        print(f"  Mortgage Term:         {scenario_config['mortgage_term_years']} years")
        print(f"  Loan-to-Value:         {scenario_config['loan_to_value_ratio']:.0%}")
        print(f"  Initial Mortgage Rate: {scenario_config['mortgage_interest_rate_annual']:.1%}")
        print(f"  Remortgage Interval:   {scenario_config['remortgage_interval_years']} years")
    else:
        print(f"  Purchase Type:         Cash Purchase")
    if cash_available is not None:
        print(f"  Cash Available:        £{cash_available:,.0f}")

    # Print distribution assumptions
    print("\nDistribution Assumptions (Mean / Std Dev):")
    print(f"  Property Apprec.:       {dist_assumptions['prop_app_mean']:.1%} / {dist_assumptions['prop_app_std_dev']:.1%}")
    print(f"  Alt. Investment Ret.:    {dist_assumptions['alt_inv_mean']:.1%} / {dist_assumptions['alt_inv_std_dev']:.1%}")
    print(f"  Service Chg Infl.:      {dist_assumptions['sc_inf_mean']:.1%} / {dist_assumptions['sc_inf_std_dev']:.1%}")
    print(f"  Rent Inflation:         {dist_assumptions['rent_inf_mean']:.1%} / {dist_assumptions['rent_inf_std_dev']:.1%}")
    if use_mortgage:
        print(f"  Mortgage Rate:         {dist_assumptions['mortgage_rate_mean']:.1%} / {dist_assumptions['mortgage_rate_std_dev']:.1%}")

    # Run simulation
    print(f"\n--- Running {num_simulations:,} Simulations ---")
    start_time = time.time()
    try:
        buy_results, rent_results = run_buy_vs_rent_mc_simulation(
            base_config=scenario_config,
            num_simulations=num_simulations,
            cash_available=cash_available,
            use_mortgage=use_mortgage,
            **dist_assumptions
        )
        end_time = time.time()
        print(f"Duration: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        return None

    # Process results
    print("\n--- Processing Results ---")
    try:
        # Combine buy and rent results into a single list
        all_results = buy_results + rent_results
        df_processed = process_buy_vs_rent_mc_results(all_results)
    except Exception as e:
        print(f"Error processing results: {str(e)}")
        return None

    # Display analysis
    if not df_processed.empty:
        print(f"\n--- Analysis for: {scenario_name} ---")
        display_summary_stats(df_processed)
        display_probability_analysis(df_processed)
        
        # Optional plotting
        if show_plots:
            plot_mc_distributions(df_processed, num_simulations, scenario_config, dist_assumptions)
    else:
        print("Error: No results processed for this scenario.")
        return None

    print(f"{'='*20} END SCENARIO: {scenario_name.upper()} {'='*20}\n")
    return df_processed

def display_summary_stats(df: pd.DataFrame):
    """
    Display summary statistics for the simulation results, including bootstrap CIs.
    """
    stats = df.groupby('scenario_type').agg({
        'net_gain': ['mean', 'median', 'std', 'min', 'max'],
        'irr': ['mean', 'median', 'std', 'min', 'max'],
        'total_net_worth': ['mean', 'median', 'std', 'min', 'max']
    }).round(2)
    print("\nSummary Statistics:")
    print(stats)
    # Add bootstrap CIs for each metric
    for scenario in df['scenario_type'].unique():
        print(f"\nBootstrap 95% Confidence Intervals for {scenario}:")
        for metric in ['net_gain', 'irr', 'total_net_worth']:
            vals = df[df['scenario_type'] == scenario][metric].values
            mean_ci = bootstrap_ci(vals, np.mean)
            median_ci = bootstrap_ci(vals, np.median)
            print(f"  {metric} mean:   {mean_ci[0]:,.2f} to {mean_ci[1]:,.2f}")
            print(f"  {metric} median: {median_ci[0]:,.2f} to {median_ci[1]:,.2f}")
    # Calculate probabilities using numpy broadcasting
    scenarios = df['scenario_type'].unique()
    for i, scenario1 in enumerate(scenarios):
        for j, scenario2 in enumerate(scenarios):
            if i != j:
                vals1 = df[df['scenario_type'] == scenario1]['total_net_worth'].values
                vals2 = df[df['scenario_type'] == scenario2]['total_net_worth'].values
                if len(vals1) > 0 and len(vals2) > 0:
                    prob = (vals1[:, None] > vals2).mean() * 100
                    print(f"\n{scenario1} better than {scenario2}: {prob:.1f}% of scenarios")

def display_probability_analysis(df: pd.DataFrame):
    """
    Display probability analysis for the simulation results.
    
    Args:
        df (pd.DataFrame): DataFrame containing simulation results
    """
    scenarios = df['scenario_type'].unique()
    
    print("\nProbability Analysis:")
    for scenario in scenarios:
        scenario_data = df[df['scenario_type'] == scenario]
        
        # Calculate probabilities for each metric
        net_gain_positive = (scenario_data['net_gain'] > 0).mean() * 100
        irr_positive = (scenario_data['irr'] > 0).mean() * 100
        net_worth_positive = (scenario_data['total_net_worth'] > scenario_data['initial_investment']).mean() * 100
        
        # Calculate additional metrics
        net_gain_median = scenario_data['net_gain'].median()
        irr_median = scenario_data['irr'].median()
        net_worth_median = scenario_data['total_net_worth'].median()
        
        print(f"\n{scenario}:")
        print(f"  Net Gain > 0: {net_gain_positive:.1f}% of scenarios (Median: £{net_gain_median:,.0f})")
        print(f"  IRR > 0: {irr_positive:.1f}% of scenarios (Median: {irr_median:.1f}%)")
        print(f"  Net Worth > Initial Investment: {net_worth_positive:.1f}% of scenarios (Median: £{net_worth_median:,.0f})")

def plot_mc_distributions(df: pd.DataFrame, num_simulations: int, config: dict, dist_assumptions: dict):
    """
    Plot distributions of Net Gains, IRRs, and Total Net Worth from Monte Carlo simulations.
    Annotate each plot with mean ± 95% CI, median ± 95% CI for each scenario/metric,
    and add 'probability better' for total net worth if two strategies are present.
    """
    import matplotlib.ticker as mtick
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))  # 3 subplots for each metric
    ax1, ax2, ax3 = axes

    # Define metrics and axes
    metrics = [
        ('net_gain', 'Net Gain (£)', ax1, lambda x: f'£{x:,.0f}'),
        ('irr', 'IRR (%)', ax2, lambda x: f'{x:.1f}%'),
        ('total_net_worth', 'Total Net Worth (£)', ax3, lambda x: f'£{x:,.0f}')
    ]

    scenario_types = df['scenario_type'].unique()
    scenario_colors = dict(zip(scenario_types, sns.color_palette(n_colors=len(scenario_types))))

    for metric, title, ax, fmt in metrics:
        stats_texts = []
        for scenario in scenario_types:
            scenario_data = df[df['scenario_type'] == scenario][metric]
            color = scenario_colors[scenario]
            # Plot histogram with KDE
            sns.histplot(scenario_data, label=scenario, alpha=0.5, ax=ax, kde=True, color=color)
            # Add vertical line for median
            median_val = scenario_data.median()
            ax.axvline(median_val, color=color, linestyle='--', alpha=0.7)
            # Bootstrap CIs
            mean_ci = bootstrap_ci(scenario_data, np.mean)
            median_ci = bootstrap_ci(scenario_data, np.median)
            mean_val = scenario_data.mean()
            # Annotate statistics
            stats_text = (
                f"{scenario}\n"
                f"  Mean:   {fmt(mean_val)}\n"
                f"    95% CI: {fmt(mean_ci[0])} to {fmt(mean_ci[1])}\n"
                f"  Median: {fmt(median_val)}\n"
                f"    95% CI: {fmt(median_ci[0])} to {fmt(median_ci[1])}"
            )
            stats_texts.append(stats_text)
            # Place annotation near median line
            ax.text(median_val, ax.get_ylim()[1]*0.95, f"{scenario}\nMedian: {fmt(median_val)}", color=color, rotation=90, va='top', ha='center', fontsize=9, alpha=0.7)
        # Add all stats as a legend box
        ax.set_title(f'Distribution of {title}', fontsize=12, pad=15)
        ax.set_xlabel(title, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
        if '£' in title:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
        else:
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
        # Place stats box in upper left
        ax.text(0.01, 0.99, '\n\n'.join(stats_texts), transform=ax.transAxes, fontsize=9, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # For total net worth, add probability better if two strategies
    if len(scenario_types) == 2:
        scen1, scen2 = scenario_types
        vals1 = df[df['scenario_type'] == scen1]['total_net_worth'].values
        vals2 = df[df['scenario_type'] == scen2]['total_net_worth'].values
        prob1 = (vals1[:, None] > vals2).mean() * 100
        prob2 = 100 - prob1
        prob_text = (f"Probability {scen1} better: {prob1:.1f}%\n"
                     f"Probability {scen2} better: {prob2:.1f}%")
        ax3.text(0.99, 0.99, prob_text, transform=ax3.transAxes, fontsize=11, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    fig.suptitle('Buy vs Rent Monte Carlo Simulation Results', fontsize=14, fontweight='bold', y=1.02)
    # Create detailed info text
    info_text = (
        f"Number of simulations: {num_simulations:,}\n"
        f"Property Value: £{config['property_value_initial']:,.0f}\n"
        f"Holding Period: {config['holding_period_years']} years\n"
    )
    if 'initial_annual_rent' in config:
        info_text += f"Initial Annual Rent: £{config['initial_annual_rent']:,.0f}\n"
    # Add distribution assumptions
    info_text += "\nDistribution Assumptions:\n"
    for key, value in dist_assumptions.items():
        if 'mean' in key:
            param_name = key.replace('_mean', '').replace('_', ' ').title()
            std_key = key.replace('mean', 'std_dev')
            if std_key in dist_assumptions:
                info_text += f"{param_name}: {value:.1%} ± {dist_assumptions[std_key]:.1%}\n"
    plt.figtext(0.02, 0.02, info_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def analyze_mc_results(df: pd.DataFrame, holding_period_years: int):
    """
    Analyze and display Monte Carlo simulation results.
    """
    # Calculate summary statistics
    summary_stats = df.groupby("scenario_type").agg({
        "net_gain": ["mean", "median", "std", "min", "max"],
        "irr": ["mean", "median", "std", "min", "max"],
        "total_net_worth": ["mean", "median", "std", "min", "max"]
    }).round(2)
    
    # Calculate probabilities using numpy broadcasting
    scenarios = df["scenario_type"].unique()
    n_scenarios = len(scenarios)
    probabilities = pd.DataFrame(index=scenarios, columns=scenarios)
    
    for i, scenario1 in enumerate(scenarios):
        for j, scenario2 in enumerate(scenarios):
            if i != j:
                vals1 = df[df["scenario_type"] == scenario1]["total_net_worth"].values
                vals2 = df[df["scenario_type"] == scenario2]["total_net_worth"].values
                if len(vals1) > 0 and len(vals2) > 0:
                    probabilities.loc[scenario1, scenario2] = (vals1[:, None] > vals2).mean() * 100
    
    # Display results
    print(f"\nMonte Carlo Analysis Results (Holding Period: {holding_period_years} years)")
    print("\nSummary Statistics:")
    print(summary_stats)
    
    print("\nProbability of Higher Total Net Worth:")
    print(probabilities.round(2))
    
    # Create and display plots
    plot_mc_distributions(df, len(df), {}, {})
    plt.show()

# --- Statistical Helper Functions ---
def bootstrap_ci(data, func=np.mean, n_bootstrap=1000, ci=95):
    data = np.array(data)
    bootstraps = [func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(bootstraps, (100 - ci) / 2)
    upper = np.percentile(bootstraps, 100 - (100 - ci) / 2)
    return lower, upper

def permutation_test(data1, data2, func=np.mean, n_permutations=10000):
    data1 = np.array(data1)
    data2 = np.array(data2)
    observed_diff = func(data1) - func(data2)
    combined = np.concatenate([data1, data2])
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        new_diff = func(combined[:len(data1)]) - func(combined[len(data1):])
        if abs(new_diff) >= abs(observed_diff):
            count += 1
    p_value = count / n_permutations
    return p_value