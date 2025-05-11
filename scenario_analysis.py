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
        net_gain = cash_flows[-1] - cash_flows[0]
        irr = npf.irr(cash_flows) * 100 if len(cash_flows) > 1 else 0
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
    Compares results across multiple scenarios (new long-format version).

    Args:
        scenarios_results (dict): Dictionary mapping scenario names to their result DataFrames
        show_plots (bool): Whether to display comparison plots
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
                    'IRR Mean': subdf['irr'].mean(),
                    'IRR Std': subdf['irr'].std(),
                    'Total Net Worth Mean': subdf['total_net_worth'].mean(),
                    'Total Net Worth Std': subdf['total_net_worth'].std(),
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
    Display summary statistics for the simulation results.
    
    Args:
        df (pd.DataFrame): DataFrame containing simulation results
    """
    # Group by scenario type and calculate statistics
    stats = df.groupby('scenario_type').agg({
        'net_gain': ['mean', 'median', 'std'],
        'irr': ['mean', 'median', 'std'],
        'total_net_worth': ['mean', 'median', 'std']
    }).round(2)
    
    print("\nSummary Statistics:")
    print(stats)
    
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
        
        print(f"\n{scenario}:")
        print(f"  Net Gain > 0: {net_gain_positive:.1f}% of scenarios")
        print(f"  IRR > 0: {irr_positive:.1f}% of scenarios")
        print(f"  Net Worth > Initial Investment: {net_worth_positive:.1f}% of scenarios")

def plot_mc_distributions(df: pd.DataFrame, num_simulations: int, config: dict, dist_assumptions: dict):
    """
    Plot distributions of Net Gains, IRRs, and Total Net Worth from Monte Carlo simulations.
    
    Args:
        df (pd.DataFrame): DataFrame containing simulation results
        num_simulations (int): Number of simulations run
        config (dict): Configuration used for the simulation
        dist_assumptions (dict): Distribution assumptions used
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

    for metric, title, ax, fmt in metrics:
        for scenario in df['scenario_type'].unique():
            scenario_data = df[df['scenario_type'] == scenario][metric]
            sns.histplot(scenario_data, label=scenario, alpha=0.5, ax=ax, kde=True)
        ax.set_title(f'Distribution of {title}', fontsize=12, pad=15)
        ax.set_xlabel(title, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
        if '£' in title:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
        else:
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

    fig.suptitle('Buy vs Rent Monte Carlo Simulation Results', fontsize=14, fontweight='bold', y=1.02)
    info_text = (
        f"Number of simulations: {num_simulations:,}\n"
        f"Property Value: £{config['property_value_initial']:,.0f}\n"
        f"Holding Period: {config['holding_period_years']} years"
    )
    if 'initial_annual_rent' in config:
        info_text += f"\nInitial Annual Rent: £{config['initial_annual_rent']:,.0f}"
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
        "total_net_worth": ["mean", "median", "std", "min", "max"]  # Add total net worth stats
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
    fig = plot_mc_distributions(df, holding_period_years)
    plt.show()