"""
Scenario analysis module for mortgage vs cash purchase Monte Carlo simulations.
Provides functionality to run and analyze different property investment scenarios.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mortgage_mc import get_base_config, run_monte_carlo_simulation
from mc_analysis import process_mc_results, display_summary_stats, display_probability_analysis, plot_mc_distributions

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
        pd.DataFrame: Processed DataFrame with Net Gain and ROI results, or None if failed
        Contains columns:
            - Cash Net Gain: Net gains for cash purchase strategy
            - Mortgage Net Gain: Net gains for mortgage strategy
            - Cash ROI (%): Annualized ROI for cash strategy
            - Mortgage ROI (%): Annualized ROI for mortgage strategy

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

def plot_scenario_comparison(scenarios_results: dict, figsize=(15, 10)):
    """
    Creates box plots comparing Net Gains and ROIs across scenarios.

    Args:
        scenarios_results (dict): Dictionary mapping scenario names to their result DataFrames
        figsize (tuple): Figure size (width, height) in inches
    """
    if not scenarios_results:
        print("No scenarios to plot.")
        return

    # Prepare data for plotting
    plot_data = []
    for scenario_name, df in scenarios_results.items():
        if df is not None and not df.empty:
            # Add Net Gain data
            for strategy in ['Cash', 'Mortgage']:
                plot_data.extend([{
                    'Scenario': scenario_name,
                    'Strategy': strategy,
                    'Metric': 'Net Gain (£)',
                    'Value': value
                } for value in df[f'{strategy} Net Gain']])
                
                # Add ROI data
                plot_data.extend([{
                    'Scenario': scenario_name,
                    'Strategy': strategy,
                    'Metric': 'ROI (%)',
                    'Value': value
                } for value in df[f'{strategy} ROI (%)']])

    if not plot_data:
        print("No valid data to plot.")
        return

    # Create DataFrame for plotting
    df_plot = pd.DataFrame(plot_data)

    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure and subplots with adjusted height ratios and spacing
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Add main title with adjusted position
    fig.suptitle('Scenario Comparison: Cash vs Mortgage', 
                fontsize=14, 
                y=0.98)  # Move title up

    # Plot Net Gains
    sns.boxplot(data=df_plot[df_plot['Metric'] == 'Net Gain (£)'],
                x='Scenario', y='Value', hue='Strategy',
                ax=ax1, palette=['skyblue', 'lightcoral'])
    ax1.set_title('Distribution of Net Gains by Scenario', pad=20)  # Add padding below subplot title
    ax1.set_ylabel('Net Gain (£)')
    # Format y-axis ticks as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
    ax1.tick_params(axis='x', rotation=45)

    # Plot ROIs
    sns.boxplot(data=df_plot[df_plot['Metric'] == 'ROI (%)'],
                x='Scenario', y='Value', hue='Strategy',
                ax=ax2, palette=['skyblue', 'lightcoral'])
    ax2.set_title('Distribution of Annualized ROI by Scenario', pad=20)  # Add padding below subplot title
    ax2.set_ylabel('Annualized ROI (%)')
    ax2.tick_params(axis='x', rotation=45)

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Adjust the layout again to make room for the suptitle
    fig.subplots_adjust(top=0.93)
    
    plt.show()

def compare_scenarios(scenarios_results: dict, show_plots: bool = True):
    """
    Compares results across multiple scenarios.

    Args:
        scenarios_results (dict): Dictionary mapping scenario names to their result DataFrames
        show_plots (bool): Whether to display comparison plots
    """
    if not scenarios_results:
        print("No scenarios to compare.")
        return

    print("\n=== Scenario Comparison ===")
    
    comparison_data = []
    for scenario_name, df in scenarios_results.items():
        if df is not None and not df.empty:
            # Calculate key metrics for this scenario
            mort_mean = df['Mortgage Net Gain'].mean()
            cash_mean = df['Cash Net Gain'].mean()
            mort_roi_mean = df['Mortgage ROI (%)'].mean()
            cash_roi_mean = df['Cash ROI (%)'].mean()
            
            # Calculate 95% confidence intervals
            mort_ci = df['Mortgage Net Gain'].std() * 1.96 / np.sqrt(len(df))
            cash_ci = df['Cash Net Gain'].std() * 1.96 / np.sqrt(len(df))
            mort_roi_ci = df['Mortgage ROI (%)'].std() * 1.96 / np.sqrt(len(df))
            cash_roi_ci = df['Cash ROI (%)'].std() * 1.96 / np.sqrt(len(df))
            
            comparison_data.append({
                'Scenario': scenario_name,
                'Mortgage Net Gain (Mean)': mort_mean,
                'Mortgage Net Gain (95% CI)': mort_ci,
                'Cash Net Gain (Mean)': cash_mean,
                'Cash Net Gain (95% CI)': cash_ci,
                'Mortgage ROI (Mean)': mort_roi_mean,
                'Mortgage ROI (95% CI)': mort_roi_ci,
                'Cash ROI (Mean)': cash_roi_mean,
                'Cash ROI (95% CI)': cash_roi_ci,
                'Net Gain Difference': mort_mean - cash_mean,
                'ROI Difference': mort_roi_mean - cash_roi_mean
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.set_index('Scenario')
        
        # Format and display the comparison
        pd.options.display.float_format = '{:,.2f}'.format
        print("\nScenario Comparison Summary:")
        print(df_comparison)
        pd.reset_option('display.float_format')
        
        # Show box plots if requested
        if show_plots:
            plot_scenario_comparison(scenarios_results)
    else:
        print("No valid scenario data for comparison.") 