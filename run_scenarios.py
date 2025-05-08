#!/usr/bin/env python3
"""
run_scenarios.py

A script to run different Monte Carlo simulation scenarios for mortgage vs. cash purchase analysis.
Provides a clean interface for testing different configurations and visualizing results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from mortgage_mc import get_base_config, run_monte_carlo_simulation
from mc_analysis import process_mc_results, display_summary_stats, display_probability_analysis, plot_mc_distributions

def create_scenario_config(
    holding_period_years: int = 15,
    fixed_term_length_years: int = 5,
    initial_rate: float = 0.0445,
    property_value: float = None,
    deposit_percentage: float = None,
    base_config: dict = None
) -> dict:
    """
    Creates a configuration dictionary for a scenario by updating the base config
    with the provided parameters.

    Args:
        holding_period_years (int): Length of the investment period
        fixed_term_length_years (int): Length of each fixed rate term
        initial_rate (float): Initial mortgage interest rate (as decimal)
        property_value (float, optional): Initial property value
        deposit_percentage (float, optional): Deposit percentage (as decimal)
        base_config (dict, optional): Base configuration to use (if None, gets fresh copy)

    Returns:
        dict: Complete configuration dictionary for the scenario
    """
    if base_config is None:
        base_config = get_base_config()

    updates = {
        "holding_period_years": holding_period_years,
        "fixed_term_length_years": fixed_term_length_years,
        "initial_mortgage_interest_rate_annual": initial_rate,
    }

    if property_value is not None:
        updates["property_value_initial"] = property_value
    if deposit_percentage is not None:
        updates["deposit_percentage"] = deposit_percentage

    return {**base_config, **updates}

def create_rate_assumptions(
    prop_app_mean: float = 0.03,
    prop_app_std: float = 0.05,
    alt_inv_mean: float = 0.05,
    alt_inv_std: float = 0.08,
    sc_inf_mean: float = 0.035,
    sc_inf_std: float = 0.025,
    future_rate_mean: float = 0.05,
    future_rate_std: float = 0.015
) -> dict:
    """
    Creates a dictionary of distribution assumptions for the Monte Carlo simulation.

    Args:
        prop_app_mean (float): Mean property appreciation rate
        prop_app_std (float): Standard deviation of property appreciation
        alt_inv_mean (float): Mean alternative investment return rate
        alt_inv_std (float): Standard deviation of alternative investment returns
        sc_inf_mean (float): Mean service charge inflation rate
        sc_inf_std (float): Standard deviation of service charge inflation
        future_rate_mean (float): Mean future fixed rate
        future_rate_std (float): Standard deviation of future fixed rates

    Returns:
        dict: Distribution assumptions dictionary
    """
    return {
        "prop_app_mean": prop_app_mean,
        "prop_app_std_dev": prop_app_std,
        "alt_inv_mean": alt_inv_mean,
        "alt_inv_std_dev": alt_inv_std,
        "sc_inf_mean": sc_inf_mean,
        "sc_inf_std_dev": sc_inf_std,
        "remort_rate_mean": future_rate_mean,
        "remort_rate_std_dev": future_rate_std
    }

def run_scenario(
    config: dict,
    distribution_assumptions: dict,
    num_simulations: int = 10000,
    show_plots: bool = True
) -> pd.DataFrame:
    """
    Runs a complete Monte Carlo simulation scenario and displays results.

    Args:
        config (dict): Configuration dictionary
        distribution_assumptions (dict): Distribution assumptions dictionary
        num_simulations (int): Number of Monte Carlo simulations to run
        show_plots (bool): Whether to display plots

    Returns:
        pd.DataFrame: Processed results DataFrame
    """
    # Calculate and display scenario structure
    num_remortgages = int(np.floor((config["holding_period_years"] - 1) / config["fixed_term_length_years"]))
    
    print("\n=== Scenario Configuration ===")
    print(f"Holding Period:        {config['holding_period_years']} years")
    print(f"Fixed Term Length:     {config['fixed_term_length_years']} years")
    print(f"Number of Remortgages: {num_remortgages}")
    print(f"Initial Fixed Rate:    {config['initial_mortgage_interest_rate_annual']:.2%}")
    print(f"Property Value:        £{config['property_value_initial']:,.0f}")
    print(f"Deposit Percentage:    {config['deposit_percentage']:.0%}")
    
    print("\n=== Rate Assumptions ===")
    print(f"Property Appreciation:  {distribution_assumptions['prop_app_mean']:.1%} ± {distribution_assumptions['prop_app_std_dev']:.1%}")
    print(f"Alternative Investment: {distribution_assumptions['alt_inv_mean']:.1%} ± {distribution_assumptions['alt_inv_std_dev']:.1%}")
    print(f"Service Charge Infl.:   {distribution_assumptions['sc_inf_mean']:.1%} ± {distribution_assumptions['sc_inf_std_dev']:.1%}")
    print(f"Future Fixed Rates:     {distribution_assumptions['remort_rate_mean']:.1%} ± {distribution_assumptions['remort_rate_std_dev']:.1%}")

    # Run simulation
    print(f"\n=== Running {num_simulations:,} Simulations ===")
    start_time = time.time()
    
    cash_results, mortgage_results = run_monte_carlo_simulation(
        base_config=config,
        num_simulations=num_simulations,
        **distribution_assumptions
    )
    
    duration = time.time() - start_time
    print(f"Simulation completed in {duration:.2f} seconds")

    # Process results
    df_results = process_mc_results(cash_results, mortgage_results, config['holding_period_years'])
    
    # Display results
    display_summary_stats(df_results, config)
    display_probability_analysis(df_results)
    
    if show_plots:
        plot_mc_distributions(df_results, num_simulations, config, distribution_assumptions)
    
    return df_results

def main():
    """Example usage of the scenario runner."""
    # Example 1: Standard 5-year fixes over long term
    print("\n=== Scenario 1: Long-term with 5-year fixes ===")
    config_1 = create_scenario_config(
        holding_period_years=25,
        fixed_term_length_years=5,
        initial_rate=0.0445
    )
    assumptions_1 = create_rate_assumptions()  # Use defaults
    results_1 = run_scenario(config_1, assumptions_1)

    # Example 2: Short-term with 2-year fixes
    print("\n=== Scenario 2: Short-term with 2-year fixes ===")
    config_2 = create_scenario_config(
        holding_period_years=7,
        fixed_term_length_years=2,
        initial_rate=0.0425,
        property_value=650000,
        deposit_percentage=0.25
    )
    assumptions_2 = create_rate_assumptions(
        future_rate_mean=0.055,  # Expecting higher future rates
        future_rate_std=0.02     # More uncertainty
    )
    results_2 = run_scenario(config_2, assumptions_2)

if __name__ == "__main__":
    main() 