# mc_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --- Helper Functions for Metric Calculation ---

def calculate_net_gain(results_dict):
    """Calculates Net Gain from a single simulation result dictionary."""
    if not results_dict: return np.nan
    net_gain = (
        results_dict.get('final_equity_in_property', 0) +
        results_dict.get('value_of_alternative_investments_at_end', 0) -
        results_dict.get('initial_cash_outlay', 0) -
        results_dict.get('total_ongoing_property_costs_paid', 0) -
        (results_dict.get('total_mortgage_interest_paid', 0) if not results_dict.get('is_cash_purchase', True) else 0)
    )
    return net_gain

def calculate_roi(results_dict, holding_period):
    """Calculates Annualized ROI from a single simulation result dictionary."""
    if not results_dict: return np.nan
    net_gain = calculate_net_gain(results_dict)
    initial_outlay = results_dict.get('initial_cash_outlay', 0)
    if initial_outlay <= 0 or holding_period <= 0: return np.nan

    ending_value = initial_outlay + net_gain
    if ending_value <= 0: return -1.0 # Represent total loss or worse as -100%

    # Calculate CAGR
    base = ending_value / initial_outlay
    try:
        roi = (base ** (1/holding_period)) - 1
    except (ValueError, TypeError): # Handle potential complex numbers or other issues
        roi = -1.0
    return roi

# --- Main Processing Function ---

def process_mc_results(cash_mc_results: list, mortgage_mc_results: list, holding_period: int) -> pd.DataFrame:
    """
    Processes the raw lists of simulation results into a Pandas DataFrame
    containing key calculated metrics (Net Gain, ROI).

    Args:
        cash_mc_results (list): List of result dictionaries for cash scenario.
        mortgage_mc_results (list): List of result dictionaries for mortgage scenario.
        holding_period (int): The simulation holding period in years.

    Returns:
        pd.DataFrame: DataFrame with columns for Cash Net Gain, Mortgage Net Gain,
                      Cash ROI (%), and Mortgage ROI (%).
    """
    if not cash_mc_results or not mortgage_mc_results:
        print("Warning: Empty results list passed to process_mc_results.")
        return pd.DataFrame() # Return empty DataFrame

    # Calculate metrics for each run
    cash_net_gains = np.array([calculate_net_gain(res) for res in cash_mc_results])
    mortgage_net_gains = np.array([calculate_net_gain(res) for res in mortgage_mc_results])
    cash_rois = np.array([calculate_roi(res, holding_period) for res in cash_mc_results])
    mortgage_rois = np.array([calculate_roi(res, holding_period) for res in mortgage_mc_results])

    # Create DataFrame
    df = pd.DataFrame({
        'Cash Net Gain': cash_net_gains,
        'Mortgage Net Gain': mortgage_net_gains,
        'Cash ROI (%)': cash_rois * 100,
        'Mortgage ROI (%)': mortgage_rois * 100
    })

    # Clean potential NaNs that might have resulted from edge cases or empty inputs
    df.dropna(inplace=True)

    return df

# --- Analysis Functions ---

def display_summary_stats(df_results: pd.DataFrame, config: dict = None):
    """
    Calculates and prints summary statistics for Net Gain and ROI from the processed DataFrame.

    Args:
        df_results (pd.DataFrame): DataFrame returned by process_mc_results.
        config (dict, optional): Configuration dictionary used for the simulation.
    """
    if df_results.empty:
        print("Cannot display summary statistics: DataFrame is empty.")
        return

    # Display remortgage configuration if available
    if config:
        holding_period = config.get('holding_period_years', 0)
        fixed_term = config.get('fixed_term_length_years', 5)
        num_remortgages = int(np.floor((holding_period - 1) / fixed_term))
        print("\n--- Mortgage Configuration ---")
        print(f"Fixed Term Length: {fixed_term} years")
        print(f"Number of Remortgage Events: {num_remortgages}")
        print(f"Initial Fixed Rate: {config.get('initial_mortgage_interest_rate_annual', 0)*100:.2f}%")

    print("\n--- Monte Carlo Results Summary ---")
    pd.options.display.float_format = '{:,.2f}'.format # Format pandas output

    for scenario_prefix in ["Cash", "Mortgage"]:
        print(f"\n--- {scenario_prefix} Scenario ---")
        net_gain_col = f"{scenario_prefix} Net Gain"
        roi_col = f"{scenario_prefix} ROI (%)"

        if net_gain_col in df_results.columns:
            print(f"  Net Gain:")
            stats_ng = df_results[net_gain_col].describe(percentiles=[.05, .25, .5, .75, .95])
            # Reformat specific stats for currency
            stats_ng['mean'] = f"£{stats_ng['mean']:,.0f}"
            stats_ng['std'] = f"£{stats_ng['std']:,.0f}"
            stats_ng['min'] = f"£{stats_ng['min']:,.0f}"
            stats_ng['5%'] = f"£{stats_ng['5%']:,.0f}"
            stats_ng['50%'] = f"£{stats_ng['50%']:,.0f}" # Median
            stats_ng['95%'] = f"£{stats_ng['95%']:,.0f}"
            stats_ng['max'] = f"£{stats_ng['max']:,.0f}"
            print(stats_ng[['mean', 'std', 'min', '5%', '50%', '95%', 'max']].to_string()) # Print selected stats

        if roi_col in df_results.columns:
            print(f"\n  Annualized ROI:")
            stats_roi = df_results[roi_col].describe(percentiles=[.05, .25, .5, .75, .95])
             # Reformat specific stats for percentage
            stats_roi['mean'] = f"{stats_roi['mean']:.2f}%"
            stats_roi['std'] = f"{stats_roi['std']:.2f}%"
            stats_roi['min'] = f"{stats_roi['min']:.2f}%"
            stats_roi['5%'] = f"{stats_roi['5%']:.2f}%"
            stats_roi['50%'] = f"{stats_roi['50%']:.2f}%" # Median
            stats_roi['95%'] = f"{stats_roi['95%']:.2f}%"
            stats_roi['max'] = f"{stats_roi['max']:.2f}%"
            print(stats_roi[['mean', 'std', 'min', '5%', '50%', '95%', 'max']].to_string())
    pd.reset_option('display.float_format') # Reset formatting


def display_probability_analysis(df_results: pd.DataFrame):
    """
    Calculates and prints the probability of the Mortgage scenario outperforming the Cash scenario.

    Args:
        df_results (pd.DataFrame): DataFrame returned by process_mc_results.
    """
    if df_results.empty or not all(col in df_results.columns for col in ['Mortgage Net Gain', 'Cash Net Gain', 'Mortgage ROI (%)', 'Cash ROI (%)']):
        print("\n--- Probability Analysis ---")
        print("    Not enough data or required columns missing for comparison.")
        return

    num_simulations = len(df_results)
    if num_simulations == 0:
        print("\n--- Probability Analysis ---")
        print("    No simulations to analyze.")
        return

    mortgage_outperforms_net_gain = np.sum(df_results['Mortgage Net Gain'] > df_results['Cash Net Gain']) / num_simulations
    mortgage_outperforms_roi = np.sum(df_results['Mortgage ROI (%)'] > df_results['Cash ROI (%)']) / num_simulations

    print("\n--- Probability Analysis ---")
    print(f"Probability Mortgage outperforms Cash (Net Gain): {mortgage_outperforms_net_gain:.1%}")
    print(f"Probability Mortgage outperforms Cash (Annualized ROI): {mortgage_outperforms_roi:.1%}")


# --- Plotting Function ---

def plot_mc_distributions(df_results: pd.DataFrame, num_simulations: int, config: dict = None, distribution_params: dict = None):
    """
    Generates and displays histograms/KDE plots for Net Gain and ROI distributions.

    Args:
        df_results (pd.DataFrame): DataFrame returned by process_mc_results.
        num_simulations (int): Number of simulations run (for plot title).
        config (dict, optional): Configuration dictionary used for the simulation.
        distribution_params (dict, optional): Dictionary containing distribution assumptions
            for the Monte Carlo simulation (means and standard deviations).
    """
    if df_results.empty:
        print("\n--- Plots Skipped (No results data) ---")
        return

    print("\n--- Generating Plots ---")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Made slightly taller for config box

    # Add main title with key metadata
    if config:
        # Calculate number of remortgage events
        holding_period = config.get('holding_period_years', 0)
        fixed_term = config.get('fixed_term_length_years', 5)
        num_remortgages = int(np.floor((holding_period - 1) / fixed_term))
        
        title = (f"Monte Carlo Analysis: £{config.get('property_value_initial', 0):,.0f} Property | "
                f"{holding_period}yr Hold | "
                f"{config.get('deposit_percentage', 0)*100:.0f}% Deposit | "
                f"{config.get('initial_mortgage_interest_rate_annual', 0)*100:.1f}% Initial Rate | "
                f"{fixed_term}yr Fixed Terms")
        plt.suptitle(title, fontsize=12, y=0.95)

    # Net Gain Plot
    if 'Cash Net Gain' in df_results.columns and 'Mortgage Net Gain' in df_results.columns:
        sns.histplot(data=df_results, x='Cash Net Gain', kde=True, ax=axes[0], label='Cash', color='skyblue', stat='density', alpha=0.6)
        sns.histplot(data=df_results, x='Mortgage Net Gain', kde=True, ax=axes[0], label='Mortgage', color='lightcoral', stat='density', alpha=0.6)
        axes[0].set_title(f'Distribution of Net Gains ({num_simulations:,} simulations)')
        axes[0].set_xlabel('Net Gain (£)')
        axes[0].legend()
        axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'£{x/1000:,.0f}k')) # Format as £Xk
    else:
        axes[0].set_title('Net Gain Data Missing')

    # ROI Plot
    if 'Cash ROI (%)' in df_results.columns and 'Mortgage ROI (%)' in df_results.columns:
        sns.histplot(data=df_results, x='Cash ROI (%)', kde=True, ax=axes[1], label='Cash', color='skyblue', stat='density', alpha=0.6, binwidth=1.5)
        sns.histplot(data=df_results, x='Mortgage ROI (%)', kde=True, ax=axes[1], label='Mortgage', color='lightcoral', stat='density', alpha=0.6, binwidth=1.5)
        axes[1].set_title(f'Distribution of Annualized ROI ({num_simulations:,} simulations)')
        axes[1].set_xlabel('Annualized ROI (%)')
        axes[1].legend()
    else:
         axes[1].set_title('ROI Data Missing')

    # Add configuration details if provided
    if config:
        config_text = "Base Configuration:\n"
        config_text += f"Property Value: £{config.get('property_value_initial', 0):,.0f}\n"
        config_text += f"Holding Period: {holding_period} years\n"
        config_text += f"Fixed Term Length: {fixed_term} years\n"
        config_text += f"Number of Remortgages: {num_remortgages}\n"
        config_text += f"Deposit: {config.get('deposit_percentage', 0)*100:.0f}%\n"
        config_text += f"Initial Rate: {config.get('initial_mortgage_interest_rate_annual', 0)*100:.2f}%\n"
        config_text += f"Mortgage Term: {config.get('mortgage_term_years', 0)} years"
        
        # Add text box with configuration
        plt.figtext(0.02, 0.02, config_text, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                   fontsize=9, family='monospace')

    # Add distribution assumptions if provided
    if distribution_params:
        dist_text = "Distribution Assumptions (μ ± σ):\n"
        # Property Appreciation
        if 'prop_app_mean' in distribution_params and 'prop_app_std_dev' in distribution_params:
            dist_text += f"Property App.: {distribution_params['prop_app_mean']*100:.1f}% ± {distribution_params['prop_app_std_dev']*100:.1f}%\n"
        # Alternative Investment Returns
        if 'alt_inv_mean' in distribution_params and 'alt_inv_std_dev' in distribution_params:
            dist_text += f"Alt. Investment: {distribution_params['alt_inv_mean']*100:.1f}% ± {distribution_params['alt_inv_std_dev']*100:.1f}%\n"
        # Service Charge Inflation
        if 'sc_inf_mean' in distribution_params and 'sc_inf_std_dev' in distribution_params:
            dist_text += f"Service Charge Infl.: {distribution_params['sc_inf_mean']*100:.1f}% ± {distribution_params['sc_inf_std_dev']*100:.1f}%\n"
        # Future Fixed Rates
        if 'remort_rate_mean' in distribution_params and 'remort_rate_std_dev' in distribution_params:
            dist_text += f"Future Fixed Rates: {distribution_params['remort_rate_mean']*100:.1f}% ± {distribution_params['remort_rate_std_dev']*100:.1f}%"

        # Add text box with distribution assumptions
        plt.figtext(0.35, 0.02, dist_text,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                   fontsize=9, family='monospace')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, top=0.85)  # Make room for config boxes and title
    plt.show()


# --- Optional: Add a simple test block if desired ---
if __name__ == "__main__":
    print("Testing mc_analysis functions...")
    # Create dummy data similar to what run_monte_carlo_simulation returns
    holding_p = 10
    dummy_cash = [{'initial_cash_outlay': 600000, 'final_equity_in_property': 700000, 'value_of_alternative_investments_at_end': 0, 'total_ongoing_property_costs_paid': 100000, 'is_cash_purchase': True, 'total_mortgage_interest_paid': 0}] * 10
    dummy_mort = [{'initial_cash_outlay': 150000, 'final_equity_in_property': 400000, 'value_of_alternative_investments_at_end': 700000, 'total_ongoing_property_costs_paid': 100000, 'is_cash_purchase': False, 'total_mortgage_interest_paid': 180000}] * 10

    df_test = process_mc_results(dummy_cash, dummy_mort, holding_p)
    print("\nTest DataFrame head:\n", df_test.head())

    display_summary_stats(df_test)
    display_probability_analysis(df_test)
    # plot_mc_distributions(df_test, 10) # Plotting might pop up during script run
    print("\nmc_analysis basic tests finished.")