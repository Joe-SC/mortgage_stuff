# mc_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats

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

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for the mean of the data.
    
    Args:
        data (np.array): Data to calculate CI for
        confidence (float): Confidence level (default 0.95 for 95% CI)
    
    Returns:
        tuple: (mean, lower bound, upper bound)
    """
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=se)
    return mean, ci[0], ci[1]

def display_probability_analysis(df_results: pd.DataFrame):
    """
    Performs statistical analysis comparing Mortgage vs Cash scenarios.
    Uses paired t-tests to calculate p-values and confidence intervals for the difference in performance.
    
    Args:
        df_results (pd.DataFrame): DataFrame returned by process_mc_results.
    """
    if df_results.empty or not all(col in df_results.columns for col in ['Mortgage Net Gain', 'Cash Net Gain', 'Mortgage ROI (%)', 'Cash ROI (%)']):
        print("\n--- Statistical Analysis ---")
        print("    Not enough data or required columns missing for comparison.")
        return

    num_simulations = len(df_results)
    if num_simulations == 0:
        print("\n--- Statistical Analysis ---")
        print("    No simulations to analyze.")
        return

    # Calculate differences (Mortgage - Cash)
    net_gain_diff = df_results['Mortgage Net Gain'].values - df_results['Cash Net Gain'].values
    roi_diff = df_results['Mortgage ROI (%)'].values - df_results['Cash ROI (%)'].values
    
    # Calculate confidence intervals for differences
    ng_diff_mean, ng_diff_lower, ng_diff_upper = calculate_confidence_interval(net_gain_diff)
    roi_diff_mean, roi_diff_lower, roi_diff_upper = calculate_confidence_interval(roi_diff)
    
    # Calculate confidence intervals for individual strategies
    ng_mort_mean, ng_mort_lower, ng_mort_upper = calculate_confidence_interval(df_results['Mortgage Net Gain'].values)
    ng_cash_mean, ng_cash_lower, ng_cash_upper = calculate_confidence_interval(df_results['Cash Net Gain'].values)
    roi_mort_mean, roi_mort_lower, roi_mort_upper = calculate_confidence_interval(df_results['Mortgage ROI (%)'].values)
    roi_cash_mean, roi_cash_lower, roi_cash_upper = calculate_confidence_interval(df_results['Cash ROI (%)'].values)
    
    # Perform one-sided t-tests
    # H0: difference <= 0 (mortgage doesn't outperform cash)
    # H1: difference > 0 (mortgage outperforms cash)
    t_stat_ng, p_value_ng = stats.ttest_1samp(net_gain_diff, 0)
    t_stat_roi, p_value_roi = stats.ttest_1samp(roi_diff, 0)
    
    # Convert to one-sided p-values
    p_value_ng = p_value_ng / 2 if t_stat_ng > 0 else 1 - p_value_ng / 2
    p_value_roi = p_value_roi / 2 if t_stat_roi > 0 else 1 - p_value_roi / 2

    print("\n--- Statistical Analysis ---")
    print("Net Gain Analysis:")
    print("  Mortgage Strategy:")
    print(f"    Mean: £{ng_mort_mean:,.0f}")
    print(f"    95% CI: [£{ng_mort_lower:,.0f}, £{ng_mort_upper:,.0f}]")
    print("  Cash Strategy:")
    print(f"    Mean: £{ng_cash_mean:,.0f}")
    print(f"    95% CI: [£{ng_cash_lower:,.0f}, £{ng_cash_upper:,.0f}]")
    print("  Difference (Mortgage - Cash):")
    print(f"    Mean: £{ng_diff_mean:,.0f}")
    print(f"    95% CI: [£{ng_diff_lower:,.0f}, £{ng_diff_upper:,.0f}]")
    print(f"    t-statistic: {t_stat_ng:.2f}")
    print(f"    p-value (one-sided): {p_value_ng:.2e}")
    
    print("\nROI Analysis:")
    print("  Mortgage Strategy:")
    print(f"    Mean: {roi_mort_mean:.2f}%")
    print(f"    95% CI: [{roi_mort_lower:.2f}%, {roi_mort_upper:.2f}%]")
    print("  Cash Strategy:")
    print(f"    Mean: {roi_cash_mean:.2f}%")
    print(f"    95% CI: [{roi_cash_lower:.2f}%, {roi_cash_upper:.2f}%]")
    print("  Difference (Mortgage - Cash):")
    print(f"    Mean: {roi_diff_mean:.2f}%")
    print(f"    95% CI: [{roi_diff_lower:.2f}%, {roi_diff_upper:.2f}%]")
    print(f"    t-statistic: {t_stat_roi:.2f}")
    print(f"    p-value (one-sided): {p_value_roi:.2e}")
    
    # Calculate proportion of simulations where mortgage outperforms
    prop_outperform_ng = np.mean(net_gain_diff > 0)
    prop_outperform_roi = np.mean(roi_diff > 0)
    
    print("\nProportion of Simulations where Mortgage Outperforms:")
    print(f"  Net Gain: {prop_outperform_ng:.1%}")
    print(f"  Annualized ROI: {prop_outperform_roi:.1%}")

    # Effect size (Cohen's d)
    cohens_d_ng = np.mean(net_gain_diff) / np.std(net_gain_diff)
    cohens_d_roi = np.mean(roi_diff) / np.std(roi_diff)
    
    print("\nEffect Size (Cohen's d):")
    print(f"  Net Gain: {cohens_d_ng:.2f}")
    print(f"  Annualized ROI: {cohens_d_roi:.2f}")

    # Interpretation
    print("\nInterpretation:")
    if p_value_ng < 0.05 and ng_diff_lower > 0:
        print("  Net Gain: Strong evidence that mortgage strategy outperforms cash")
        print(f"  Expected outperformance: £{ng_diff_mean:,.0f} (95% CI: £{ng_diff_lower:,.0f} to £{ng_diff_upper:,.0f})")
    elif p_value_ng < 0.05:
        print("  Net Gain: Evidence of difference, but direction uncertain")
    else:
        print("  Net Gain: No strong evidence of difference between strategies")

    if p_value_roi < 0.05 and roi_diff_lower > 0:
        print("  ROI: Strong evidence that mortgage strategy outperforms cash")
        print(f"  Expected outperformance: {roi_diff_mean:.2f}% (95% CI: {roi_diff_lower:.2f}% to {roi_diff_upper:.2f}%)")
    elif p_value_roi < 0.05:
        print("  ROI: Evidence of difference, but direction uncertain")
    else:
        print("  ROI: No strong evidence of difference between strategies")


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


def calculate_net_position_buy_cash(results_dict):
    # Final net worth position for buying scenario
    if not results_dict: return np.nan
    return (results_dict.get('final_equity_in_property', 0) -
            results_dict.get('total_ongoing_property_costs_paid', 0))

def calculate_net_position_rent(results_dict):
    # Final net worth position for renting scenario
    if not results_dict: return np.nan
    return (results_dict.get('value_of_alternative_investments_at_end', 0) -
            results_dict.get('total_rent_paid', 0) -
            results_dict.get('total_renter_ongoing_costs_paid', 0))

# ROI calculation is trickier for rent vs. buy with full cash.
# The "initial outlay" is the full property price for buying.
# For renting, the "initial outlay" towards housing is minimal (deposit).
# The comparison often focuses on the difference in final net worth.
# Or, ROI for renting can be seen as the ROI on the alternative investments made with the cash
# that *would* have gone into the property.

def process_buy_vs_rent_mc_results(buy_cash_mc_results: list, rent_mc_results: list, holding_period: int) -> pd.DataFrame:
    if not buy_cash_mc_results or not rent_mc_results:
        return pd.DataFrame()

    buy_cash_net_positions = np.array([calculate_net_position_buy_cash(res) + res.get('initial_cash_outlay',0) for res in buy_cash_mc_results])
    # Adding back initial_cash_outlay to represent final asset value relative to doing nothing with that cash.
    # Or, more simply, just final_equity_in_property already accounts for initial outlay implicitly if we define "Net Gain" as
    # (final_equity - ongoing_costs) - initial_cash_outlay.
    # Let's define "Final Net Worth" for easier comparison:
    # Buy_Cash: final_equity_in_property (as it's net of selling costs)
    # Rent: value_of_alternative_investments_at_end - total_rent_paid - total_renter_ongoing_costs
    # The "initial_cash_outlay" for buying scenario is the benchmark for what's invested in the Rent scenario.
    
    buy_final_net_worth = np.array([res.get('final_equity_in_property',0) for res in buy_cash_mc_results])
    
    rent_final_net_worth = np.array([
        res.get('value_of_alternative_investments_at_end', 0) - res.get('total_rent_paid', 0) - res.get('total_renter_ongoing_costs_paid',0)
        for res in rent_mc_results
    ])
    
    # For ROI, let's consider the initial capital (property_price + buying_costs) as the base for both.
    # Buy ROI: ((final_equity_in_property / initial_cash_outlay_buy_prop)**(1/hp)) - 1
    # Rent ROI: ((value_of_alt_invest_end / cash_available_for_investment_rent)**(1/hp)) - 1 
    #           (This needs adjustment for rent paid, as that's a cost against the alt. investment strategy)
    # A simpler "Net Gain over doing nothing" might be:
    # Buy Net Gain: final_equity_in_property - initial_cash_outlay_buy
    # Rent Net Gain: (value_of_alt_inv_end - total_rent_paid - total_renter_ongoing_costs) - cash_available_for_investment_rent 
    #                (This results in negative if rent > investment growth. The "cash_available_for_investment" is the opportunity cost)
    # For this, let's use a "Net Financial Advantage" metric instead of direct ROI if it's confusing.
    # Net Financial Advantage = Final Net Worth (Strategy) - Final Net Worth (Baseline of just investing the initial sum and paying rent from elsewhere or vice-versa)

    # Sticking to Net Gain as "Final Assets - Initial Outlay - All Running Costs" is clearer for comparison.
    # Initial Outlay (Buy): property_value_initial + total_initial_buying_costs
    # Initial Outlay (Rent): initial_rental_deposit (cash invested in alternatives is the same base as buying)
    
    cash_net_gains = np.array([(res.get('final_equity_in_property',0) - res.get('initial_cash_outlay',0) - res.get('total_ongoing_property_costs_paid',0) ) for res in buy_cash_mc_results])
    
    # For rent, the "initial_cash_outlay" for the *strategy* is the rental deposit.
    # The "cash_available_for_investment" is the opportunity.
    # Net Gain (Rent) = value_of_alt_inv_end - initial_rental_deposit - total_rent_paid - total_renter_ongoing_costs
    rent_net_gains = np.array([(res.get('value_of_alternative_investments_at_end',0) - res.get('initial_cash_outlay',0) - res.get('total_rent_paid',0) - res.get('total_renter_ongoing_costs_paid',0)) for res in rent_mc_results])

    # For ROI, let's make it ROI on the "cash_if_buying" amount for both scenarios.
    # Buy ROI: (Net Gain Buy / cash_if_buying)
    # Rent ROI: (Net Gain Rent / cash_if_buying) -- This is also a bit forced.
    # Better: Compare absolute Net Gains or Final Net Worth.
    
    # Let's use the ROI calculation similar to your mc_analysis.py
    # but initial outlay for rent is the deposit. This highlights return on actual cash tied up.
    
    initial_outlay_buy = np.array([res.get('initial_cash_outlay', 0) for res in buy_cash_mc_results])
    initial_outlay_rent = np.array([res.get('initial_cash_outlay', 0) for res in rent_mc_results]) # This is the deposit

    cash_rois = []
    for i, res_c in enumerate(buy_cash_mc_results):
        ending_val = initial_outlay_buy[i] + cash_net_gains[i]
        if initial_outlay_buy[i] > 0 and holding_period > 0:
            roi_c = ((ending_val / initial_outlay_buy[i]) ** (1/holding_period)) - 1
        else:
            roi_c = np.nan
        cash_rois.append(roi_c)

    rent_rois = []
    # For renting, the ROI should reflect the growth of the *entire sum of cash* that wasn't spent on the house,
    # offset by rent. So, (Value of Alt Investments - Rent Paid) / Original Cash Pool.
    # Or ROI on actual "tied up capital" (deposit) which is less meaningful for comparison.
    # Let's re-frame ROI for renting: The "investment" is the large sum of cash NOT spent on the house.
    # Net gain from this sum = (growth of sum) - (cost of renting).
    
    cash_if_buying_for_rent_roi = np.array([res.get('cash_available_for_investment', 0) for res in rent_mc_results]) #This needs to be passed into results
    
    for i, res_r in enumerate(rent_mc_results):
        # Net gain considering the 'cash_if_buying' as the base that was alternatively invested
        net_gain_rent_alt_inv = (res_r.get('value_of_alternative_investments_at_end',0) -
                                 cash_if_buying_for_rent_roi[i] - # subtract the principal
                                 res_r.get('total_rent_paid',0) - 
                                 res_r.get('total_renter_ongoing_costs_paid',0))
        
        ending_val_rent = cash_if_buying_for_rent_roi[i] + net_gain_rent_alt_inv # This effectively becomes (value_of_alt_inv_end - total_rent - total_renter_costs)
        
        if cash_if_buying_for_rent_roi[i] > 0 and holding_period > 0:
            roi_r = ((ending_val_rent / cash_if_buying_for_rent_roi[i]) ** (1/holding_period)) - 1
        else:
            roi_r = np.nan
        rent_rois.append(roi_r)


    df = pd.DataFrame({
        'Buy (Cash) Net Gain': cash_net_gains,
        'Rent Net Gain': rent_net_gains,
        'Buy (Cash) ROI (%)': np.array(cash_rois) * 100,
        'Rent ROI (%)': np.array(rent_rois) * 100 
    })
    df.dropna(inplace=True)
    return df
