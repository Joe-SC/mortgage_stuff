# mortgage_mc.py

import numpy as np
import numpy_financial as npf
import copy

# --- BASE CASE ASSUMPTIONS (Stored as a private variable) ---
_BASE_CONFIG = {
    "property_value_initial": 575000.0,
    "buying_costs_percentage_other": 0.015, # Legal, survey etc. (1.5%) SDLT separate.
    "annual_property_appreciation_rate": 0.03, # Base average for single runs
    "holding_period_years": 10,
    "service_charge_annual_initial": 6500.0,
    "ground_rent_annual": 200.0,
    "council_tax_annual_initial": 2000.0, # Estimate
    "maintenance_allowance_annual": 600.0,
    "insurance_annual": 300.0, # Contents insurance
    "service_charge_inflation_rate": 0.03, # Base average for single runs
    "selling_costs_percentage": 0.02, # 2% agent/legal on sale

    # Mortgage specific
    "deposit_percentage": 0.20,
    "mortgage_interest_rate_annual": 0.045, # Base average/fixed rate
    "mortgage_term_years": 25,
    "mortgage_arrangement_fee": 1500.0,

    # Alternative investment specific
    "alternative_investment_return_rate_annual": 0.05, # Base average for single runs
}

def get_base_config():
    """
    Returns a fresh deep copy of the base configuration dictionary.
    Ensures modifications in one simulation run don't affect others.

    Returns:
        dict: A deep copy of the base configuration.
    """
    return copy.deepcopy(_BASE_CONFIG)

# --- Helper Functions ---
def calculate_stamp_duty(price):
    """
    Calculates Standard UK Stamp Duty Land Tax (SDLT) applicable from April 1, 2025.
    Assumes standard rates (no First-Time Buyer relief due to price > £500k).

    Args:
        price (float): The purchase price of the property.

    Returns:
        float: The calculated SDLT amount.
    """
    if price <= 125000:
        return 0.0
    elif price <= 250000:
        # 2% on portion £125,001 - £250,000
        return (price - 125000) * 0.02
    elif price <= 925000:
        # £2,500 (from 2% band) + 5% on portion £250,001 - £925,000
        return 2500.0 + (price - 250000) * 0.05
    elif price <= 1500000:
        # £2,500 (2% band) + £33,750 (5% band) + 10% on portion £925,001 - £1.5m
        return 2500.0 + 33750.0 + (price - 925000) * 0.10
    else:
        # £2,500 + £33,750 + £57,500 (10% band) + 12% on portion above £1.5m
        return 2500.0 + 33750.0 + 57500.0 + (price - 1500000) * 0.12


# --- Modified Simulation Function for Annual Rates ---
def simulate_investment_annual_rates(
    # Pass most base config items directly using **config
    property_value_initial: float,
    buying_costs_percentage_other: float,
    holding_period_years: int,
    service_charge_annual_initial: float,
    ground_rent_annual: float,
    council_tax_annual_initial: float,
    maintenance_allowance_annual: float,
    insurance_annual: float,
    selling_costs_percentage: float,
    # Accept annual rate sequences for varying params:
    property_appreciation_rates_annual: list,
    service_charge_inflation_rates_annual: list,
    alternative_investment_return_rates_annual: list,
    # Keep mortgage rate fixed within this simulation run:
    mortgage_interest_rate_annual: float,
    # Other necessary params
    is_cash_purchase: bool,
    deposit_percentage: float,
    mortgage_term_years: int,
    mortgage_arrangement_fee: float,
    **kwargs # To absorb any other unused config items
):
    """
    Simulates one investment scenario run using annually varying rates for
    key uncertain assumptions.

    Args:
        (Parameters corresponding to _BASE_CONFIG keys)
        property_appreciation_rates_annual (list): Year-by-year appreciation rates.
        service_charge_inflation_rates_annual (list): Year-by-year inflation rates for SC/CTax.
        alternative_investment_return_rates_annual (list): Year-by-year alt investment returns.
        is_cash_purchase (bool): Determines the scenario.

    Returns:
        dict: A dictionary containing the key results of the simulation run.
    """
    # --- Input Validation ---
    if not (len(property_appreciation_rates_annual) == holding_period_years and \
            len(service_charge_inflation_rates_annual) == holding_period_years and \
            len(alternative_investment_return_rates_annual) == holding_period_years):
        raise ValueError("Length of annual rate lists must equal holding_period_years")

    # --- Initial Costs ---
    stamp_duty = calculate_stamp_duty(property_value_initial)
    other_buying_costs = property_value_initial * buying_costs_percentage_other
    total_initial_buying_costs = stamp_duty + other_buying_costs

    # --- Initial State Variables ---
    current_property_value = property_value_initial
    total_ongoing_costs_paid = 0.0
    total_mortgage_interest_paid = 0.0
    value_of_alternative_investments = 0.0
    initial_cash_outlay = 0.0
    remaining_mortgage_balance = 0.0
    loan_amount = 0.0 # Initialize loan amount

    # --- Scenario Specific Setup ---
    if is_cash_purchase:
        initial_cash_outlay = property_value_initial + total_initial_buying_costs
        value_of_alternative_investments = 0.0 # No alternative investments
    else: # Mortgage Purchase
        deposit_amount = property_value_initial * deposit_percentage
        loan_amount = property_value_initial - deposit_amount
        remaining_mortgage_balance = loan_amount # Initial balance
        initial_cash_outlay = deposit_amount + total_initial_buying_costs + mortgage_arrangement_fee
        # Cash freed up is invested
        cash_freed_up_for_alt_investment = loan_amount
        value_of_alternative_investments = cash_freed_up_for_alt_investment # Initial investment value

    # --- Annual Simulation ---
    current_service_charge = service_charge_annual_initial
    current_council_tax = council_tax_annual_initial

    # Store annual data if needed for detailed plotting/analysis (optional)
    # property_values_over_time = [current_property_value]
    # remaining_mortgage_balances_over_time = [remaining_mortgage_balance]
    # alternative_investments_values_over_time = [value_of_alternative_investments]

    for year in range(1, holding_period_years + 1):
        # Use the specific rate for this year from the input lists
        prop_app_rate = property_appreciation_rates_annual[year-1]
        sc_inf_rate = service_charge_inflation_rates_annual[year-1]
        alt_inv_rate = alternative_investment_return_rates_annual[year-1] # Used only in mortgage

        # 1. Property Appreciation
        current_property_value *= (1 + prop_app_rate)
        # property_values_over_time.append(current_property_value)

        # 2. Ongoing Costs for the year
        annual_property_related_costs = (
            current_service_charge +
            ground_rent_annual +
            current_council_tax +
            maintenance_allowance_annual +
            insurance_annual
        )
        total_ongoing_costs_paid += annual_property_related_costs

        # 3. Inflate costs for *next* year's calculation
        current_service_charge *= (1 + sc_inf_rate)
        current_council_tax *= (1 + sc_inf_rate) # Assuming council tax inflates similarly

        # 4. Mortgage Scenario Calculations
        if not is_cash_purchase:
            if remaining_mortgage_balance > 0:
                # Calculate interest for *this year* based on balance at start of year
                interest_this_year = remaining_mortgage_balance * mortgage_interest_rate_annual
                total_mortgage_interest_paid += interest_this_year

                # Calculate Principal Repayment for the year to update balance
                # Using npf.pmt to get the constant payment for remaining term/balance
                # Note: Assumes interest rate is constant for this calculation within the year
                remaining_term_years = mortgage_term_years - (year - 1)
                if remaining_term_years > 0:
                    if mortgage_interest_rate_annual > 0:
                        # Calculate the theoretical full monthly payment for P&I
                        monthly_pmt_val = npf.pmt(
                            mortgage_interest_rate_annual / 12,
                            remaining_term_years * 12,
                            -remaining_mortgage_balance # Present value is the outstanding balance
                        )
                        annual_payment_total = monthly_pmt_val * 12
                        # Principal paid is the total payment less the interest paid
                        principal_paid_this_year = annual_payment_total - interest_this_year
                    else: # Handle 0% interest rate case
                        principal_paid_this_year = remaining_mortgage_balance / remaining_term_years

                    # Ensure principal paid doesn't exceed remaining balance
                    principal_paid_this_year = min(principal_paid_this_year, remaining_mortgage_balance)
                    remaining_mortgage_balance -= principal_paid_this_year
                    # Ensure balance doesn't go below zero due to rounding etc.
                    remaining_mortgage_balance = max(0, remaining_mortgage_balance)
                else:
                     # Should be zero if term ended, defensive coding
                     remaining_mortgage_balance = 0


            # remaining_mortgage_balances_over_time.append(remaining_mortgage_balance)

            # 5. Alternative investments grow using the specific year's rate
            value_of_alternative_investments *= (1 + alt_inv_rate)
            # alternative_investments_values_over_time.append(value_of_alternative_investments)

        # else: # Optional tracking for cash scenario
            # remaining_mortgage_balances_over_time.append(0)
            # alternative_investments_values_over_time.append(0)

    # --- At End of Holding Period ---
    selling_costs = current_property_value * selling_costs_percentage
    net_sale_proceeds_from_property = current_property_value - selling_costs
    final_equity_in_property = net_sale_proceeds_from_property - remaining_mortgage_balance

    # Return key final results
    return {
        "is_cash_purchase": is_cash_purchase,
        "initial_cash_outlay": initial_cash_outlay,
        "final_property_value": current_property_value,
        "selling_costs": selling_costs,
        "net_proceeds_from_property_sale": net_sale_proceeds_from_property,
        "remaining_mortgage_balance_at_end": remaining_mortgage_balance,
        "final_equity_in_property": final_equity_in_property,
        "total_ongoing_property_costs_paid": total_ongoing_costs_paid,
        "total_mortgage_interest_paid": total_mortgage_interest_paid,
        "value_of_alternative_investments_at_end": value_of_alternative_investments,
        # Add tracked lists here if needed:
        # "property_values_over_time": property_values_over_time,
        # "remaining_mortgage_balances_over_time": remaining_mortgage_balances_over_time,
        # "alternative_investments_values_over_time": alternative_investments_values_over_time
    }


# --- Monte Carlo Runner Function ---
def run_monte_carlo_simulation(
    base_config: dict,
    num_simulations: int = 10000,
    # Define distributions (mean, std_dev) for key variables
    prop_app_mean: float = 0.03, prop_app_std_dev: float = 0.04,
    alt_inv_mean: float = 0.05, alt_inv_std_dev: float = 0.07,
    sc_inf_mean: float = 0.035, sc_inf_std_dev: float = 0.025
):
    """
    Runs the investment simulation multiple times using Monte Carlo
    for specified variables.

    Args:
        base_config (dict): Base configuration dictionary from get_base_config().
        num_simulations (int): Number of simulations to run.
        prop_app_mean (float): Mean annual property appreciation rate.
        prop_app_std_dev (float): Standard deviation of property appreciation rate.
        alt_inv_mean (float): Mean annual alternative investment return rate.
        alt_inv_std_dev (float): Standard deviation of alt investment return rate.
        sc_inf_mean (float): Mean annual service charge/cost inflation rate.
        sc_inf_std_dev (float): Standard deviation of service charge/cost inflation rate.

    Returns:
        tuple: A tuple containing two lists: (cash_results_list, mortgage_results_list),
               where each list contains result dictionaries from each simulation run.
    """
    cash_results_list = []
    mortgage_results_list = []
    holding_period = base_config['holding_period_years']

    # Use a single random number generator for consistency if desired
    rng = np.random.default_rng()

    for i in range(num_simulations):
        # Generate sequences of random rates for this run using the RNG
        prop_app_rates = rng.normal(prop_app_mean, prop_app_std_dev, holding_period)
        alt_inv_rates = rng.normal(alt_inv_mean, alt_inv_std_dev, holding_period)
        sc_inf_rates = rng.normal(sc_inf_mean, sc_inf_std_dev, holding_period)

        # Apply constraints if necessary (e.g., prevent nonsensical negative rates)
        prop_app_rates = np.maximum(-0.99, prop_app_rates) # Limit loss to -99%
        alt_inv_rates = np.maximum(-0.99, alt_inv_rates)
        # Allow for cost deflation but perhaps limit extreme deflation?
        sc_inf_rates = np.maximum(-0.1, sc_inf_rates) # Limit deflation to -10%

        # --- Run simulation for CASH scenario ---
        # Create a fresh config copy for this specific simulation run
        sim_config_cash = copy.deepcopy(base_config) # Important!
        
        # Add the rate sequences to the config instead of passing separately
        sim_config_cash.update({
            'is_cash_purchase': True,
            'property_appreciation_rates_annual': prop_app_rates.tolist(),
            'service_charge_inflation_rates_annual': sc_inf_rates.tolist(),
            'alternative_investment_return_rates_annual': [0.0] * holding_period
        })

        cash_res = simulate_investment_annual_rates(**sim_config_cash)
        cash_results_list.append(cash_res)

        # --- Run simulation for MORTGAGE scenario ---
        # Use the SAME random rate sequences generated above for a fair comparison
        sim_config_mort = copy.deepcopy(base_config)
        
        # Add the rate sequences to the config instead of passing separately
        sim_config_mort.update({
            'is_cash_purchase': False,
            'property_appreciation_rates_annual': prop_app_rates.tolist(),
            'service_charge_inflation_rates_annual': sc_inf_rates.tolist(),
            'alternative_investment_return_rates_annual': alt_inv_rates.tolist()
        })

        mort_res = simulate_investment_annual_rates(**sim_config_mort)
        mortgage_results_list.append(mort_res)

        # Optional: Progress indicator for long simulations
        if (i + 1) % (num_simulations // 10) == 0:
             print(f"  Completed simulation {i+1}/{num_simulations}")

    return cash_results_list, mortgage_results_list


# --- Optional: Main block for basic testing within the module ---
if __name__ == "__main__":
    print("Running basic module test...")
    test_config = get_base_config()
    # Modify config for testing if needed
    # test_config['holding_period_years'] = 5

    # Test the annual rates simulation function with fixed rates first
    print("\nTesting simulate_investment_annual_rates with fixed rates...")
    n_years = test_config['holding_period_years']
    fixed_prop_app = [test_config['annual_property_appreciation_rate']] * n_years
    fixed_sc_inf = [test_config['service_charge_inflation_rate']] * n_years
    fixed_alt_inv = [test_config['alternative_investment_return_rate_annual']] * n_years

    test_cash_fixed = simulate_investment_annual_rates(
        **test_config, is_cash_purchase=True,
        property_appreciation_rates_annual=fixed_prop_app,
        service_charge_inflation_rates_annual=fixed_sc_inf,
        alternative_investment_return_rates_annual=[0.0]*n_years,
        holding_period_years=n_years
    )
    test_mort_fixed = simulate_investment_annual_rates(
        **test_config, is_cash_purchase=False,
        property_appreciation_rates_annual=fixed_prop_app,
        service_charge_inflation_rates_annual=fixed_sc_inf,
        alternative_investment_return_rates_annual=fixed_alt_inv,
        holding_period_years=n_years
    )
    print(f"Test Cash (Fixed Rates) - Final Equity: £{test_cash_fixed['final_equity_in_property']:,.0f}")
    print(f"Test Mortgage (Fixed Rates) - Final Equity: £{test_mort_fixed['final_equity_in_property']:,.0f}, Alt Inv: £{test_mort_fixed['value_of_alternative_investments_at_end']:,.0f}")


    # Test the Monte Carlo runner with a small number of simulations
    print("\nTesting run_monte_carlo_simulation (10 runs)...")
    mc_cash, mc_mort = run_monte_carlo_simulation(
        base_config=test_config,
        num_simulations=10,
        prop_app_mean=test_config['annual_property_appreciation_rate'], # Use base means for test
        alt_inv_mean=test_config['alternative_investment_return_rate_annual'],
        sc_inf_mean=test_config['service_charge_inflation_rate']
        # Use default std devs or specify test ones
    )
    print(f"Monte Carlo test completed.")
    print(f"  Generated {len(mc_cash)} cash results.")
    print(f"  Example Cash Final Equity: £{mc_cash[0]['final_equity_in_property']:,.0f}")
    print(f"  Generated {len(mc_mort)} mortgage results.")
    print(f"  Example Mortgage Final Equity: £{mc_mort[0]['final_equity_in_property']:,.0f}")
    print(f"  Example Mortgage Alt Inv Value: £{mc_mort[0]['value_of_alternative_investments_at_end']:,.0f}")
    print("\nModule basic tests finished.")