# mortgage_mc.py (Version with 5-Year Remortgage Logic)

import numpy as np
import numpy_financial as npf
import copy

# --- BASE CONFIGURATION ---
_BASE_CONFIG = {
    "property_value_initial": 575000.0,
    "buying_costs_percentage_other": 0.015, # Legal, survey etc.
    "annual_property_appreciation_rate": 0.03, # Mean assumption for MC
    "holding_period_years": 10,
    "service_charge_annual_initial": 6500.0,
    "ground_rent_annual": 200.0,
    "council_tax_annual_initial": 2000.0,
    "maintenance_allowance_annual": 600.0,
    "insurance_annual": 300.0, # Contents
    "service_charge_inflation_rate": 0.03, # Mean assumption for MC
    "selling_costs_percentage": 0.02,

    # --- Mortgage Specific (Updated) ---
    "deposit_percentage": 0.20,
    "initial_mortgage_interest_rate_annual": 0.045, # Rate for Year 1-5
    "mortgage_term_years": 25,                      # Original full term
    "mortgage_arrangement_fee": 1500.0,             # Fee for initial mortgage
    "remortgage_fee": 1000.0,                       # Fee incurred at Year 5/Start of Year 6
    # Note: Remortgage interest rate will be generated randomly in MC run

    # --- Alternative Investment Specific ---
    "alternative_investment_return_rate_annual": 0.05, # Mean assumption for MC
}

def get_base_config():
    """Returns a fresh deep copy of the base configuration."""
    return copy.deepcopy(_BASE_CONFIG)

# --- Helper Functions ---
def calculate_stamp_duty(price):
    """Calculates Standard UK SDLT applicable from April 1, 2025."""
    if price <= 125000: return 0.0
    elif price <= 250000: return (price - 125000) * 0.02
    elif price <= 925000: return 2500.0 + (price - 250000) * 0.05
    elif price <= 1500000: return 2500.0 + 33750.0 + (price - 925000) * 0.10
    else: return 2500.0 + 33750.0 + 57500.0 + (price - 1500000) * 0.12

# --- Simulation Function with Remortgage Logic ---
def simulate_investment_annual_rates(
    # Pass base config items using **config
    property_value_initial: float,
    buying_costs_percentage_other: float,
    holding_period_years: int,
    service_charge_annual_initial: float,
    ground_rent_annual: float,
    council_tax_annual_initial: float,
    maintenance_allowance_annual: float,
    insurance_annual: float,
    selling_costs_percentage: float,
    # Annual rate sequences
    property_appreciation_rates_annual: list,
    service_charge_inflation_rates_annual: list,
    alternative_investment_return_rates_annual: list,
    # Mortgage Rates & Fees
    initial_mortgage_interest_rate_annual: float,
    remortgage_rate_annual_year_6_onwards: float, # Rate from year 6+ for this run
    remortgage_fee: float,
    # Other necessary params
    is_cash_purchase: bool,
    deposit_percentage: float,
    mortgage_term_years: int,
    mortgage_arrangement_fee: float,
    **kwargs # Absorb unused config items
):
    """Simulates one scenario run with annual rates and 5-year remortgage logic."""
    # --- Input Validation ---
    if not (len(property_appreciation_rates_annual) == holding_period_years and \
            len(service_charge_inflation_rates_annual) == holding_period_years and \
            len(alternative_investment_return_rates_annual) == holding_period_years):
        raise ValueError("Length of annual rate lists must equal holding_period_years")
    if holding_period_years < 5 and not is_cash_purchase:
         print("Warning: Holding period less than 5 years, remortgage logic may not trigger.")


    # --- Initial Costs & State ---
    stamp_duty = calculate_stamp_duty(property_value_initial)
    other_buying_costs = property_value_initial * buying_costs_percentage_other
    total_initial_buying_costs = stamp_duty + other_buying_costs

    current_property_value = property_value_initial
    total_ongoing_costs_paid = 0.0
    total_mortgage_interest_paid = 0.0
    total_remortgage_fees_paid = 0.0
    value_of_alternative_investments = 0.0
    initial_cash_outlay = 0.0
    remaining_mortgage_balance = 0.0
    loan_amount = 0.0
    current_monthly_payment = 0.0 # Store current calculated monthly P&I payment

    if is_cash_purchase:
        initial_cash_outlay = property_value_initial + total_initial_buying_costs
    else: # Mortgage Setup
        deposit_amount = property_value_initial * deposit_percentage
        loan_amount = property_value_initial - deposit_amount
        remaining_mortgage_balance = loan_amount
        initial_cash_outlay = deposit_amount + total_initial_buying_costs + mortgage_arrangement_fee
        value_of_alternative_investments = loan_amount # Initial investment
        # Calculate initial monthly payment
        if remaining_mortgage_balance > 0 and initial_mortgage_interest_rate_annual >= 0 and mortgage_term_years > 0:
             if initial_mortgage_interest_rate_annual > 0:
                 current_monthly_payment = npf.pmt(
                     initial_mortgage_interest_rate_annual / 12,
                     mortgage_term_years * 12,
                     -remaining_mortgage_balance
                 )
             else: # 0% interest
                 current_monthly_payment = remaining_mortgage_balance / (mortgage_term_years * 12)


    current_service_charge = service_charge_annual_initial
    current_council_tax = council_tax_annual_initial

    # --- Annual Simulation Loop ---
    for year in range(1, holding_period_years + 1):

        # Determine current year's mortgage rate
        current_year_mortgage_rate = 0.0
        if not is_cash_purchase:
             current_year_mortgage_rate = (
                 initial_mortgage_interest_rate_annual if year <= 5
                 else remortgage_rate_annual_year_6_onwards
             )
             # Ensure rate is not negative (defensive)
             current_year_mortgage_rate = max(0.0, current_year_mortgage_rate)


             # --- Remortgage Event at start of Year 6 ---
             if year == 6 and remaining_mortgage_balance > 0:
                 total_remortgage_fees_paid += remortgage_fee
                 remaining_term_years_at_remortgage = mortgage_term_years - 5 # 5 years have passed
                 if remaining_term_years_at_remortgage > 0:
                     # Recalculate monthly payment based on new rate and remaining term/balance
                     if remortgage_rate_annual_year_6_onwards > 0:
                         current_monthly_payment = npf.pmt(
                             remortgage_rate_annual_year_6_onwards / 12,
                             remaining_term_years_at_remortgage * 12,
                             -remaining_mortgage_balance
                         )
                     else: # Handle 0% remortgage rate
                         current_monthly_payment = remaining_mortgage_balance / (remaining_term_years_at_remortgage * 12)
                 else: # If original term was <= 5 years
                      current_monthly_payment = 0 # Loan should be paid off or handled differently


        # Use rates specific to this year
        prop_app_rate = property_appreciation_rates_annual[year-1]
        sc_inf_rate = service_charge_inflation_rates_annual[year-1]
        alt_inv_rate = alternative_investment_return_rates_annual[year-1]

        # 1. Property Appreciation
        current_property_value *= (1 + prop_app_rate)

        # 2. Ongoing Costs for the year (excluding mortgage interest/remort fees)
        annual_property_related_costs = (
            current_service_charge +
            ground_rent_annual +
            current_council_tax +
            maintenance_allowance_annual +
            insurance_annual
        )
        total_ongoing_costs_paid += annual_property_related_costs

        # 3. Inflate costs for next year
        current_service_charge *= (1 + sc_inf_rate)
        current_council_tax *= (1 + sc_inf_rate)

        # 4. Mortgage Calculations (if applicable)
        if not is_cash_purchase and remaining_mortgage_balance > 0:
            # Calculate interest for this year based on balance at START of year & current rate
            interest_this_year = remaining_mortgage_balance * current_year_mortgage_rate
            total_mortgage_interest_paid += interest_this_year

            # Calculate total payment for the year based on current monthly payment
            annual_payment_total = current_monthly_payment * 12

            # Calculate principal paid this year
            principal_paid_this_year = annual_payment_total - interest_this_year

            # Ensure principal paid doesn't exceed remaining balance or go negative
            principal_paid_this_year = max(0, principal_paid_this_year)
            principal_paid_this_year = min(principal_paid_this_year, remaining_mortgage_balance)

            remaining_mortgage_balance -= principal_paid_this_year
            remaining_mortgage_balance = max(0, remaining_mortgage_balance) # Ensure non-negative

        # 5. Alternative Investments Grow (if applicable)
        if not is_cash_purchase:
            value_of_alternative_investments *= (1 + alt_inv_rate)

    # --- At End of Holding Period ---
    selling_costs = current_property_value * selling_costs_percentage
    net_sale_proceeds_from_property = current_property_value - selling_costs
    final_equity_in_property = net_sale_proceeds_from_property - remaining_mortgage_balance

    # Add remortgage fees to total costs paid for net gain calculation
    total_costs_paid_incl_remort_fees = total_ongoing_costs_paid + total_remortgage_fees_paid

    return {
        "is_cash_purchase": is_cash_purchase,
        "initial_cash_outlay": initial_cash_outlay,
        "final_property_value": current_property_value,
        "selling_costs": selling_costs,
        "net_proceeds_from_property_sale": net_sale_proceeds_from_property,
        "remaining_mortgage_balance_at_end": remaining_mortgage_balance,
        "final_equity_in_property": final_equity_in_property,
        "total_ongoing_property_costs_paid": total_costs_paid_incl_remort_fees, # Now includes remortgage fees
        "total_mortgage_interest_paid": total_mortgage_interest_paid,
        "value_of_alternative_investments_at_end": value_of_alternative_investments,
    }


# --- Monte Carlo Runner Function (Updated) ---
def run_monte_carlo_simulation(
    base_config: dict,
    num_simulations: int = 10000,
    # Define distributions for varying params
    prop_app_mean: float = 0.03, prop_app_std_dev: float = 0.04,
    alt_inv_mean: float = 0.05, alt_inv_std_dev: float = 0.07,
    sc_inf_mean: float = 0.035, sc_inf_std_dev: float = 0.025,
    # --- Add distribution params for remortgage rate ---
    remort_rate_mean: float = 0.045, # Mean expected rate in 5 years
    remort_rate_std_dev: float = 0.015 # Volatility/uncertainty of rate in 5 years
):
    """Runs the simulation many times with random inputs, including remortgage rate."""
    cash_results_list = []
    mortgage_results_list = []
    holding_period = base_config['holding_period_years']
    initial_mortgage_rate = base_config['initial_mortgage_interest_rate_annual'] # Get fixed initial rate

    rng = np.random.default_rng()

    for i in range(num_simulations):
        # Generate sequences of random rates for this run
        prop_app_rates = rng.normal(prop_app_mean, prop_app_std_dev, holding_period)
        alt_inv_rates = rng.normal(alt_inv_mean, alt_inv_std_dev, holding_period)
        sc_inf_rates = rng.normal(sc_inf_mean, sc_inf_std_dev, holding_period)

        # Generate the single remortgage rate for this simulation run (applicable year 6+)
        remortgage_rate = rng.normal(remort_rate_mean, remort_rate_std_dev)

        # Apply constraints
        prop_app_rates = np.maximum(-0.99, prop_app_rates)
        alt_inv_rates = np.maximum(-0.99, alt_inv_rates)
        sc_inf_rates = np.maximum(-0.1, sc_inf_rates)
        remortgage_rate = max(0.001, remortgage_rate) # Ensure mortgage rate > 0

        # --- Run simulation for CASH ---
        sim_config_cash = copy.deepcopy(base_config)
        # Update config with generated rates for this run
        sim_config_cash.update({
            'property_appreciation_rates_annual': prop_app_rates.tolist(),
            'service_charge_inflation_rates_annual': sc_inf_rates.tolist(),
            # Dummy rates for cash scenario
            'alternative_investment_return_rates_annual': [0.0] * holding_period,
            'remortgage_rate_annual_year_6_onwards': 0.0 # Not used
        })
        cash_res = simulate_investment_annual_rates(**sim_config_cash, is_cash_purchase=True)
        cash_results_list.append(cash_res)

        # --- Run simulation for MORTGAGE ---
        sim_config_mort = copy.deepcopy(base_config)
        # Update config with generated rates for this run
        sim_config_mort.update({
            'property_appreciation_rates_annual': prop_app_rates.tolist(),
            'service_charge_inflation_rates_annual': sc_inf_rates.tolist(),
            'alternative_investment_return_rates_annual': alt_inv_rates.tolist(),
            # Pass the randomly generated remortgage rate
            'remortgage_rate_annual_year_6_onwards': remortgage_rate,
             # Ensure the initial fixed rate is passed correctly (it's already in base_config)
             # 'initial_mortgage_interest_rate_annual': initial_mortgage_rate # Already in sim_config_mort
        })
        mort_res = simulate_investment_annual_rates(**sim_config_mort, is_cash_purchase=False)
        mortgage_results_list.append(mort_res)

        # Optional: Progress indicator
        # if (i + 1) % (num_simulations // 10) == 0:
        #      print(f"  Completed simulation {i+1}/{num_simulations}")

    return cash_results_list, mortgage_results_list


# --- Optional: Main block for testing ---
if __name__ == "__main__":
    print("Running basic module test with remortgage logic...")
    test_config = get_base_config()
    n_years = test_config['holding_period_years']

    # Test Monte Carlo runner with small number of runs
    print("\nTesting run_monte_carlo_simulation (10 runs)...")
    # Add remortgage rate assumptions for the test run
    test_remort_mean = test_config['initial_mortgage_interest_rate_annual'] # Assume remort mean = initial for test
    test_remort_std_dev = 0.01

    mc_cash, mc_mort = run_monte_carlo_simulation(
        base_config=test_config,
        num_simulations=10,
        prop_app_mean=test_config['annual_property_appreciation_rate'],
        alt_inv_mean=test_config['alternative_investment_return_rate_annual'],
        sc_inf_mean=test_config['service_charge_inflation_rate'],
        remort_rate_mean=test_remort_mean,
        remort_rate_std_dev=test_remort_std_dev
    )
    print(f"Monte Carlo test completed.")
    print(f"  Generated {len(mc_cash)} cash results.")
    print(f"  Example Cash Final Equity: £{mc_cash[0]['final_equity_in_property']:,.0f}")
    print(f"  Generated {len(mc_mort)} mortgage results.")
    print(f"  Example Mortgage Final Equity: £{mc_mort[0]['final_equity_in_property']:,.0f}")
    print(f"  Example Mortgage Alt Inv Value: £{mc_mort[0]['value_of_alternative_investments_at_end']:,.0f}")
    print(f"  Example Mortgage Interest Paid: £{mc_mort[0]['total_mortgage_interest_paid']:,.0f}")
    print(f"  Example Mortgage Remaining Balance: £{mc_mort[0]['remaining_mortgage_balance_at_end']:,.0f}")
    print("\nModule basic tests finished.")