"""buy vs rent calculations"""
import numpy as np
import numpy_financial as npf
import copy
from scipy.stats import truncnorm

# --- BASE CONFIGURATION (extended for buy vs. rent) ---
_BASE_CONFIG_BUY_VS_RENT = {
    # --- Buying Scenario Specific ---
    "property_value_initial": 550000.0, # Example, align with your case studies
    "buying_costs_percentage_other": 0.015,
    "annual_property_appreciation_rate": 0.03, # Mean assumption
    "holding_period_years": 15, # Example
    "service_charge_annual_initial": 6500.0,
    "ground_rent_annual": 200.0,
    "council_tax_annual_initial": 2000.0,
    "maintenance_allowance_annual": 600.0,
    "insurance_annual_homeowner": 300.0, # Building + Contents for owner
    "service_charge_inflation_rate": 0.035, # Mean assumption
    "selling_costs_percentage": 0.02,

    # --- Mortgage Specific Parameters ---
    "mortgage_interest_rate_annual": 0.045, # Mean assumption
    "mortgage_term_years": 25, # Standard mortgage term
    "loan_to_value_ratio": 0.75, # 75% LTV
    "mortgage_fee": 1000.0, # Arrangement fee
    "remortgage_fee": 1000.0, # Fee for remortgaging
    "remortgage_interval_years": 5, # How often to remortgage

    # --- Renting Scenario Specific ---
    "initial_annual_rent": 33000.0, # Example: £2750/month
    "annual_rent_inflation_rate": 0.03, # Mean assumption
    "insurance_annual_renter": 200.0, # Contents insurance for renter
    "initial_rental_deposit": 3000.0, # Example, assumed returned at end for simplicity here

    # --- Alternative Investment Specific (for cash not used in property) ---
    "alternative_investment_return_rate_annual": 0.05, # Mean assumption
    
    # --- Monte Carlo Distribution Assumptions (placeholders, to be overridden by scenario_analysis) ---
    "prop_app_mean": 0.03, "prop_app_std_dev": 0.05,
    "alt_inv_mean": 0.05, "alt_inv_std_dev": 0.08,
    "sc_inf_mean": 0.035, "sc_inf_std_dev": 0.025,
    "rent_inf_mean": 0.03, "rent_inf_std_dev": 0.02,
    "mortgage_rate_mean": 0.045, "mortgage_rate_std_dev": 0.015,
}

def get_base_config_buy_vs_rent():
    """Returns a fresh deep copy of the buy vs. rent base configuration."""
    return copy.deepcopy(_BASE_CONFIG_BUY_VS_RENT)

def calculate_stamp_duty(price):
    # (Using existing function from mortgage.py)
    if price <= 125000: return 0.0
    elif price <= 250000: return (price - 125000) * 0.02
    elif price <= 925000: return 2500.0 + (price - 250000) * 0.05
    elif price <= 1500000: return 2500.0 + 33750.0 + (price - 925000) * 0.10
    else: return 2500.0 + 33750.0 + 57500.0 + (price - 1500000) * 0.12

def generate_truncated_normal(mean, std, lower, upper, size):
    """Generate truncated normal distribution values."""
    a = (lower - mean) / std
    b = (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

def validate_inputs(property_value, interest_rate, term_years):
    """Validate input parameters."""
    if property_value <= 0:
        raise ValueError("Property value must be positive")
    if interest_rate < 0:
        raise ValueError("Interest rate cannot be negative")
    if term_years <= 0:
        raise ValueError("Term years must be positive")

def calculate_mortgage_payment(principal, annual_rate, term_years):
    """Calculate monthly mortgage payment using the standard amortization formula."""
    validate_inputs(principal, annual_rate, term_years)
    monthly_rate = annual_rate / 12
    num_payments = term_years * 12
    monthly_payment = -npf.pmt(monthly_rate, num_payments, principal)
    return monthly_payment

def calculate_remaining_mortgage_balance(principal, annual_rate, term_years, years_elapsed):
    """Calculate remaining mortgage balance after a given number of years."""
    validate_inputs(principal, annual_rate, term_years)
    if years_elapsed < 0:
        raise ValueError("Years elapsed cannot be negative")
    monthly_rate = annual_rate / 12
    num_payments = term_years * 12
    payments_made = years_elapsed * 12
    remaining_balance = npf.fv(monthly_rate, payments_made, 
                              -calculate_mortgage_payment(principal, annual_rate, term_years),
                              principal)
    return max(0, remaining_balance)

def simulate_buy_scenario_annual_rates(
    property_value_initial: float,
    buying_costs_percentage_other: float,
    holding_period_years: int,
    service_charge_annual_initial: float,
    ground_rent_annual: float,
    council_tax_annual_initial: float,
    maintenance_allowance_annual: float,
    insurance_annual_homeowner: float,
    selling_costs_percentage: float,
    property_appreciation_rates_annual: list,
    service_charge_inflation_rates_annual: list,
    cash_available: float,
    use_mortgage: bool = True,
    mortgage_interest_rate_annual: float = None,
    mortgage_term_years: int = None,
    loan_to_value_ratio: float = None,
    mortgage_fee: float = None,
    remortgage_fee: float = None,
    remortgage_interval_years: int = None,
    mortgage_rates_annual: list = None,
    **kwargs
):
    """Simulate buying scenario with option for cash or mortgage purchase."""
    # Validate inputs
    validate_inputs(property_value_initial, mortgage_interest_rate_annual if use_mortgage else 0, holding_period_years)
    
    stamp_duty = calculate_stamp_duty(property_value_initial)
    other_buying_costs = property_value_initial * buying_costs_percentage_other
    total_initial_buying_costs = stamp_duty + other_buying_costs

    if use_mortgage:
        # Calculate mortgage details
        loan_amount = property_value_initial * loan_to_value_ratio
        initial_cash_outlay = (property_value_initial - loan_amount) + total_initial_buying_costs + mortgage_fee
        remaining_cash = cash_available - initial_cash_outlay
        current_mortgage_rate = mortgage_rates_annual[0]
        
        # Calculate initial mortgage payment
        initial_monthly_payment = calculate_mortgage_payment(
            loan_amount,
            current_mortgage_rate,
            mortgage_term_years
        )
        annual_mortgage_payment = initial_monthly_payment * 12
    else:
        # Cash purchase
        initial_cash_outlay = property_value_initial + total_initial_buying_costs
        remaining_cash = cash_available - initial_cash_outlay
        loan_amount = 0
        annual_mortgage_payment = 0

    # Initialize tracking variables
    current_property_value = property_value_initial
    total_ongoing_property_costs_paid = 0.0
    total_mortgage_payments = 0.0
    current_service_charge = service_charge_annual_initial
    current_council_tax = council_tax_annual_initial
    current_mortgage_balance = loan_amount
    value_of_alternative_investments = remaining_cash
    cash_flows = [-initial_cash_outlay]  # Year 0 outlay

    for year in range(1, holding_period_years + 1):
        prop_app_rate = property_appreciation_rates_annual[year-1]
        sc_inf_rate = service_charge_inflation_rates_annual[year-1]
        
        if use_mortgage:
            # Update mortgage rate if remortgaging year
            if year % remortgage_interval_years == 0 and year < holding_period_years:
                current_mortgage_rate = mortgage_rates_annual[year-1]
                total_ongoing_property_costs_paid += remortgage_fee
                
                # Recalculate mortgage payment for new rate
                remaining_term = mortgage_term_years - (year - 1)
                initial_monthly_payment = calculate_mortgage_payment(
                    current_mortgage_balance,
                    current_mortgage_rate,
                    remaining_term
                )
                annual_mortgage_payment = initial_monthly_payment * 12

        # Update property value
        current_property_value *= (1 + prop_app_rate)
        
        # Calculate annual costs
        annual_property_related_costs = (
            current_service_charge +
            ground_rent_annual +
            current_council_tax +
            maintenance_allowance_annual +
            insurance_annual_homeowner
        )
        
        if use_mortgage:
            # Calculate mortgage payment and update balance
            total_mortgage_payments += annual_mortgage_payment
            interest_this_year = current_mortgage_balance * current_mortgage_rate
            principal_paid_this_year = annual_mortgage_payment - interest_this_year
            current_mortgage_balance = max(0, current_mortgage_balance - principal_paid_this_year)
            annual_property_related_costs += annual_mortgage_payment

        total_ongoing_property_costs_paid += annual_property_related_costs
        
        # Track cash flow for this year (negative outflow)
        year_costs = annual_property_related_costs
        if use_mortgage and (year % remortgage_interval_years == 0 and year < holding_period_years):
            year_costs += remortgage_fee
        cash_flows.append(-year_costs)
        
        # Update inflating costs
        current_service_charge *= (1 + sc_inf_rate)
        current_council_tax *= (1 + sc_inf_rate)
        
        # Grow alternative investments if any cash remaining
        if value_of_alternative_investments > 0:
            value_of_alternative_investments *= (1 + kwargs.get('alternative_investment_return_rate_annual', 0.05))

    # Calculate final property equity
    selling_costs = current_property_value * selling_costs_percentage
    net_sale_proceeds_from_property = current_property_value - selling_costs
    final_equity_in_property = net_sale_proceeds_from_property - current_mortgage_balance
    
    # Add net proceeds from property sale as separate cash flow entry
    cash_flows.append(net_sale_proceeds_from_property)
    
    # Calculate total net worth at end
    total_net_worth = final_equity_in_property + value_of_alternative_investments
    
    return {
        "scenario_type": "Buy_Mortgage" if use_mortgage else "Buy_Cash",
        "initial_cash_outlay": initial_cash_outlay,
        "final_property_value": current_property_value,
        "net_proceeds_from_property_sale": net_sale_proceeds_from_property,
        "final_equity_in_property": final_equity_in_property,
        "total_ongoing_property_costs_paid": total_ongoing_property_costs_paid,
        "value_of_alternative_investments_at_end": value_of_alternative_investments,
        "remaining_mortgage_balance": current_mortgage_balance if use_mortgage else 0,
        "cash_flows": cash_flows,
        "total_net_worth": total_net_worth,
    }

def simulate_renting_scenario_annual_rates(
    cash_available_for_investment: float, # This is the key: property_value_initial + total_initial_buying_costs from buy scenario
    initial_annual_rent: float,
    holding_period_years: int,
    insurance_annual_renter: float,
    initial_rental_deposit: float, # Added for completeness
    annual_rent_inflation_rates: list, # array for MC
    alternative_investment_return_rates_annual: list, # array for MC
    # Assuming council tax and renter's specific service charges (if any) are part of rent or negligible.
    # If renter pays council tax separately, that needs to be added.
    **kwargs # Absorb unused config items
):
    value_of_alternative_investments = cash_available_for_investment - initial_rental_deposit 
    # Assuming deposit is set aside and not invested, returned at end.
    current_annual_rent = initial_annual_rent
    total_rent_paid = 0.0
    total_renter_ongoing_costs_paid = 0.0 # For insurance, etc.
    cash_flows = [-initial_rental_deposit]  # Year 0 outlay
    for year in range(1, holding_period_years + 1):
        rent_inf_rate = annual_rent_inflation_rates[year-1]
        alt_inv_rate = alternative_investment_return_rates_annual[year-1]
        # Grow alternative investments
        value_of_alternative_investments *= (1 + alt_inv_rate)
        # Pay rent for the year
        total_rent_paid += current_annual_rent
        # Pay other renter ongoing costs
        total_renter_ongoing_costs_paid += insurance_annual_renter # Assuming insurance is the main one
        # Track cash flow for this year (negative outflow)
        year_costs = current_annual_rent + insurance_annual_renter
        cash_flows.append(-year_costs)
        # Inflate rent for next year
        current_annual_rent *= (1 + rent_inf_rate)
    # At end, add back rental deposit and value of alternative investments to last year's cash flow (inflow)
    cash_flows[-1] += value_of_alternative_investments + initial_rental_deposit
    
    # Calculate total net worth at end
    total_net_worth = value_of_alternative_investments + initial_rental_deposit
    
    return {
        "scenario_type": "Rent",
        "initial_cash_outlay": initial_rental_deposit, # The only cash "tied up" initially not earning alternative returns
        "total_rent_paid": total_rent_paid,
        "total_renter_ongoing_costs_paid": total_renter_ongoing_costs_paid,
        "value_of_alternative_investments_at_end": value_of_alternative_investments,
        "final_equity_in_property": 0, # No property owned
        "cash_available_for_investment": cash_available_for_investment,
        "cash_flows": cash_flows,
        "total_net_worth": total_net_worth,  # Added total net worth
    }

def run_buy_vs_rent_mc_simulation(
    base_config: dict,
    prop_app_mean: float, prop_app_std_dev: float,
    alt_inv_mean: float, alt_inv_std_dev: float,
    sc_inf_mean: float, sc_inf_std_dev: float,
    rent_inf_mean: float, rent_inf_std_dev: float,
    mortgage_rate_mean: float, mortgage_rate_std_dev: float,
    num_simulations: int = 10000,
    cash_available: float = None,
    use_mortgage: bool = True,
):
    """Run Monte Carlo simulation for buy vs rent comparison."""
    buy_results_list = []
    rent_results_list = []
    holding_period = base_config['holding_period_years']
    
    # If cash_available not specified, use property value plus costs as default
    if cash_available is None:
        property_value_initial = base_config['property_value_initial']
        stamp_duty_buy = calculate_stamp_duty(property_value_initial)
        other_buying_costs_buy = property_value_initial * base_config['buying_costs_percentage_other']
        cash_available = property_value_initial + stamp_duty_buy + other_buying_costs_buy

    rng = np.random.default_rng()

    for _ in range(num_simulations):
        # Generate annual rate sequences using truncated normal distributions
        prop_app_rates = generate_truncated_normal(prop_app_mean, prop_app_std_dev, -0.99, 0.99, holding_period)
        alt_inv_rates = generate_truncated_normal(alt_inv_mean, alt_inv_std_dev, -0.99, 0.99, holding_period)
        sc_inf_rates = generate_truncated_normal(sc_inf_mean, sc_inf_std_dev, -0.1, 0.2, holding_period)
        rent_inf_rates = generate_truncated_normal(rent_inf_mean, rent_inf_std_dev, -0.1, 0.2, holding_period)
        mortgage_rates = generate_truncated_normal(mortgage_rate_mean, mortgage_rate_std_dev, 0.01, 0.15, holding_period)

        # --- Run BUY Scenario ---
        sim_config_buy = copy.deepcopy(base_config)
        sim_config_buy.update({
            'property_appreciation_rates_annual': prop_app_rates.tolist(),
            'service_charge_inflation_rates_annual': sc_inf_rates.tolist(),
            'cash_available': cash_available,
            'use_mortgage': use_mortgage,
            'mortgage_rates_annual': mortgage_rates.tolist()
        })
        buy_res = simulate_buy_scenario_annual_rates(**sim_config_buy)
        buy_results_list.append(buy_res)

        # --- Run RENT Scenario ---
        sim_config_rent = copy.deepcopy(base_config)
        sim_config_rent.update({
            'cash_available_for_investment': cash_available,
            'annual_rent_inflation_rates': rent_inf_rates.tolist(),
            'alternative_investment_return_rates_annual': alt_inv_rates.tolist()
        })
        rent_res = simulate_renting_scenario_annual_rates(**sim_config_rent)
        rent_results_list.append(rent_res)
        
    return buy_results_list, rent_results_list
