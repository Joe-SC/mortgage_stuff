import numpy_financial as npf # For mortgage calculations if we get more complex
import copy

# --- BASE CASE ASSUMPTIONS ---
_BASE_CONFIG = {
    "property_value_initial": 575000.0,
    "buying_costs_percentage_other": 0.015, # For legal, survey etc. (1.5%) SDLT is separate.
    "annual_property_appreciation_rate": 0.03, # 3% average appreciation
    "holding_period_years": 10,
    "service_charge_annual_initial": 6500.0,
    "ground_rent_annual": 200.0,
    "council_tax_annual_initial": 2000.0, # Estimate for London Band D/E
    "maintenance_allowance_annual": 600.0, # 0.1% of value, or a fixed sum
    "insurance_annual": 300.0, # Contents insurance
    "service_charge_inflation_rate": 0.03, # Assuming service charge & council tax inflate at 3%
    "selling_costs_percentage": 0.02, # 2% for estate agent, legal on sale

    # Mortgage specific (ignored if is_cash_purchase=True)
    "deposit_percentage": 0.20, # 20%
    "mortgage_interest_rate_annual": 0.045, # 4.5%
    "mortgage_term_years": 25,
    "mortgage_arrangement_fee": 1500.0,

    # Alternative investment specific (ignored if is_cash_purchase=True)
    "alternative_investment_return_rate_annual": 0.05, # 5% net return elsewhere
}

def get_base_config():
    """
    Returns a fresh copy of the base configuration.
    This prevents accidental modifications to the base config and ensures
    each caller gets their own independent copy to modify.
    
    Returns:
        dict: A deep copy of the base configuration
    """
    return copy.deepcopy(_BASE_CONFIG)

# --- Helper Functions ---
def calculate_stamp_duty(price):
    # Implement UK SDLT rules (standard rates, as FTB relief won't apply over £500k)
    # This needs to be accurate for May 2025 rates.
    # For a property of £575,000 (standard rates from April 1, 2025):
    # 0% on first £125,000 = £0
    # 2% on next £125,000 (£125,001 to £250,000) = £2,500
    # 5% on remaining £325,000 (£250,001 to £575,000) = £16,250
    # Total = £18,750
    if price <= 125000:
        return 0.0
    elif price <= 250000:
        return (price - 125000) * 0.02
    elif price <= 925000:
        return (125000 * 0.02) + (price - 250000) * 0.05
    elif price <= 1500000:
        return (125000 * 0.02) + (675000 * 0.05) + (price - 925000) * 0.10
    else:
        return (125000 * 0.02) + (675000 * 0.05) + (575000 * 0.10) + (price - 1500000) * 0.12

def calculate_annual_mortgage_payment(principal, annual_interest_rate, loan_term_years):
    if principal == 0: # handles cash scenario
        return 0, 0, 0
    monthly_rate = annual_interest_rate / 12
    num_payments = loan_term_years * 12
    if monthly_rate > 0:
        pmt = npf.pmt(monthly_rate, num_payments, -principal) # type: ignore
        annual_payment = pmt * 12
    else: # Interest free loan (unlikely for mortgage but good for robustness)
        pmt = principal / num_payments
        annual_payment = pmt*12

    # For simplicity, let's estimate annual interest part.
    # A full amortization schedule would be more accurate for principal vs interest split.
    # Roughly, average annual interest for year 1:
    annual_interest_paid = principal * annual_interest_rate # Simplified for now
    # This simplification is significant. A proper model would track principal repayment.
    # For this iteration, let's assume the user wants to compare interest cost vs. opportunity cost.
    # Or, we calculate the actual amortized payment and then try to split.

    # Let's calculate the actual P&I payment and track remaining balance
    return annual_payment # This is Principal + Interest


# --- Scenario Simulation Function ---
def simulate_investment(
    property_value_initial,
    buying_costs_percentage_other, # Other costs like legal, survey (SDLT calculated separately)
    annual_property_appreciation_rate,
    holding_period_years,
    service_charge_annual_initial,
    ground_rent_annual,
    council_tax_annual_initial,
    maintenance_allowance_annual,
    insurance_annual,
    service_charge_inflation_rate, # Assume council tax also inflates at this rate for simplicity
    selling_costs_percentage,
    # Mortgage specific
    is_cash_purchase,
    deposit_percentage=0.20, # Default if mortgage
    mortgage_interest_rate_annual=0.04, # Default
    mortgage_term_years=25, # Default
    mortgage_arrangement_fee=1000.0, # Default
    # Alternative investment specific
    alternative_investment_return_rate_annual=0.05 # Default
):
    # --- Initial Costs ---
    stamp_duty = calculate_stamp_duty(property_value_initial)
    other_buying_costs = property_value_initial * buying_costs_percentage_other
    total_initial_buying_costs = stamp_duty + other_buying_costs

    current_property_value = property_value_initial
    total_ongoing_costs_paid = 0
    total_mortgage_interest_paid = 0 # More accurately, total mortgage payments made
    value_of_alternative_investments = 0
    initial_cash_outlay = 0
    remaining_mortgage_balance = 0

    if is_cash_purchase:
        initial_cash_outlay = property_value_initial + total_initial_buying_costs
        invested_in_alternative_assets = 0
    else: # Mortgage Purchase
        deposit_amount = property_value_initial * deposit_percentage
        loan_amount = property_value_initial - deposit_amount
        remaining_mortgage_balance = loan_amount # Initial balance
        initial_cash_outlay = deposit_amount + total_initial_buying_costs + mortgage_arrangement_fee
        # Cash that would have been used for property purchase, now free for alternative investments
        cash_freed_up_for_alt_investment = property_value_initial - deposit_amount # This is essentially the loan amount
        value_of_alternative_investments = cash_freed_up_for_alt_investment # Initial investment

    # --- Annual Simulation ---
    current_service_charge = service_charge_annual_initial
    current_council_tax = council_tax_annual_initial

    # Store annual data for detailed review if needed
    property_values_over_time = [current_property_value]
    remaining_mortgage_balances_over_time = [remaining_mortgage_balance]
    alternative_investments_values_over_time = [value_of_alternative_investments]


    for year in range(1, holding_period_years + 1):
        # Property appreciation
        current_property_value *= (1 + annual_property_appreciation_rate)
        property_values_over_time.append(current_property_value)

        # Ongoing costs for the year
        annual_property_related_costs = (
            current_service_charge +
            ground_rent_annual +
            current_council_tax +
            maintenance_allowance_annual +
            insurance_annual
        )
        total_ongoing_costs_paid += annual_property_related_costs

        # Inflate costs for next year
        current_service_charge *= (1 + service_charge_inflation_rate)
        current_council_tax *= (1 + service_charge_inflation_rate) # Assuming same inflation

        if not is_cash_purchase:
            # Mortgage payments
            # For a repayment mortgage, a portion of the payment is principal, reducing the loan
            # This requires an amortization schedule for accuracy.
            # Let's use numpy_financial to calculate annual payment and interest for the year.
            if remaining_mortgage_balance > 0:
                # Calculate interest for *this year* on current balance
                interest_this_year = remaining_mortgage_balance * mortgage_interest_rate_annual
                total_mortgage_interest_paid += interest_this_year # Accumulating just the interest part

                # Calculate full P&I payment for the year
                # For simplicity, let's find the constant annual payment if it were a new loan each year - this is not quite right.
                # A better way: calculate monthly payment at start, then annualize.
                # And track principal reduction.
                if mortgage_term_years - (year -1) > 0 : # Ensure remaining term is positive
                    # npf.pmt is monthly, so annual_interest_rate/12, term in months
                    # npf.fv to find future value (balance) after n periods
                    # For annual tracking of P&I:
                    # This is a simplification for annual interest. A full model tracks monthly.
                    # Let's calculate how much principal is paid off this year
                    # Full annual payment based on current remaining balance and REMAINING term
                    # This is still not a perfect amortization schedule for annual steps but better
                    if mortgage_interest_rate_annual > 0:
                        monthly_pmt_val = npf.pmt(mortgage_interest_rate_annual / 12,
                                               (mortgage_term_years - (year-1)) * 12,
                                               -remaining_mortgage_balance) # type: ignore
                        annual_payment_total = monthly_pmt_val * 12
                        principal_paid_this_year = annual_payment_total - interest_this_year
                        remaining_mortgage_balance -= principal_paid_this_year
                        remaining_mortgage_balance = max(0, remaining_mortgage_balance) # Ensure it doesn't go negative
                    else: # interest free
                        principal_paid_this_year = loan_amount / mortgage_term_years
                        remaining_mortgage_balance -= principal_paid_this_year
                        remaining_mortgage_balance = max(0, remaining_mortgage_balance)


            remaining_mortgage_balances_over_time.append(remaining_mortgage_balance)

            # Alternative investments grow
            value_of_alternative_investments *= (1 + alternative_investment_return_rate_annual)
            alternative_investments_values_over_time.append(value_of_alternative_investments)
        else: # Cash purchase has no mortgage, no alternative investments (all cash is in property)
             remaining_mortgage_balances_over_time.append(0)
             alternative_investments_values_over_time.append(0)


    # --- At End of Holding Period ---
    selling_costs = current_property_value * selling_costs_percentage
    net_sale_proceeds_from_property = current_property_value - selling_costs

    final_equity_from_property = net_sale_proceeds_from_property - remaining_mortgage_balance

    # Total Net Worth derived from this venture
    # For cash purchase: final_equity_from_property is the main asset.
    # For mortgage purchase: final_equity_from_property + value_of_alternative_investments.
    total_final_assets = final_equity_from_property + value_of_alternative_investments

    # Net financial outcome
    # This is (Total Final Assets - Initial Cash Outlay - Total Ongoing Costs - Total Mortgage Interest (if any))
    # However, a more common way to look at "profit" is (Total Final Assets - Initial Cash Outlay)
    # And then compare that against having done nothing or another investment.
    # Let's calculate "Net Financial Position Change"
    # This should represent the change in net worth due to this investment strategy
    # Initial state: Your cash outlay. Final state: Value of assets minus liabilities created
    net_financial_position_change = total_final_assets - initial_cash_outlay - total_ongoing_costs_paid - total_mortgage_interest_paid

    # Return on Initial Cash Outlay
    # Net profit = (total_final_assets - total_ongoing_costs_paid - total_mortgage_interest_paid) - initial_cash_outlay
    # ROI is (Net Profit / Initial Cash Outlay)
    # Let's define Net Profit clearly for ROI:
    # Total benefits = Final property equity + final alternative investments
    # Total costs for those benefits = initial cash outlay + ongoing property costs + total mortgage interest
    # No, ROI should be on the initial cash put in.
    # Final value of what your initial cash outlay turned into, MINUS the costs along the way.
    # So, (final_equity_from_property + value_of_alternative_investments - total_ongoing_costs_paid - total_mortgage_interest_paid)
    # compared to initial_cash_outlay

    # Simpler "Net Gain/Loss" for ROI calculation:
    # This is how much your *initial specific cash investment* grew or shrunk by after all venture-specific costs.
    # For cash: (Final Equity - Ongoing Costs) - Initial Outlay
    # For mortgage: (Final Equity - Ongoing Costs - Mortgage Interest + Alt. Investment Growth) - Initial Outlay
    # No, the alternative investment growth is the return ON THE FREED UP CAPITAL, not part of ROI of the PROPERTY investment directly
    # Let's make this clearer:
    # Scenario 1: Cash. Invest P. Your net outcome is Final Equity - Ongoing Costs. ROI = (Outcome - P)/P
    # Scenario 2: Mortgage. Invest D (deposit). Your net outcome related to property is Final Property Equity - Ongoing - Mortgage Interest. ROI_prop = (Outcome_prop - D)/D
    # Separately, the Freed Up Capital (P-D) grows.

    # For this model, let's define "Total Wealth Generated by Strategy"
    # This is the sum of equity in property and value of alternative investments at the end.
    total_wealth_at_end = final_equity_from_property + value_of_alternative_investments
    # Overall "profit" of the strategy:
    # total_wealth_at_end - initial_cash_outlay - total_ongoing_costs_paid - total_mortgage_interest_paid
    # This needs to be thought through carefully.
    # Let's return key components and calculate high-level metrics outside.

    return {
        "is_cash_purchase": is_cash_purchase,
        "initial_cash_outlay": initial_cash_outlay,
        "final_property_value": current_property_value,
        "selling_costs": selling_costs,
        "net_proceeds_from_property_sale": net_sale_proceeds_from_property,
        "remaining_mortgage_balance_at_end": remaining_mortgage_balance,
        "final_equity_in_property": final_equity_from_property,
        "total_ongoing_property_costs_paid": total_ongoing_costs_paid,
        "total_mortgage_interest_paid": total_mortgage_interest_paid, # This accumulates *only interest*
        "value_of_alternative_investments_at_end": value_of_alternative_investments,
        "property_values_over_time": property_values_over_time,
        "remaining_mortgage_balances_over_time": remaining_mortgage_balances_over_time,
        "alternative_investments_values_over_time": alternative_investments_values_over_time
    }

# --- Main Execution & Sensitivity ---
if __name__ == "__main__":
    # --- BASE CASE ASSUMPTIONS ---
    print("--- Simulating Base Case ---")
    cash_results_base = simulate_investment(**get_base_config(), is_cash_purchase=True)
    mortgage_results_base = simulate_investment(**get_base_config(), is_cash_purchase=False)

    def print_summary(results, scenario_name):
        print(f"\n--- Summary for {scenario_name} ---")
        print(f"Initial Cash Outlay: £{results['initial_cash_outlay']:,.2f}")
        print(f"Final Property Value: £{results['final_property_value']:,.2f}")
        print(f"Net Sale Proceeds from Property (after selling costs): £{results['net_proceeds_from_property_sale']:,.2f}")
        print(f"Remaining Mortgage at End: £{results['remaining_mortgage_balance_at_end']:,.2f}")
        print(f"Final Equity in Property: £{results['final_equity_in_property']:,.2f}")
        print(f"Total Ongoing Property Costs Paid: £{results['total_ongoing_property_costs_paid']:,.2f}")
        if not results['is_cash_purchase']:
            print(f"Total Mortgage Interest Paid: £{results['total_mortgage_interest_paid']:,.2f}")
            print(f"Value of Alternative Investments at End: £{results['value_of_alternative_investments_at_end']:,.2f}")

        # Overall Financial Position:
        # Assets at end = final equity in property + value of alternative investments
        # Total "cost" of achieving these assets = initial cash outlay + ongoing property costs + mortgage interest
        total_assets_at_end = results['final_equity_in_property'] + results['value_of_alternative_investments_at_end']
        total_direct_financial_costs_for_strategy = (
            results['initial_cash_outlay'] +
            results['total_ongoing_property_costs_paid'] +
            (results['total_mortgage_interest_paid'] if not results['is_cash_purchase'] else 0)
        )
        net_position_change = total_assets_at_end - total_direct_financial_costs_for_strategy # This is not quite right.
                                                                            # The initial outlay *is* part of the assets.

        # Let's focus on Net Wealth Change.
        # Initial Net Worth Impact = -initial_cash_outlay
        # Final Net Worth Position = Final Equity + Alt Investments - Ongoing Costs - Mortgage Interest
        # This is more like: (Final Equity + Alt Investments) - (Initial Outlay + Ongoing Property Costs + Mortgage Interest)
        # Or simpler: (Final Equity + Alt Investments) vs Initial Outlay.
        # What's the overall change in wealth *relative to having done nothing with the initial outlay*?
        # Wealth at end of strategy = property_equity + alternative_investment_value
        # Less costs during strategy = ongoing_property_costs + mortgage_interest
        # Net gain = (Wealth at end - costs during strategy) - initial_cash_outlay
        # This is effectively (Final equity + Alt investments) - (Initial Outlay + All running costs including interest)
        
        net_gain_over_initial_outlay_plus_costs = (
            results['final_equity_in_property'] +
            results['value_of_alternative_investments_at_end'] -
            results['initial_cash_outlay'] - # The initial investment itself
            results['total_ongoing_property_costs_paid'] -
            (results['total_mortgage_interest_paid'] if not results['is_cash_purchase'] else 0)
        )
        print(f"Net Gain (Final Assets - Initial Outlay - All Running Costs): £{net_gain_over_initial_outlay_plus_costs:,.2f}")

        # Return on Initial Cash Outlay (Annualized)
        # ( (Ending Value of Initial Cash / Initial Cash Outlay) ^ (1/Years) ) - 1
        # Ending value of initial cash = initial_cash_outlay + net_gain_over_initial_outlay_plus_costs
        ending_value_of_investment_vehicle = results['initial_cash_outlay'] + net_gain_over_initial_outlay_plus_costs
        if results['initial_cash_outlay'] > 0 :
            roi_annualized = ((ending_value_of_investment_vehicle / results['initial_cash_outlay']) ** (1/get_base_config()['holding_period_years'])) - 1
            print(f"Annualized ROI on Initial Cash Outlay: {roi_annualized:.2%}")
        else:
            print("Annualized ROI on Initial Cash Outlay: N/A (no initial outlay)")


    print_summary(cash_results_base, "Cash Purchase (Base Case)")
    print_summary(mortgage_results_base, "Mortgage Purchase (Base Case)")

    # --- Sensitivity Analysis ---
    print("\n\n--- Sensitivity Analysis ---")
    sensitivity_params = {
        "annual_property_appreciation_rate": [0.01, 0.03, 0.05], # Low, Medium, High
        "mortgage_interest_rate_annual": [0.035, 0.045, 0.06], # Low, Medium, High
        "alternative_investment_return_rate_annual": [0.03, 0.05, 0.07] # Low, Medium, High
    }

    for param, values in sensitivity_params.items():
        print(f"\n--- Sensitivity to: {param} ---")
        for value in values:
            print(f"  {param} = {value:.3f}")
            temp_config = get_base_config().copy()
            temp_config[param] = value

            cash_res = simulate_investment(**temp_config, is_cash_purchase=True)
            mort_res = simulate_investment(**temp_config, is_cash_purchase=False)

            # Simplified output for sensitivity
            net_gain_cash = (
                cash_res['final_equity_in_property'] + cash_res['value_of_alternative_investments_at_end'] -
                cash_res['initial_cash_outlay'] - cash_res['total_ongoing_property_costs_paid']
            )
            net_gain_mort = (
                mort_res['final_equity_in_property'] + mort_res['value_of_alternative_investments_at_end'] -
                mort_res['initial_cash_outlay'] - mort_res['total_ongoing_property_costs_paid'] -
                mort_res['total_mortgage_interest_paid']
            )
            print(f"    Cash Net Gain: £{net_gain_cash:,.0f}")
            print(f"    Mortgage Net Gain: £{net_gain_mort:,.0f}")
            
            # Add ROI for sensitivity
            ending_val_cash = cash_res['initial_cash_outlay'] + net_gain_cash
            if cash_res['initial_cash_outlay'] > 0:
                 roi_cash_sens = ((ending_val_cash / cash_res['initial_cash_outlay']) ** (1/get_base_config()['holding_period_years'])) - 1
                 print(f"    Cash Annualized ROI: {roi_cash_sens:.2%}")

            ending_val_mort = mort_res['initial_cash_outlay'] + net_gain_mort
            if mort_res['initial_cash_outlay'] > 0:
                roi_mort_sens = ((ending_val_mort / mort_res['initial_cash_outlay']) ** (1/get_base_config()['holding_period_years'])) - 1
                print(f"    Mortgage Annualized ROI: {roi_mort_sens:.2%}")

    # Example of plotting (requires matplotlib)
    # import matplotlib.pyplot as plt
    # years_axis = list(range(get_base_config()['holding_period_years'] + 1))

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(years_axis, cash_results_base['property_values_over_time'], label="Cash - Property Value")
    # plt.plot(years_axis, mortgage_results_base['property_values_over_time'], label="Mortgage - Property Value", linestyle='--')
    # plt.plot(years_axis, mortgage_results_base['alternative_investments_values_over_time'], label="Mortgage - Alt. Investments", linestyle=':')
    # plt.xlabel("Years")
    # plt.ylabel("Value (£)")
    # plt.title("Asset Growth Over Time")
    # plt.legend()
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # plt.plot(years_axis, mortgage_results_base['remaining_mortgage_balances_over_time'], label="Mortgage - Remaining Balance")
    # plt.xlabel("Years")
    # plt.ylabel("Value (£)")
    # plt.title("Mortgage Balance Over Time")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()