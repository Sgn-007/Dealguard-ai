# DealGuard Streamlit deployment — based on the uploaded Colab app.py
# The validated financial and customer analytics functions are preserved.
import os
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="DealGuard | Deal Intelligence", page_icon="📊", layout="wide")
ROOT = Path(__file__).resolve().parent
@st.cache_data
def load_data():
    names = {"accounts":"Accounts.csv", "retail":"online_retail.csv", "product_catalogue":"products.csv", "opportunities":"sales_opportunities.csv", "discount_history":"discount_history.csv", "sales_reps":"sales_representatives.csv"}
    data = {key: pd.read_csv(ROOT / name) for key, name in names.items()}
    import json
    with (ROOT / "business_policy.json").open(encoding="utf-8") as f:
        data["policy"] = json.load(f)
    return data

try:
    data = load_data()
except Exception as exc:
    st.error(f"Cannot load DealGuard data files: {exc}")
    st.stop()
accounts = data["accounts"]
retail = data["retail"]
product_catalogue = data["product_catalogue"]
opportunities = data["opportunities"]
discount_history = data["discount_history"]
sales_reps = data["sales_reps"]
policy = data["policy"]
MIN_MARGIN = Decimal(str(policy["minimum_contribution_margin_pct"]))
APPROVAL_THRESHOLD = Decimal(str(policy["discount_approval_threshold_pct"]))

# Historical analytics: use the corrected day-month-year parsing from the notebook.
retail_analysis = retail.copy()
retail_analysis["InvoiceDate"] = pd.to_datetime(retail_analysis["InvoiceDate"].astype(str).str.strip(), format="%d-%m-%Y %H:%M", errors="raise")
retail_analysis["Transaction_Value"] = retail_analysis["Quantity"] * retail_analysis["Price"]
customer_profiles = retail_analysis.groupby("Customer ID").agg(Total_Transactions=("Invoice", "nunique"), Total_Items_Purchased=("Quantity", "sum"), Total_Revenue=("Transaction_Value", "sum"), Last_Purchase_Date=("InvoiceDate", "max"), Unique_Products=("StockCode", "nunique")).reset_index()
customer_profiles = customer_profiles.merge(accounts[["Customer ID", "Account Name", "Customer Type"]], on="Customer ID", how="left", validate="one_to_one")
customer_profiles["Average_Invoice_Value"] = customer_profiles["Total_Revenue"] / customer_profiles["Total_Transactions"]
segments = customer_profiles.copy()
revenue_median = segments["Total_Revenue"].median()
def D(value):
    return Decimal(str(value))

def money(value):
    return value.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP
    )

def percent(value):
    return value.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP
    )

def analyze_deal(
    quantity,
    unit_price,
    unit_cost,
    discount_pct
):
    quantity = D(quantity)
    unit_price = D(unit_price)
    unit_cost = D(unit_cost)
    discount_pct = D(discount_pct)

    # Validate inputs
    if quantity <= 0:
        raise ValueError("Quantity must be positive.")

    if unit_price <= 0:
        raise ValueError("Unit price must be positive.")

    if unit_cost < 0:
        raise ValueError("Unit cost cannot be negative.")

    if not all(
        x.is_finite()
        for x in [
            quantity, unit_price,
            unit_cost, discount_pct
        ]
    ):
        raise ValueError("Inputs must be finite numbers.")

    if not (D(0) <= discount_pct <= D(35)):
        raise ValueError(
            "Discount must be between 0% and 35%."
        )

    # Financial calculations
    list_value = quantity * unit_price

    selling_value = list_value * (
        D(1) - discount_pct / D(100)
    )

    total_cost = quantity * unit_cost

    contribution = selling_value - total_cost

    margin_pct = (
        contribution / selling_value
    ) * D(100)

    # Maximum discount that preserves
    # the minimum contribution margin
    target_margin = MIN_MARGIN / D(100)

    max_safe_discount = D(100) * (
        D(1) -
        total_cost / (
            list_value * (D(1) - target_margin)
        )
    )

    margin_breach = margin_pct < MIN_MARGIN

    # Approval is required only ABOVE 15%
    approval_required = (
        discount_pct > APPROVAL_THRESHOLD
    )

    human_review_required = (
        margin_breach or approval_required
    )

    if margin_breach:
        status = "MARGIN BREACH - HUMAN REVIEW"
    elif approval_required:
        status = "MANAGER APPROVAL REQUIRED"
    else:
        status = "WITHIN POLICY"

    # A negative maximum means even a
    # zero-discount deal cannot meet policy
    feasible_at_zero_discount = (
        max_safe_discount >= D(0)
    )

    recommended_discount = (
        min(
            max(discount_pct, D(0)),
            max_safe_discount,
            D(35)
        )
        if feasible_at_zero_discount
        else None
    )

    return {
        "list_value": money(list_value),
        "selling_value": money(selling_value),
        "total_direct_cost": money(total_cost),
        "contribution": money(contribution),
        "contribution_margin_pct": percent(margin_pct),
        "minimum_margin_pct": MIN_MARGIN,
        "proposed_discount_pct": discount_pct,
        "maximum_margin_safe_discount_pct": (
            percent(max_safe_discount)
        ),
        "recommended_discount_pct": (
            percent(recommended_discount)
            if recommended_discount is not None
            else None
        ),
        "margin_breach": margin_breach,
        "manager_approval_required": approval_required,
        "human_review_required": human_review_required,
        "feasible_at_zero_discount": (
            feasible_at_zero_discount
        ),
        "policy_status": status
    }

def classify_customer(row):
    revenue = row["Total_Revenue"]
    invoices = row["Total_Transactions"]
    products_count = row["Unique_Products"]

    # Illustrative rules, not statistically
    # validated customer-value predictions
    if revenue >= revenue_median and invoices >= 5:
        return "High Observed Value"

    elif invoices >= 5:
        return "Frequent - Lower Observed Value"

    elif products_count >= 3:
        return "Product-Diverse Customer"

    else:
        return "Limited Observed Activity"

segments["Customer_Segment"] = segments.apply(classify_customer, axis=1)
all_customer_segments = accounts.merge(segments[["Customer ID", "Total_Transactions", "Total_Revenue", "Unique_Products", "Customer_Segment"]], on="Customer ID", how="left", validate="one_to_one")
all_customer_segments["Customer_Segment"] = all_customer_segments["Customer_Segment"].fillna("Insufficient Historical Data")
profitability_data = retail_analysis.merge(product_catalogue[["Product_ID", "Estimated_Direct_Cost_Per_Unit"]], left_on="StockCode", right_on="Product_ID", how="left", validate="many_to_one")
if profitability_data["Estimated_Direct_Cost_Per_Unit"].isna().any():
    st.error("Some product cost estimates are missing.")
    st.stop()
profitability_data["Estimated_Total_Cost"] = profitability_data["Quantity"] * profitability_data["Estimated_Direct_Cost_Per_Unit"]
profitability_data["Estimated_Contribution"] = profitability_data["Transaction_Value"] - profitability_data["Estimated_Total_Cost"]
customer_profitability = profitability_data.groupby("Customer ID").agg(Historical_Revenue=("Transaction_Value", "sum"), Estimated_Direct_Cost=("Estimated_Total_Cost", "sum"), Estimated_Contribution=("Estimated_Contribution", "sum")).reset_index()
customer_profitability["Estimated_Margin_Pct"] = 100 * customer_profitability["Estimated_Contribution"] / customer_profitability["Historical_Revenue"]
customer_profitability = customer_profitability.merge(all_customer_segments[["Customer ID", "Account Name", "Customer_Segment"]], on="Customer ID", how="left", validate="one_to_one")
customer_profitability[["Historical_Revenue", "Estimated_Direct_Cost", "Estimated_Contribution", "Estimated_Margin_Pct"]] = customer_profitability[["Historical_Revenue", "Estimated_Direct_Cost", "Estimated_Contribution", "Estimated_Margin_Pct"]].round(2)
tool_call_log = []
def get_customer_profile(customer_id: int) -> dict:
    """
    Retrieve customer information, purchasing segment,
    and estimated historical profitability.
    """
    customer_id = int(customer_id)

    account_match = accounts.loc[
        accounts["Customer ID"] == customer_id
    ]

    if account_match.empty:
        return {
            "error": f"Customer {customer_id} not found."
        }

    account = account_match.iloc[0]

    result = {
        "customer_id": customer_id,
        "account_name": str(account["Account Name"]),
        "customer_type": str(account["Customer Type"]),
        "country": str(account["Country"]),
        "customer_segment": "Insufficient Historical Data",
        "historical_transactions_available": False
    }

    segment_match = all_customer_segments.loc[
        all_customer_segments["Customer ID"] == customer_id
    ]

    if not segment_match.empty:
        result["customer_segment"] = str(
            segment_match.iloc[0]["Customer_Segment"]
        )

    profit_match = customer_profitability.loc[
        customer_profitability["Customer ID"] == customer_id
    ]

    if not profit_match.empty:
        p = profit_match.iloc[0]

        result.update({
            "historical_transactions_available": True,
            "historical_revenue": float(
                p["Historical_Revenue"]
            ),
            "estimated_direct_cost": float(
                p["Estimated_Direct_Cost"]
            ),
            "estimated_contribution": float(
                p["Estimated_Contribution"]
            ),
            "estimated_margin_pct": float(
                p["Estimated_Margin_Pct"]
            ),
            "profitability_note": (
                "Historical revenue uses the supplied "
                "retail sample; direct costs are simulated."
            )
        })

    return result

def analyze_opportunity(opportunity_id: str) -> dict:
    """
    Retrieve a sales opportunity and analyse its
    profitability using the financial engine.
    """

    opportunity_id = str(opportunity_id).strip()

    match = opportunities.loc[
        opportunities["Opportunity_ID"].astype(str)
        == opportunity_id
    ]

    if match.empty:
        return {
            "error": f"Opportunity {opportunity_id} not found."
        }

    row = match.iloc[0]

    analysis = analyze_deal(
        quantity=row["Quantity"],
        unit_price=row["Reference_Unit_Price"],
        unit_cost=row["Estimated_Direct_Cost_Per_Unit"],
        discount_pct=row["Proposed_Discount_Pct"]
    )

    return {
        "opportunity_id": str(row["Opportunity_ID"]),
        "customer_id": int(row["Customer_ID"]),
        "account_name": str(row["Account_Name"]),
        "product_name": str(row["Product_Name"]),
        "quantity": int(row["Quantity"]),
        "sales_representative": str(row["Account_Manager"]),
        "opportunity_stage": str(row["Opportunity_Stage"]),
        "financial_analysis": {
            key: (
                float(value)
                if hasattr(value, "as_tuple")
                else value
            )
            for key, value in analysis.items()
        },
        "data_note": (
            "Synthetic opportunity. Product direct costs "
            "are simulated estimates."
        )
    }

def compare_discount_scenarios(
    opportunity_id: str
) -> dict:
    """
    Compare discount alternatives using the validated
    DealGuard financial calculation engine.
    """

    opportunity_id = str(opportunity_id).strip()

    match = opportunities.loc[
        opportunities["Opportunity_ID"].astype(str)
        == opportunity_id
    ]

    if match.empty:
        return {
            "error": f"Opportunity {opportunity_id} not found."
        }

    row = match.iloc[0]

    proposed_discount = float(
        row["Proposed_Discount_Pct"]
    )

    # Include the proposed discount alongside
    # standard alternatives.
    discount_levels = sorted(
        set([0, 5, 8, 12, 16, 20, 25, proposed_discount])
    )

    scenarios = []

    for discount in discount_levels:
        result = analyze_deal(
            quantity=row["Quantity"],
            unit_price=row["Reference_Unit_Price"],
            unit_cost=row["Estimated_Direct_Cost_Per_Unit"],
            discount_pct=discount
        )

        scenarios.append({
            "discount_pct": float(discount),
            "selling_value": float(
                result["selling_value"]
            ),
            "estimated_contribution": float(
                result["contribution"]
            ),
            "estimated_margin_pct": float(
                result["contribution_margin_pct"]
            ),
            "margin_breach": bool(
                result["margin_breach"]
            ),
            "manager_approval_required": bool(
                result["manager_approval_required"]
            ),
            "human_review_required": bool(
                result["human_review_required"]
            ),
            "policy_status": result["policy_status"]
        })

    # Calculate the maximum safe discount independently
    # using full-precision Decimal arithmetic.
    quantity = Decimal(str(row["Quantity"]))
    price = Decimal(str(row["Reference_Unit_Price"]))
    cost = Decimal(str(row["Estimated_Direct_Cost_Per_Unit"]))

    list_value = quantity * price
    total_cost = quantity * cost
    minimum_margin = Decimal("0.20")

    safe_discount = (
        Decimal("100")
        * (
            Decimal("1")
            - total_cost
            / (list_value * (Decimal("1") - minimum_margin))
        )
    )

    # Round DOWN, never up, to avoid exceeding
    # the actual margin-safe discount.
    safe_discount = max(
        Decimal("0"),
        safe_discount.quantize(
            Decimal("0.01"),
            rounding=ROUND_DOWN
        )
    )

    zero_discount_result = analyze_deal(
        quantity=row["Quantity"],
        unit_price=row["Reference_Unit_Price"],
        unit_cost=row["Estimated_Direct_Cost_Per_Unit"],
        discount_pct=0
    )

    feasible_at_zero = not zero_discount_result["margin_breach"]

    if not feasible_at_zero:
        recommended_discount = None
        recommendation_note = (
            "Even a zero discount fails the minimum margin. "
            "Escalate for human review; no margin-compliant "
            "discount is available at the current cost and price."
        )
    else:
        recommended_discount = float(safe_discount)
        recommendation_note = (
            "Maximum discount that preserves the estimated "
            "minimum contribution margin. This is not "
            "automatic deal approval."
        )

    return {
        "opportunity_id": opportunity_id,
        "account_name": str(row["Account_Name"]),
        "currency": "GBP (assumed)",
        "proposed_discount_pct": proposed_discount,
        "minimum_margin_pct": 20.0,
        "manager_approval_threshold_pct": 15.0,
        "recommended_maximum_margin_safe_discount_pct":
            recommended_discount,
        "recommendation_note": recommendation_note,
        "scenarios": scenarios,
        "data_note": (
            "Synthetic opportunity; estimated direct costs. "
            "Scenario results are calculations, not "
            "observed customer responses or sales outcomes."
        )
    }

def dealguard_verified_opportunity_tool(
    opportunity_id: str
) -> dict:
    """
    Retrieve an opportunity and its verified customer
    profile using the customer ID from the dataset.
    """

    print(
        f"TOOL CALLED: verified_opportunity({opportunity_id})"
    )

    tool_call_log.append({
        "tool": "verified_opportunity",
        "opportunity_id": opportunity_id
    })

    # Retrieve validated opportunity analysis
    opportunity_result = analyze_opportunity(
        opportunity_id
    )

    if "error" in opportunity_result:
        return opportunity_result

    # Use the authoritative customer ID from Python
    customer_id = int(
        opportunity_result["customer_id"]
    )

    # Retrieve the corresponding customer profile
    customer_result = get_customer_profile(
        customer_id
    )

    # Verify customer identity before returning data
    if "error" in customer_result:
        return {
            "error": "Customer profile could not be verified.",
            "opportunity_id": opportunity_id,
            "customer_id": customer_id
        }

    if (
        customer_result["account_name"]
        != opportunity_result["account_name"]
    ):
        return {
            "error": "Customer account name mismatch.",
            "opportunity_id": opportunity_id,
            "customer_id": customer_id
        }

    return {
        "opportunity": opportunity_result,
        "verified_customer_profile": customer_result,
        "currency": "GBP (assumed)",
        "verification_status": "CUSTOMER ID VERIFIED",
        "data_note": (
            "Opportunity data is synthetic. "
            "Historical customer data is sample-based. "
            "Direct product costs are simulated estimates."
        )
    }

original_compare_discount_scenarios = compare_discount_scenarios
def compare_discount_scenarios_verified(opportunity_id: str):
    """
    Extend the existing scenario comparison with
    a Python-verified maximum-safe-discount scenario.
    """

    result = original_compare_discount_scenarios(
        opportunity_id
    )

    if "error" in result:
        return result

    opportunity = opportunities.loc[
        opportunities["Opportunity_ID"] == opportunity_id
    ].iloc[0]

    safe_discount = result[
        "recommended_maximum_margin_safe_discount_pct"
    ]

    if safe_discount is None:
        result["verified_recommended_scenario"] = None
        return result

    safe_analysis = analyze_deal(
        quantity=opportunity["Quantity"],
        unit_price=opportunity["Reference_Unit_Price"],
        unit_cost=opportunity["Estimated_Direct_Cost_Per_Unit"],
        discount_pct=safe_discount
    )

    # Convert Decimal values for clean tool output.
    safe_scenario = {
        key: (
            float(value)
            if isinstance(value, Decimal)
            else value
        )
        for key, value in safe_analysis.items()
    }

    safe_scenario["scenario_type"] = (
        "RECOMMENDED MAXIMUM MARGIN-SAFE DISCOUNT"
    )

    # Keep the original scenarios and add a separate
    # verified recommendation for the AI to reference.
    result["verified_recommended_scenario"] = safe_scenario

    return result

def dealguard_scenario_tool(opportunity_id: str) -> dict:
    """
    Gemini-facing scenario tool with a verified
    maximum-safe-discount financial calculation.
    """

    print(
        f"TOOL CALLED: compare_discount_scenarios({opportunity_id})"
    )

    tool_call_log.append({
        "tool": "compare_discount_scenarios",
        "opportunity_id": opportunity_id
    })

    return compare_discount_scenarios_verified(
        opportunity_id
    )

DEALGUARD_DECISION_RULES = """
DealGuard is an advisory decision-support agent.

Never claim to approve, reject, submit, execute,
or change an opportunity in Salesforce.

When a proposed discount breaches policy:
- Flag the discount as non-compliant.
- Explain each applicable policy violation.
- Recommend a compliant alternative where feasible.
- Recommend referral to an authorised human reviewer
  when a policy exception is requested.
- Do not say "reject the deal" or "reject the discount"
  as an action taken by DealGuard.

When a proposed discount is within policy:
- State that no special policy review is triggered.
- Recommend continuation through the normal
  human-controlled sales process.
- Do not describe policy compliance as deal approval.

Financial values and customer identities must come
from verified Python tool results.
"""


def run_dealguard_final(opportunity_id, user_question):
    """
    Final DealGuard advisory agent.

    Uses verified Python analytical tools and
    prohibits autonomous deal decisions.
    """

    opportunity_id = str(opportunity_id).strip().upper()

    valid_ids = set(
        opportunities["Opportunity_ID"].astype(str)
    )

    if opportunity_id not in valid_ids:
        print(f"Opportunity {opportunity_id} not found.")
        return None

    tool_call_log.clear()

    prompt = f"""
    Analyse opportunity {opportunity_id}.

    User question:
    {user_question}

    Call BOTH tools:
    1. dealguard_verified_opportunity_tool
    2. dealguard_scenario_tool

    Provide a structured advisory report covering:
    1. Verified opportunity and customer details
    2. Proposed discount and estimated deal profitability
    3. Historical customer analytics, where available
    4. Discount scenario comparison
    5. Margin and manager-approval policy checks
    6. Recommended next action

    Do not calculate new financial scenario values yourself.
    Use only figures returned by Python tools.

    Distinguish margin breaches from manager-approval
    requirements.

    The final recommendation must remain advisory.
    """

    system_instructions = (
        "You are DealGuard, an AI-powered deal profitability "
        "and discount intelligence agent designed for a "
        "Salesforce Sales Cloud workflow. "
        "Use the supplied Python tools for customer identities, "
        "historical customer analytics, policy checks, and "
        "financial calculations. "
        "Never invent customer IDs, transactions, or financial "
        "scenario values. "
        "Use GBP as the assumed currency. "
        "Opportunity records are synthetic. "
        "Direct costs are simulated estimates. "
        "Historical customer analytics use a limited sample. "
        "If history is unavailable, state "
        "'Insufficient Historical Data'. "
        "Do not claim Salesforce records were updated. "
        "Do not predict customer acceptance. "
        + DEALGUARD_DECISION_RULES
    )

    try:
        response = gemini_client.models.generate_content(
            model=DEALGUARD_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[
                    dealguard_verified_opportunity_tool,
                    dealguard_scenario_tool
                ],
                system_instruction=system_instructions
            )
        )

        print("\n" + "=" * 60)
        print("DEALGUARD - FINAL ADVISORY REPORT")
        print("=" * 60)
        print(response.text)

        print("\n" + "=" * 60)
        print("TOOL EXECUTION LOG")
        print("=" * 60)

        for call in tool_call_log:
            print(call)

        return response

    except Exception as e:
        print("DealGuard encountered an error.")
        print("Error type:", type(e).__name__)
        print("Error details:", str(e)[:1000])
        return None


# Streamlit stores secrets outside GitHub. No API calls occur until Analyze is clicked.
try:
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
except Exception:
    api_key = os.environ.get("GEMINI_API_KEY", "")
# Override in Streamlit Secrets if your Gemini account uses a different supported model.
try:
    DEALGUARD_MODEL = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
except Exception:
    DEALGUARD_MODEL = "gemini-2.5-flash"
gemini_client = genai.Client(api_key=api_key) if api_key else None

st.title("DealGuard")
st.caption("AI-Powered Deal Profitability & Discount Intelligence | Salesforce Sales Cloud concept prototype")
st.info("Academic demonstration: synthetic opportunities, simulated direct costs and GBP assumed. No live Salesforce connection or automatic approvals.")

ids = opportunities["Opportunity_ID"].astype(str).tolist()
selected = st.selectbox("Sales opportunity", ids, index=ids.index("OPP00001") if "OPP00001" in ids else 0)
question = st.text_area("Question for DealGuard", "Evaluate this opportunity's proposed discount, margin and customer history. Compare alternatives and recommend the next human-controlled action.")

opp = analyze_opportunity(selected)
financial = opp["financial_analysis"]
scenario = compare_discount_scenarios_verified(selected)
customer = get_customer_profile(opp["customer_id"])

st.subheader(f"{selected} · {opp['account_name']}")
st.caption(f"Product: {opp['product_name']} | Quantity: {opp['quantity']} | Sales representative: {opp['sales_representative']} | Stage: {opp['opportunity_stage']}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Proposed discount", f"{float(financial['proposed_discount_pct']):.2f}%")
c2.metric("Estimated contribution margin", f"{float(financial['contribution_margin_pct']):.2f}%")
c3.metric("Minimum margin", f"{float(MIN_MARGIN):.0f}%")
safe = scenario["recommended_maximum_margin_safe_discount_pct"]
c4.metric("Maximum margin-safe discount", f"{safe:.2f}%" if safe is not None else "Not feasible")

if financial["margin_breach"]:
    st.error("Contribution margin below policy minimum — human review required.")
else:
    st.success("Contribution margin meets the minimum policy threshold.")
if financial["manager_approval_required"]:
    st.warning("Proposed discount exceeds the manager-approval threshold — approval required.")
else:
    st.info("Manager approval not triggered by the discount threshold.")

left,right = st.columns(2)
with left:
    st.subheader("Deal financials (GBP assumed)")
    st.metric("Estimated selling value", f"£{float(financial['selling_value']):,.2f}")
    st.metric("Estimated contribution", f"£{float(financial['contribution']):,.2f}")
    st.write("**Policy status:**", financial["policy_status"])
with right:
    st.subheader("Customer analytics")
    st.write("**Customer:**", customer["account_name"])
    st.write("**Segment:**", customer["customer_segment"])
    if customer["historical_transactions_available"]:
        st.write("**Historical revenue:**", f"£{customer['historical_revenue']:,.2f}")
        st.write("**Estimated historical contribution:**", f"£{customer['estimated_contribution']:,.2f}")
    else:
        st.write("Insufficient Historical Data")

st.subheader("Discount scenario comparison")
scenarios_df = pd.DataFrame(scenario["scenarios"])
st.dataframe(scenarios_df, use_container_width=True, hide_index=True)
st.line_chart(scenarios_df.set_index("discount_pct")["estimated_margin_pct"], x_label="Discount (%)", y_label="Estimated contribution margin (%)")
if scenario["verified_recommended_scenario"] is not None:
    st.caption("Maximum margin-safe discount is a financial boundary, not deal approval. Discounts above 15% independently require manager approval.")
else:
    st.warning("No discount can meet the minimum margin at current price and estimated cost. Escalate for human review.")

st.subheader("AI advisory report")
if st.button("Generate AI recommendation", type="primary"):
    if not api_key:
        st.error("Add GEMINI_API_KEY in Streamlit Community Cloud → App settings → Secrets.")
    else:
        with st.spinner("Gemini is reviewing verified DealGuard calculations..."):
            response = run_dealguard_final(selected, question)
        if response is not None and response.text:
            st.markdown(response.text)
            with st.expander("Verified tool execution log"):
                st.json(tool_call_log)
        else:
            st.error("Gemini did not return an advisory report. Check app logs, Gemini API access and GEMINI_MODEL secret.")
