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
        "currency": "Not specified",
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
        "currency": "Not specified",
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
        "Currency is unspecified; report monetary amounts without a currency label or symbol. "
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


# Hosting configuration: secrets are configured in Streamlit Cloud, not GitHub.
try:
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    DEALGUARD_MODEL = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
except Exception:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    DEALGUARD_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
gemini_client = genai.Client(api_key=api_key) if api_key else None


# ---------------------------------------------------------------------
# DealGuard | Opportunity-grounded, prompt-driven decision workspace
# ---------------------------------------------------------------------
import json
import html

st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at 80% 0%, #183355 0%, #0b1425 38%, #080f1d 100%); color: #e9f2ff;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {display: none;}
[data-testid="stSidebarCollapsedControl"] {display: none;}
.block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 4rem;}
h1,h2,h3 {color: #f0f6ff !important; letter-spacing: -.025em;}
p, label, .stMarkdown {color: #d6e3f5;}
[data-testid="stForm"] {background: #111f35; border: 1px solid #294262; border-radius: 20px; padding: 24px;}
[data-testid="stMetric"] {background: #14243b; border: 1px solid #2b4667; border-radius: 16px; padding: 18px;}
[data-testid="stMetricLabel"] {color: #a9c0df;}
[data-testid="stMetricValue"] {color: #ffffff;}
div.stButton > button, div.stFormSubmitButton > button {
  background: linear-gradient(100deg, #10b6a6, #3675f5); color: white;
  border: none; border-radius: 12px; font-weight: 700; min-height: 44px;
}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {color:white; border: 1px solid #9aece1;}
input, textarea {border-radius: 10px !important; color: #17253b !important; background-color: #f1f5fa !important; caret-color: #2563eb !important; -webkit-text-fill-color: #17253b !important;}
input::placeholder, textarea::placeholder {color: #66758a !important; -webkit-text-fill-color: #66758a !important; opacity: 1 !important;}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {color: #17253b !important; background-color: #f1f5fa !important; -webkit-text-fill-color: #17253b !important;}
[data-testid="stTextInput"] [data-baseweb="input"], [data-testid="stTextArea"] [data-baseweb="textarea"] {background-color: #f1f5fa !important;}
input:focus, textarea:focus {outline-color: #3b82f6 !important;}
.hero {background:linear-gradient(120deg,#142e4b,#10233b 56%,#123c48);
 border:1px solid #2c5570; border-radius:24px; padding:34px 36px; margin-bottom:24px;}
.eyebrow {color:#64e4d4; font-size:.8rem; font-weight:800; letter-spacing:.16em;}
.hero-title {font-size:3.1rem; font-weight:850; color:#fff; line-height:1.1; margin:10px 0;}
.hero-sub {color:#b9cce3; max-width:780px; line-height:1.6;}
.pill {display:inline-block; padding:6px 11px; border:1px solid #397f85; border-radius:99px;
 color:#80f0df; font-size:.78rem; margin-top:16px; margin-right:8px;}
.section-kicker {color:#7fd8ee; font-weight:800; letter-spacing:.07em; font-size:.82rem; margin-top:18px;}
.note {background:#12233b; border-left:3px solid #39c8bc; border-radius:10px; padding:15px 18px; color:#c6d9ed;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
 <div class="eyebrow">DEAL INTELLIGENCE WORKSPACE</div>
 <div class="hero-title">DealGuard <span style="color:#58decf">AI</span></div>
 <div class="hero-sub">Enter your proposed deal and ask a question. DealGuard calculates profitability, checks approval rules,
 and consults comparable demonstration opportunities automatically to produce tailored recommendations.</div>
 <span class="pill">Automatic reference retrieval</span>
 <span class="pill">Prompt-driven recommendations</span>
 <span class="pill">Human-controlled decisions</span>
</div>
""", unsafe_allow_html=True)

# Sidebar intentionally omitted: streamlined single-page interface.

def clean_decimal(raw, label, *, positive=False, nonnegative=False):
    if not raw.strip():
        raise ValueError(f"Enter {label}.")
    try:
        value = D(raw.strip().replace(",", ""))
    except Exception:
        raise ValueError(f"{label} must be numeric.")
    if not value.is_finite():
        raise ValueError(f"{label} must be finite.")
    if positive and value <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    if nonnegative and value < 0:
        raise ValueError(f"{label} cannot be negative.")
    return value

def scenario_table(quantity, price, cost, proposed):
    levels = sorted(set([D(0), D(5), D(8), D(12), D(15), D(16), D(20), D(25), proposed]))
    items = []
    for level in levels:
        result = analyze_deal(quantity, price, cost, level)
        items.append({
            "Discount (%)": float(level),
            "Selling value": float(result["selling_value"]),
            "Contribution": float(result["contribution"]),
            "Margin (%)": float(result["contribution_margin_pct"]),
            "Manager approval": "Required" if result["manager_approval_required"] else "Not required",
            "Margin breach": "Yes" if result["margin_breach"] else "No",
            "Policy status": result["policy_status"]
        })
    return items

def safe_discount_floor(quantity, price, cost):
    target = MIN_MARGIN / D(100)
    exact = D(100) * (D(1) - (quantity * cost) / (quantity * price * (D(1) - target)))
    if exact < 0:
        return None
    return min(D(35), exact.quantize(D("0.01"), rounding=ROUND_DOWN))


def find_reference_opportunities(quantity, price, cost, discount, count=5):
    """Retrieve comparable synthetic opportunities without claiming a customer match."""
    source = opportunities.copy()
    for col in ("Quantity", "Reference_Unit_Price", "Estimated_Direct_Cost_Per_Unit", "Proposed_Discount_Pct"):
        source[col] = pd.to_numeric(source[col], errors="coerce")
    source = source.dropna(subset=["Quantity", "Reference_Unit_Price",
                                   "Estimated_Direct_Cost_Per_Unit", "Proposed_Discount_Pct"])
    # Relative differences balance price, cost and quantity; discount is in percentage points.
    source["_distance"] = (
        (source["Quantity"] - float(quantity)).abs() / max(float(quantity), 1)
        + (source["Reference_Unit_Price"] - float(price)).abs() / max(float(price), 1)
        + (source["Estimated_Direct_Cost_Per_Unit"] - float(cost)).abs() / max(float(cost), 1)
        + (source["Proposed_Discount_Pct"] - float(discount)).abs() / 35
    )
    records = []
    for _, row in source.nsmallest(count, "_distance").iterrows():
        result = analyze_deal(
            row["Quantity"], row["Reference_Unit_Price"],
            row["Estimated_Direct_Cost_Per_Unit"], row["Proposed_Discount_Pct"]
        )
        records.append({
            "opportunity_id": str(row["Opportunity_ID"]),
            "product_name": str(row["Product_Name"]),
            "quantity": float(row["Quantity"]),
            "unit_price": float(row["Reference_Unit_Price"]),
            "estimated_unit_cost": float(row["Estimated_Direct_Cost_Per_Unit"]),
            "discount_pct": float(row["Proposed_Discount_Pct"]),
            "estimated_margin_pct": float(result["contribution_margin_pct"]),
            "policy_status": result["policy_status"],
            "similarity_note": "Nearest numeric example; not necessarily the same product or customer."
        })
    return records

def build_verified_context(inputs, financial, rows, safe, references):
    return {
        "user_proposed_deal": {k: float(v) for k, v in inputs.items()},
        "calculated_financials": {
            k: float(v) if isinstance(v, Decimal) else v for k, v in financial.items()
        },
        "maximum_margin_safe_discount_pct": float(safe) if safe is not None else None,
        "verified_discount_scenarios": rows,
        "background_reference_opportunities": references,
        "reference_selection_method": "Five closest opportunities by relative quantity, price, cost and discount differences.",
        "currency_label": "Currency not specified",
        "data_limitations": (
            "User-entered deal has no verified customer identity. Reference opportunities "
            "are synthetic comparators, NOT the user's own opportunity, customer, "
            "historical transactions, or proof of sales outcomes."
        ),
    }

def ask_gemini(context, question):
    def retrieve_verified_deal_analysis() -> dict:
        """Return user-entered deal figures and deterministic profitability/policy calculations."""
        return {
            "user_proposed_deal": context["user_proposed_deal"],
            "calculated_financials": context["calculated_financials"],
            "maximum_margin_safe_discount_pct": context["maximum_margin_safe_discount_pct"],
            "currency_label": context["currency_label"],
            "data_limitations": context["data_limitations"],
        }

    def retrieve_verified_scenarios_and_references() -> dict:
        """Return calculated discount alternatives and comparable synthetic opportunity records."""
        return {
            "discount_scenarios": context["verified_discount_scenarios"],
            "background_reference_opportunities": context["background_reference_opportunities"],
            "reference_selection_method": context["reference_selection_method"],
            "data_limitations": context["data_limitations"],
        }

    instructions = (
        "You are DealGuard, a business-grade AI advisory agent for deal profitability and "
        "discount intelligence. Call BOTH Python tools to ground your response. "
        "Prioritize answering the USER'S PROMPT directly and specifically. "
        "Use the user-entered deal as the ONLY basis for current deal profitability "
        "and policy recommendations. Use the automatically retrieved opportunities "
        "ONLY as illustrative background comparisons. The user did not select an "
        "opportunity or provide a customer ID: never claim that any reference "
        "opportunity or customer belongs to this proposed deal. Do not invent "
        "customer history, predicted sales acceptance or measured market benchmarks. "
        "Do not perform independent financial arithmetic; use verified tool outputs. "
        "The user has not specified a currency. Present monetary amounts as plain numbers without any currency symbol or currency label; do not infer one. "
        "Minimum contribution margin is 20 percent; discounts strictly above "
        "15 percent require manager approval; margin breaches require human review "
        "independently of discount approval. Margin-safe does not mean approved. "
        "Give practical next steps and explain trade-offs relevant to the prompt. "
        "Never claim Salesforce integration, data updates or autonomous deal approval. "
        "Synthetic reference opportunities and estimated direct costs must be disclosed. "
        + DEALGUARD_DECISION_RULES
    )
    response = gemini_client.models.generate_content(
        model=DEALGUARD_MODEL,
        contents=(
            f"User's question: {question}\n"
            "Use both tools for the submitted deal, scenario alternatives and "
            "background reference opportunities. Provide a concise, decision-oriented "
            "answer to the question, with a clear recommendation and policy caveats."
        ),
        config=types.GenerateContentConfig(
            tools=[retrieve_verified_deal_analysis, retrieve_verified_scenarios_and_references],
            system_instruction=instructions,
        ),
    )
    return response.text or ""

st.markdown('<div class="section-kicker">01 / PROPOSED DEAL</div>', unsafe_allow_html=True)
st.markdown("### Enter your deal details")
st.caption("All fields begin blank. DealGuard automatically consults comparable synthetic opportunities in the background; no opportunity ID is needed.")

with st.form("deal_intelligence_form", clear_on_submit=False):
    left, right = st.columns(2)
    with left:
        qty_text = st.text_input("Quantity *", value="", placeholder="e.g. 100")
        cost_text = st.text_input("Estimated direct cost per unit *", value="", placeholder="e.g. 30")
    with right:
        price_text = st.text_input("Unit selling price *", value="", placeholder="e.g. 50")
        discount_text = st.text_input("Proposed discount (%) *", value="", placeholder="e.g. 16")

    st.markdown('<div class="section-kicker">02 / YOUR QUESTION</div>', unsafe_allow_html=True)
    st.markdown("### What would you like DealGuard to recommend?")
    question = st.text_area(
        "Your prompt *", value="", height=120,
        placeholder="For example: Is this discount commercially advisable? Compare alternatives that protect margin and explain the required approvals.",
    )
    submitted = st.form_submit_button(
        "Analyse deal & generate recommendation", type="primary", use_container_width=True
    )

if submitted:
    st.session_state.pop("dg_report", None)
    if not question.strip():
        st.warning("Enter a question so the recommendation can address your prompt.")
    else:
        try:
            q = clean_decimal(qty_text, "quantity", positive=True)
            if q != q.to_integral_value():
                raise ValueError("Quantity must be a whole number.")
            price = clean_decimal(price_text, "unit selling price", positive=True)
            cost = clean_decimal(cost_text, "estimated unit cost", nonnegative=True)
            discount = clean_decimal(discount_text, "discount", nonnegative=True)
            if discount > D(35):
                raise ValueError("Proposed discount must be between 0% and 35%.")
            financial = analyze_deal(q, price, cost, discount)
            safe = safe_discount_floor(q, price, cost)
            scenarios = scenario_table(q, price, cost, discount)
            references = find_reference_opportunities(q, price, cost, discount)
            inputs = {"quantity": q, "unit_price": price, "unit_cost": cost, "discount_pct": discount}
            context = build_verified_context(inputs, financial, scenarios, safe, references)
            st.session_state["dg_report"] = {
                "context": context, "question": question.strip(), "ai_text": None, "ai_error": None
            }
            if gemini_client is not None:
                with st.spinner("Calculating profitability, reviewing reference deals and generating your recommendation..."):
                    try:
                        st.session_state["dg_report"]["ai_text"] = ask_gemini(context, question.strip())
                    except Exception as exc:
                        st.session_state["dg_report"]["ai_error"] = str(exc)
            else:
                st.session_state["dg_report"]["ai_error"] = (
                    "GEMINI_API_KEY is not configured in Streamlit Secrets."
                )
        except (ValueError, ArithmeticError) as exc:
            st.error(str(exc))

report = st.session_state.get("dg_report")
if report:
    context = report["context"]
    f = context["calculated_financials"]
    st.divider()
    st.markdown('<div class="section-kicker">03 / FINANCIAL & POLICY ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown("## Your deal intelligence report")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selling value", f"{f['selling_value']:,.2f}")
    c2.metric("Contribution", f"{f['contribution']:,.2f}")
    c3.metric("Contribution margin", f"{f['contribution_margin_pct']:.2f}%")
    c4.metric(
        "Margin-safe discount ceiling",
        f"{context['maximum_margin_safe_discount_pct']:.2f}%"
        if context["maximum_margin_safe_discount_pct"] is not None else "Not feasible",
    )

    if f["margin_breach"]:
        st.error("Margin breach: below the minimum 20% contribution margin. Human review is required.")
    elif f["manager_approval_required"]:
        st.warning("Margin meets policy, but the discount exceeds 15%. Manager approval is required.")
    else:
        st.success("Within the stated financial policy thresholds. This is not a deal approval.")
    if f["margin_breach"] and f["manager_approval_required"]:
        st.warning("Discount exceeds 15%, so manager approval is also required.")
    if context["maximum_margin_safe_discount_pct"] is None:
        st.warning("No margin-compliant discount is possible at these prices and costs, even at zero discount.")

    st.markdown('<div class="section-kicker">04 / AI RECOMMENDATION</div>', unsafe_allow_html=True)
    st.markdown("### Recommendation based on your prompt")
    st.info(f"**Your question:** {report['question']}")
    if report["ai_text"]:
        st.markdown(report["ai_text"])
    else:
        if report["ai_error"]:
            st.warning("AI recommendation is unavailable: " + report["ai_error"])
        st.markdown("**Verified next action:** " + (
            "Escalate for human review and revise the deal terms."
            if f["margin_breach"] else
            "Request manager approval before proceeding."
            if f["manager_approval_required"] else
            "Continue through the standard human-controlled sales review."
        ))
        if gemini_client is not None and st.button("Retry AI recommendation"):
            with st.spinner("Generating recommendation..."):
                try:
                    st.session_state["dg_report"]["ai_text"] = ask_gemini(context, report["question"])
                    st.session_state["dg_report"]["ai_error"] = None
                    st.rerun()
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")

    with st.expander("See scenario comparisons and background reference opportunities"):
        st.markdown("#### Calculated discount alternatives")
        frame = pd.DataFrame(context["verified_discount_scenarios"])
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.line_chart(frame.set_index("Discount (%)")["Margin (%)"])
        st.markdown("#### Similar opportunities retrieved automatically")
        st.caption(
            "These are numerically similar synthetic examples, not the customer's own "
            "opportunities and not evidence of customer purchasing behavior."
        )
        st.dataframe(
            pd.DataFrame(context["background_reference_opportunities"]),
            use_container_width=True, hide_index=True
        )
    st.caption(
        "Academic advisory prototype · Amounts shown without an assumed currency · "
        "Synthetic opportunity references · No live Salesforce access or autonomous approvals."
    )
