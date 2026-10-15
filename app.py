import datetime
from typing import Literal, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------- Load models (full pipelines: imputer + encoder + model) ----------
REG_MODEL = joblib.load("/content/ecommerce_sales_regressor.pkl")
CLS_MODEL = joblib.load("/content/ecommerce_return_model_v2.pkl")

BASE_RETURN_RATE = 0.096   # overall return rate in the training data

OPTIONS = {
    "gender": ["Male", "Female"],
    "city": ["Karachi", "Lahore", "Islamabad", "Peshawar",
             "Rawalpindi", "Quetta", "Multan", "Faisalabad"],
    "product_category": ["Electronics", "Fashion", "Home & Kitchen", "Beauty",
                         "Sports", "Books", "Toys", "Grocery"],
    "payment_method": ["Cash on Delivery", "Credit Card", "Debit Card",
                       "Easypaisa/JazzCash", "Bank Transfer"],
    "device_type": ["Mobile", "Desktop", "Tablet"],
}

# Feature lists (must match training notebook exactly)
REG_COLS = ["customer_age", "previous_orders", "is_member", "year", "month", "quarter",
            "day_of_week", "is_weekend", "is_month_end", "age_group", "customer_type",
            "gender", "city", "city_tier", "product_category", "payment_method",
            "device_type", "season"]
CLS_COLS = ["low_rating", "discount_pct", "delivery_days", "is_cod",
            "previous_orders", "product_category"]


# ---------- Same feature engineering as training ----------
def add_features(d):
    d = d.copy()
    d["order_date"] = pd.to_datetime(d["order_date"])
    d["year"] = d["order_date"].dt.year
    d["month"] = d["order_date"].dt.month
    d["quarter"] = d["order_date"].dt.quarter
    d["day_of_week"] = d["order_date"].dt.dayofweek
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)
    d["is_month_end"] = d["order_date"].dt.is_month_end.astype(int)
    d["season"] = d["month"].map({12: "Winter", 1: "Winter", 2: "Winter",
                                  3: "Spring", 4: "Spring", 5: "Spring",
                                  6: "Summer", 7: "Summer", 8: "Summer",
                                  9: "Autumn", 10: "Autumn", 11: "Autumn"})
    d["age_group"] = pd.cut(d["customer_age"], bins=[0, 24, 34, 44, 54, 100],
                            labels=["18-24", "25-34", "35-44", "45-54", "55+"]).astype(object)
    d["customer_type"] = pd.cut(d["previous_orders"], bins=[-1, 1, 4, 8, 100],
                                labels=["New", "Regular", "Loyal", "VIP"]).astype(object)
    d["is_cod"] = (d["payment_method"] == "Cash on Delivery").astype(int)
    d["is_high_discount"] = (d["discount_pct"] >= 20).astype(int)
    d["fast_delivery"] = (d["delivery_days"] <= 3).astype(int)
    d["city_tier"] = d["city"].map({"Karachi": "Tier1", "Lahore": "Tier1", "Islamabad": "Tier1",
                                    "Rawalpindi": "Tier2", "Faisalabad": "Tier2",
                                    "Multan": "Tier2", "Peshawar": "Tier2", "Quetta": "Tier3"})
    d["low_rating"] = (d["customer_rating"] <= 2).astype(int)
    return d


# ---------- Request / response schemas ----------
class Order(BaseModel):
    order_date: str = Field(..., description="YYYY-MM-DD")
    customer_age: int = Field(..., ge=18, le=80)
    gender: Literal["Male", "Female"]
    city: Literal["Karachi", "Lahore", "Islamabad", "Peshawar",
                  "Rawalpindi", "Quetta", "Multan", "Faisalabad"]
    product_category: Literal["Electronics", "Fashion", "Home & Kitchen", "Beauty",
                              "Sports", "Books", "Toys", "Grocery"]
    unit_price_usd: float = Field(..., gt=0, le=5000)
    quantity: int = Field(..., ge=1, le=20)
    discount_pct: int = Field(..., ge=0, le=90)
    shipping_fee_usd: float = Field(..., ge=0, le=50)
    payment_method: Literal["Cash on Delivery", "Credit Card", "Debit Card",
                            "Easypaisa/JazzCash", "Bank Transfer"]
    device_type: Literal["Mobile", "Desktop", "Tablet"]
    is_member: int = Field(..., ge=0, le=1)
    previous_orders: int = Field(..., ge=0, le=100)
    delivery_days: int = Field(..., ge=1, le=30)
    customer_rating: int = Field(..., ge=1, le=5)


def risk_level(p: float):
    if p < 0.09:
        return "low", "Low risk"
    if p < 0.18:
        return "medium", "Medium risk"
    return "high", "High risk"


def build_factors(o: Order) -> List[dict]:
    f = []
    if o.payment_method == "Cash on Delivery":
        f.append({"type": "bad", "text": "Cash on Delivery orders are easier to refuse or return"})
    else:
        f.append({"type": "good", "text": "Prepaid payment lowers return risk"})
    if o.customer_rating <= 2:
        f.append({"type": "bad", "text": f"Low customer rating ({o.customer_rating}/5) is a strong return signal"})
    elif o.customer_rating >= 4:
        f.append({"type": "good", "text": f"Good customer rating ({o.customer_rating}/5)"})
    if o.delivery_days >= 7:
        f.append({"type": "bad", "text": f"Slow delivery ({o.delivery_days} days) hurts satisfaction"})
    elif o.delivery_days <= 3:
        f.append({"type": "good", "text": "Fast delivery (3 days or less)"})
    if o.discount_pct >= 20:
        f.append({"type": "bad", "text": f"Heavy discount ({o.discount_pct}%) can trigger impulse buying"})
    if o.product_category in ("Fashion", "Electronics"):
        f.append({"type": "bad", "text": f"{o.product_category} has a higher return rate than other categories"})
    if o.previous_orders <= 1 and not o.is_member:
        f.append({"type": "info", "text": "New non-member customer, so there is little history"})
    elif o.previous_orders >= 5 or o.is_member:
        f.append({"type": "good", "text": "Loyal or member customer"})
    return f


def run_prediction(o: Order) -> dict:
    try:
        datetime.date.fromisoformat(o.order_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="order_date must be YYYY-MM-DD")

    row = o.model_dump() if hasattr(o, "model_dump") else o.dict()
    df = add_features(pd.DataFrame([row]))

    p_return = float(CLS_MODEL.predict_proba(df[CLS_COLS])[0, 1])
    sales = float(REG_MODEL.predict(df[REG_COLS])[0])
    level, label = risk_level(p_return)
    cart_total = o.unit_price_usd * o.quantity * (1 - o.discount_pct / 100) + o.shipping_fee_usd

    advice = {
        "low": "Looks safe. Process normally.",
        "medium": "Keep an eye on it. A confirmation message can help.",
        "high": "Call or WhatsApp the customer to confirm the address and order before shipping.",
    }[level]

    return {
        "return_probability": round(p_return, 4),
        "risk_level": level,
        "risk_label": label,
        "base_rate": BASE_RETURN_RATE,
        "expected_order_value": round(max(sales, 0), 2),
        "cart_total": round(cart_total, 2),
        "advice": advice,
        "factors": build_factors(o),
    }


# ---------- App ----------
app = FastAPI(title="E-Commerce AI Dashboard", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/options")
def options():
    return OPTIONS


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(order: Order):
    return run_prediction(order)
