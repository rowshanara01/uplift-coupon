"""Coupon decision API: kake coupon dibo, kake dibo na"""
import json

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

import uplift  # noqa: F401  (joblib model load er jonno dorkar)

B = joblib.load("model/uplift_model.joblib")
MODEL, FEATURES, COST, MARGIN_RATE = B["model"], B["features"], B["cost"], B["margin_rate"]
app = FastAPI(title="Coupon Uplift API")


class Customer(BaseModel):
    age: float = Field(ge=16, le=110)
    tenure_months: float = Field(ge=0, le=600)
    past_purchases: int = Field(ge=0, le=1000)
    avg_basket: float = Field(gt=0, le=1_000_000)
    days_since_last_purchase: float = Field(ge=0, le=3650)
    app_user: int = Field(ge=0, le=1)
    coupon_use_rate: float = Field(ge=0, le=1)


def decide(customers):
    X = np.array([[getattr(c, f) for f in FEATURES] for c in customers], float)
    up, base = MODEL.uplift(X), MODEL.baseline(X)
    out = []
    for c, u, b in zip(customers, up, base):
        profit = float(u * MARGIN_RATE * c.avg_basket - COST)
        if profit > 0:
            seg = "persuadable"          # coupon dile ALADA kore kinbe
        elif u < -0.02:
            seg = "sleeping_dog"         # coupon dile ulta kharap hobe
        elif b >= 0.5:
            seg = "sure_thing"           # coupon chhara-i kinbe
        else:
            seg = "lost_cause"           # coupon dileo kinbe na
        out.append({"uplift": round(float(u), 4), "baseline_purchase_prob": round(float(b), 4),
                    "expected_profit_if_coupon": round(profit, 2),
                    "action": "send_coupon" if profit > 0 else "do_not_send", "segment": seg})
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model": B["name"]}


@app.get("/model-info")
def model_info():
    return json.load(open("model/metrics.json"))


@app.post("/score")
def score(c: Customer):
    return decide([c])[0]


@app.post("/score-batch")
def score_batch(cs: list[Customer]):
    return {"count": len(cs), "results": decide(cs)}
