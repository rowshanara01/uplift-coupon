import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
PERSUADABLE = {"age": 30, "tenure_months": 10, "past_purchases": 2, "avg_basket": 600,
               "days_since_last_purchase": 70, "app_user": 0, "coupon_use_rate": 0.8}
LOYAL = {"age": 45, "tenure_months": 60, "past_purchases": 10, "avg_basket": 600,
         "days_since_last_purchase": 3, "app_user": 1, "coupon_use_rate": 0.03}


def test_persuadable_gets_coupon_loyal_does_not():
    p = client.post("/score", json=PERSUADABLE).json()
    l = client.post("/score", json=LOYAL).json()
    assert p["uplift"] > l["uplift"]
    assert p["action"] == "send_coupon" and p["segment"] == "persuadable"
    assert l["action"] == "do_not_send"


def test_batch_and_validation():
    assert client.post("/score-batch", json=[PERSUADABLE, LOYAL]).json()["count"] == 2
    assert client.post("/score", json={**LOYAL, "coupon_use_rate": 3}).status_code == 422


def test_uplift_model_beats_naive_and_random():
    m = json.load(open("model/metrics.json"))
    assert m["models"][m["best"]]["test_qini"] > m["models"]["naive_response"]["test_qini"] > 0
    assert m["models"][m["best"]]["spearman_with_true_uplift"] > 0.7


def test_policy_ranking_by_true_profit():
    p = json.load(open("model/metrics.json"))["policies"]
    best = json.load(open("model/metrics.json"))["best"]
    assert p[f"{best}_top30"]["true_profit"] > p["naive_response_top30"]["true_profit"]
    assert p[f"{best}_top30"]["true_profit"] > p["treat_everyone"]["true_profit"]
    # RCT theke estimate ar asol profit kachakachi (estimator bhoroshajoggo)
    assert abs(p[f"{best}_top30"]["estimated_profit"] - p[f"{best}_top30"]["true_profit"]) < 3
