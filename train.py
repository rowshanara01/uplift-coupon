"""Uplift modeling: 4 model compare -> Qini -> coupon policy er profit (estimate vs asol)"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import metrics as M
from uplift import ResponseModel, SLearner, TLearner, XLearner

FEATURES = ["age", "tenure_months", "past_purchases", "avg_basket", "days_since_last_purchase",
            "app_user", "coupon_use_rate"]
COST = 10.0          # coupon er khoroch (taka) prottek customer e
MARGIN_RATE = 0.3    # kenakata hole basket er 30% labh
BUDGET = 0.30        # max 30% customer ke coupon

df = pd.read_csv("data/campaign.csv")
rng = np.random.default_rng(0)
idx = rng.permutation(len(df))
n = len(df)
tr, va, te = df.iloc[idx[:int(.6 * n)]], df.iloc[idx[int(.6 * n):int(.8 * n)]], df.iloc[idx[int(.8 * n):]]


def parts(d):
    return d[FEATURES].values, d.treatment.values, d.purchased.values


Xtr, ttr, ytr = parts(tr)
models = {"naive_response": ResponseModel(), "s_learner": SLearner(), "t_learner": TLearner(),
          "x_learner": XLearner()}
for m in models.values():
    m.fit(Xtr, ttr, ytr)

# ---- Model select: VALIDATION e Qini (test ke ekbar-i chhuyi) ----
Xva, tva, yva = parts(va)
val_qini = {k: M.qini(m.uplift(Xva), tva, yva) for k, m in models.items()}
best = max((k for k in models if k != "naive_response"), key=val_qini.get)
print("validation Qini:", {k: round(v, 4) for k, v in val_qini.items()}, "\nbest uplift model:", best)

# ---- Final TEST ----
Xte, tte, yte = parts(te)
margin = MARGIN_RATE * te.avg_basket.values
p0, tau = te.true_p0.values, te.true_tau.values
k = int(BUDGET * len(te))


def top_budget(score):
    treat = np.zeros(len(score), int)
    treat[np.argsort(-score)[:k]] = 1
    return treat


def value(treat):
    return {"estimated_profit": M.ips_value(treat, tte, yte, margin, COST),
            "true_profit": M.true_value(treat, p0, tau, margin, COST), "treated_share": float(treat.mean())}


report = {"validation_qini": val_qini, "best": best, "models": {}, "policies": {}}
for name, m in models.items():
    s = m.uplift(Xte)
    report["models"][name] = {"test_qini": M.qini(s, tte, yte),
                              "spearman_with_true_uplift": float(spearmanr(s, tau)[0]),
                              "true_uplift_in_top30pct": float(tau[top_budget(s) == 1].mean())}
    report["policies"][f"{name}_top30"] = value(top_budget(s))
best_up = models[best].uplift(Xte)
report["policies"][f"{best}_profit_rule"] = value((best_up * margin - COST > 0).astype(int))
report["policies"]["random_30pct"] = value(top_budget(rng.random(len(te))))
report["policies"]["treat_everyone"] = value(np.ones(len(te), int))
report["policies"]["treat_nobody"] = value(np.zeros(len(te), int))
report["policies"]["ORACLE_true_uplift"] = value((tau * margin - COST > 0).astype(int))

print("\nMODEL QUALITY (test):")
for k_, v in report["models"].items():
    print(f"  {k_:15s} Qini={v['test_qini']:.4f}  corr_with_true_uplift={v['spearman_with_true_uplift']:.2f}  "
          f"true_uplift_of_targeted={v['true_uplift_in_top30pct']:.3f}")
print("\nPOLICY PROFIT per customer (taka):")
for k_, v in sorted(report["policies"].items(), key=lambda kv: -kv[1]["true_profit"]):
    print(f"  {k_:26s} true={v['true_profit']:7.2f}  estimated={v['estimated_profit']:7.2f}  treated={v['treated_share']:.0%}")

joblib.dump({"model": models[best], "name": best, "features": FEATURES, "cost": COST,
             "margin_rate": MARGIN_RATE}, "model/uplift_model.joblib")
json.dump(report, open("model/metrics.json", "w"), indent=2)
print("\nsaved model/uplift_model.joblib")
