"""Randomized coupon experiment (A/B test) er synthetic data.
true_p0 / true_tau shudhu synthetic bole ache (asol jiboney thake na). Eta feature NOY, shudhu model shothik kina jachai korar jonno."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(5)
n = 40000
age = rng.normal(38, 12, n).clip(18, 75).round()
tenure = rng.exponential(24, n).clip(1, 120).round()
past = rng.poisson(4, n)
basket = rng.lognormal(6.0, 0.5, n).round(2)
recency = rng.exponential(30, n).clip(1, 180).round()
app_user = rng.binomial(1, 0.6, n)
coupon_rate = rng.beta(2, 5, n).round(3)   # ager coupon koto bar byabohar korechhe (price sensitivity)

sigmoid = lambda z: 1 / (1 + np.exp(-z))
p0 = sigmoid(-1.4 + 0.18 * past - 0.02 * recency + 0.4 * app_user)          # coupon chhara kenar chance
tau = np.clip(0.8 * (coupon_rate - 0.2) * (1 - p0), -0.10, 0.40)           # coupon er asol prabhab (uplift)
t = rng.binomial(1, 0.5, n)                                                 # RANDOM coupon dewa (RCT)
y = rng.binomial(1, np.clip(p0 + t * tau, 0.01, 0.99))

pd.DataFrame({"age": age, "tenure_months": tenure, "past_purchases": past, "avg_basket": basket,
              "days_since_last_purchase": recency, "app_user": app_user, "coupon_use_rate": coupon_rate,
              "treatment": t, "purchased": y, "true_p0": p0.round(4), "true_tau": tau.round(4)}
             ).to_csv("data/campaign.csv", index=False)
print(f"rows={n} | purchase rate control={y[t==0].mean():.3f} treated={y[t==1].mean():.3f} "
      f"| avg true uplift={tau.mean():.3f} | sleeping dogs (tau<0)={np.mean(tau<0):.1%}")
