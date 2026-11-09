import numpy as np


def uplift_curve(score, t, y):
    """Score onujayi sajiye prothom k jon ke coupon dile koto extra purchase (Qini curve)"""
    o = np.argsort(-score)
    t, y = t[o], y[o]
    ct, cc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    n = np.arange(1, len(y) + 1)
    return (yt / np.maximum(ct, 1) - yc / np.maximum(cc, 1)) * n


def qini(score, t, y):
    """Random coupon dewar cheye koto bhalo (0 = random er moto, beshi = bhalo)"""
    c = uplift_curve(score, t, y)
    return float((c - np.linspace(0, c[-1], len(c))).mean() / len(c))


def ips_value(treat, t, y, margin, cost, e=0.5):
    """RCT data theke policy er expected profit (importance weighting)"""
    w = np.where(treat == 1, 1 / e, 1 / (1 - e)) * (t == treat)
    return float(np.mean(w * (y * margin - treat * cost)))


def true_value(treat, p0, tau, margin, cost):
    """Synthetic data te asol profit (jani bole)"""
    return float(np.mean((p0 + treat * tau) * margin - treat * cost))
