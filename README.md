# Uplift Modeling: kake coupon dile asol labh? (Causal ML)

## Shomossha
Company coupon pathay. Kintu:
- **Sure thing**: coupon chhara-i kinto -> coupon-er taka nosto
- **Lost cause**: coupon dileo kinbe na -> nosto
- **Sleeping dog**: coupon dile ulta birokto hoy -> khoti
- **Persuadable**: coupon dile-i KINBE -> shudhu eke dite hobe

Prai shob company "ke kinbe" (response model) predict kore ar sure-thing der coupon dey.
Uplift model predict kore "coupon dile KOTO BESHI kinbe" (causal effect).

## Chalanor niyom
pip install -r requirements.txt
python generate_data.py     # randomized experiment (A/B test) data
python train.py             # naive vs S/T/X-learner, Qini, policy profit
pytest -q
uvicorn app.main:app --reload     # POST /score, /score-batch

## Company te ja ja use hoy (ei project e ache)
- Randomized experiment data (RCT) theke causal effect shekha
- S-learner, T-learner, X-learner
- Qini curve / AUUC (accuracy na, uplift er jonno alada metric)
- Policy evaluation: RCT data theke IPS diye profit estimate, synthetic data te asol profit er sathe milano
- Business rule: expected_profit = uplift x margin - coupon cost
- Train / validation / test: model validation e baachi, test e ekbar-i report

## Mone rakho
- Correlation != causation. Observational data te eta kora onek kothin (confounding), tai company RCT chalay.
- True uplift jana jay shudhu synthetic data te. Real jibone shudhu Qini ar IPS ache.

## Challenge
1. Propensity score soho observational (non-random) data banao, dekho model kothay vul kore
2. Doubly-robust estimator (DR-learner) add koro
3. Qini curve ke plot koro (matplotlib)
4. Budget constraint optimization: 20% customer er maddhome max profit
5. Uplift er confidence interval (bootstrap)
6. Multi-treatment: coupon 10% vs 20% vs free delivery
