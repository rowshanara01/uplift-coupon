#### Uplift Modeling: Who Should Receive a Coupon? (Causal Machine Learning)

#### The Problem

Companies frequently send coupons to customers to encourage purchases.

However, not every customer needs a coupon.

There are four types of customers:

#### 1. Sure Things
These customers will purchase even without a coupon.

Giving them a coupon wastes money because they would have purchased anyway.

#### 2. Lost Causes
These customers will not purchase, even if they receive a coupon.

Giving them a coupon is also a waste of money.

#### 3. Sleeping Dogs
These customers may react negatively to receiving a coupon.

Instead of encouraging a purchase, the coupon may annoy them and reduce the likelihood of buying.

#### 4. Persuadables
These customers will purchase because they receive a coupon.

They are the most valuable target group for a coupon campaign.

#### Why Do Companies Need Uplift Modeling?

Most companies use traditional response models to predict:

"Who is most likely to purchase?"

The problem is that these models may target Sure Things, who would purchase without any discount.

Uplift Modeling predicts:

"How much more likely is this customer to purchase because of receiving a coupon?"

The goal is to identify Persuadable customers and maximize incremental profit rather than simply predicting purchase probability.

#### How to Run the Project

Install the dependencies:

pip install -r requirements.txt

Generate the dataset:

python generate_data.py

Generates randomized experiment (A/B testing) data.

Train and evaluate the models:

python train.py

This compares:
- Naive Response Model.
- S-Learner.
- T-Learner.
- X-Learner.
- Qini Curve.
- Policy Profit.

Run automated tests:

pytest -q

Start the API:

uvicorn app.main:app --reload

Available API endpoints:

- POST /score
- POST /score-batch

These endpoints predict individual customer uplift scores or process multiple customers in a batch.

#### What Real-World Companies Use in Uplift Modeling

### 1. Randomized Controlled Trials (RCT)

Learn causal effects using randomized experiment data.

Customers are randomly divided into two groups:

- Treatment Group: Customers who receive coupons.
- Control Group: Customers who do not receive coupons.

Compare their purchase behavior to estimate the incremental effect of the coupon.

Randomization helps reduce confounding and makes causal effect estimation more reliable.

### 2. S-Learner

The S-Learner uses a single machine learning model.

It receives customer features and treatment status as inputs.

For example:

- Customer age.
- Previous purchases.
- Customer activity.
- Coupon received (0 or 1).

The model predicts purchase probability under two scenarios:

- Without a coupon.
- With a coupon.

The difference between these predictions estimates the customer's uplift.

### 3. T-Learner

The T-Learner trains two separate models:

- Treatment Model: Learns purchase behavior among customers who received coupons.
- Control Model: Learns purchase behavior among customers who did not receive coupons.

The difference between their predictions estimates the treatment effect.

### 4. X-Learner

The X-Learner is an advanced meta-learning approach.

It uses treatment and control group information to estimate individual treatment effects.

It can be especially useful when treatment and control group sizes are imbalanced.

### 5. Qini Curve and AUUC

Traditional classification metrics such as accuracy and AUC are not sufficient to evaluate uplift models.

Use uplift-specific evaluation metrics:

- Qini Curve.
- AUUC (Area Under the Uplift Curve).

These metrics evaluate how effectively the model ranks customers according to their incremental treatment effects.

The objective is to identify customers who generate the greatest additional response from a coupon campaign.

### 6. Policy Evaluation Using IPS

Evaluate how profitable a coupon-targeting strategy would be using randomized experiment data.

Use IPS (Inverse Propensity Scoring) to estimate the expected value of a targeting policy.

Compare the estimated profit from the policy against the actual profit available in the synthetic dataset.

This helps evaluate whether the targeting strategy creates additional business value.

### 7. Business Decision-Making

The goal is to maximize profit, not simply maximize purchase probability.

Use this business formula:

Expected Profit = Uplift × Margin - Coupon Cost

For example:

If a customer has:
- 20% estimated uplift.
- $100 profit margin.
- $5 coupon cost.

Then:

Expected Profit = 0.20 × 100 - 5

Expected Profit = $15

The company can use this calculation to decide whether offering a coupon is financially worthwhile.

### 8. Train, Validation, and Test Strategy

Split the dataset into three parts:

- Training Set: Train the models.
- Validation Set: Select models and tune hyperparameters.
- Test Set: Evaluate final performance.

Use the validation set for model selection and report test performance only once.

This helps prevent overfitting and data leakage.

## Important Things to Remember

### 1. Correlation Is Not Causation

A correlation between coupon usage and purchases does not prove that coupons caused those purchases.

For example:

Customers who are already interested in buying may be more likely to use coupons.

This creates a confounding problem in observational data.

That is why companies conduct randomized controlled trials (RCTs) to estimate causal effects more reliably.

### 2. True Uplift Is Usually Unknown in Real Life

In synthetic datasets, we can know the true treatment effect because the data generation process is controlled.

However, in real-world business environments, we cannot directly observe what would have happened to the same customer both with and without a coupon.

This is called the fundamental problem of causal inference.

Therefore, real-world uplift models are evaluated using methods such as:

- Qini Curve.
- AUUC.
- IPS-based policy evaluation.

These methods provide estimates rather than directly revealing every customer's true individual uplift.

## Advanced Challenges

### 1. Observational Data with Propensity Scores

Generate non-randomized observational data using propensity scores.

Make treatment assignment depend on customer characteristics.

Then investigate where the model makes incorrect predictions because of confounding.

Compare the results with randomized experiment data.

### 2. Doubly Robust Estimation

Implement a DR-Learner (Doubly Robust Learner).

Combine:
- Outcome models.
- Propensity score models.

This provides another approach to estimating individual treatment effects and can remain consistent when either the outcome model or propensity model is correctly specified, under the required causal assumptions.

### 3. Plot the Qini Curve

Use Matplotlib to visualize the Qini Curve.

Compare different uplift models and understand which model identifies valuable customers more effectively.

### 4. Budget-Constrained Optimization

Assume the company has a limited coupon budget.

The goal:

Target only 20% of customers while maximizing total incremental profit.

Develop a targeting strategy that selects the most profitable customers within the available budget.

### 5. Uplift Confidence Intervals

Use Bootstrap sampling to estimate confidence intervals for uplift predictions and model performance.

This helps understand how uncertain the estimated treatment effects are.

### 6. Multi-Treatment Uplift Modeling

Extend the project to support multiple treatment options:

- 10% Discount Coupon.
- 20% Discount Coupon.
- Free Delivery.

Predict which offer generates the highest incremental profit for each customer.

Instead of simply deciding whether to send a coupon, the system should determine which type of offer is most effective.

---

## Project Goal

Build an end-to-end, production-oriented Causal Machine Learning system that identifies which customers are genuinely influenced by marketing campaigns.

The goal is to move beyond traditional purchase prediction and build an intelligent decision-making system that estimates incremental treatment effects, avoids unnecessary discounts, reduces marketing costs, and maximizes business profit.
