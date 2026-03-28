# Telco Customer Churn Analysis

**IBM Watson Dataset** · EDA + Logistic Regression · Python

---

I did this project to get comfortable with the full analyst workflow — from messy raw data through to a model I could actually hand to a business team with a straight face. Churn analysis felt like the right problem because the business question is obvious (who's leaving and why?) but the data always has interesting wrinkles.

The dataset is IBM's Telco churn dataset: 7,043 customers, 21 features, binary churn label. Freely available on Kaggle.

---

## What's in the notebook

The analysis runs in two parts:

**Part 1 — EDA.** Before touching any model, I wanted to understand the shape of the problem. Who actually churns? When? What do they have in common? Most of the useful findings came from this phase.

**Part 2 — Modelling.** Logistic regression as a first pass — interpretable enough that you can explain a prediction to a non-technical stakeholder, which matters in practice. AUC came out at **0.84** without any hyperparameter tuning.

---

## The findings that actually stood out

**The contract type gap is enormous.**

| Contract | Churn Rate | Avg Tenure |
|---|---|---|
| Month-to-month | **42.7%** | 18 months |
| One year | 11.3% | 42 months |
| Two year | **2.8%** | 57 months |

That's not a marginal difference — month-to-month customers churn at 15x the rate of two-year customers. Getting someone to commit to an annual plan is probably the single highest-leverage retention move available.

**The first 6 months are make-or-break.**

47% of eventual churners leave within the first 6 months. By the time a customer hits 3 years, churn rate drops below 5%. Whatever retention effort you're going to make, front-load it heavily.

**Fiber optic customers churn at 42% — same as month-to-month.**

This one surprised me. Faster internet should mean happier customers. But fiber customers also pay significantly more per month (~$20 more on average than DSL), and my read is that the price-to-value perception is off somewhere. A targeted NPS survey for this segment would tell you more than any model could.

**Add-on services roughly halve churn rate.**

Across online security, tech support, device protection, and online backup — customers *without* these add-ons churn at roughly twice the rate of customers who have them. I don't think the services themselves are the cause. I think customers who set up multiple services have made a real commitment. The act of buying them is the signal, not the feature itself.

---

## Model results

Logistic regression with balanced class weights (the dataset is 74/26 split, so without balancing the model just predicts "retained" for everyone and looks misleadingly accurate).
```
              precision    recall    f1
Retained          0.91      0.72    0.80
Churned           0.51      0.80    0.62

ROC-AUC: 0.84
```

Recall of 0.80 on churners means the model catches 4 in 5 actual churners — which is what matters if you're using this to prioritize a retention team's outreach list.

**Top features by coefficient magnitude:**
1. Tenure (↓ churn risk — strongest predictor)
2. Monthly charges (↑ churn risk)
3. Contract type (↓ churn risk)
4. Total charges (↑ churn risk)
5. Online security (↓ churn risk)

**Revenue impact estimate** (from the notebook):
- Average revenue lost per churner: ~$1,269
- High-risk tier (top 20% by predicted probability) catches 71% of actual churners
- Saving even 20% of high-risk customers ≈ **$67,000 retained** in the test set alone

---

## What I'd do next

The logistic regression is a solid baseline but it has limits — it assumes linear relationships between features and churn probability, which probably isn't true for something like monthly charges. Next step would be XGBoost or LightGBM, plus SHAP values to explain individual predictions rather than just average feature importance.

I'd also want to build segment-level models — fiber vs. DSL customers probably have very different churn dynamics and a single model averaging over both might be missing something.

---

## Files
```
├── Telco_Customer_Churn_Analysis.ipynb   # Full analysis with outputs
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Source data (IBM Watson / Kaggle)
└── README.md
```

**Dataset source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Tools:** Python · Pandas · Matplotlib · Seaborn · Scikit-learn

---

*Part of my analytics portfolio — [github.com/payaljarviya](https://github.com/payaljarviya)*
