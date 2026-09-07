# Model Card — Employee Attrition Prediction

## Intended use
Portfolio demonstration of an HR decision-support workflow: estimate attrition likelihood, surface risk bands, and explain model drivers. It is not a hiring, firing, promotion, or disciplinary decision system.

## Data
IBM HR Analytics Employee Attrition & Performance sample dataset: 1,470 employees, 35 original attributes. Target is Attrition (Yes/No).

## Modeling
- Preprocessing: duplicate removal, identifier/constant-column removal, median/most-frequent imputation, scaling, one-hot encoding.
- Feature engineering: five interpretable HR ratios/composites.
- Candidates: Logistic Regression, Decision Tree, Random Forest, XGBoost.
- Selection: top two candidates tuned with stratified CV; final model selected by mean CV F1, never by holdout test performance.
- Final model in this release: XGBoost (Tuned).
- Explainability: SHAP TreeExplainer with native-importance fallback.

## Limitations
- Single-company/sample dataset; generalization is unverified.
- Associations are not causal.
- Attrition timing is not modeled.
- Risk thresholds are configurable business choices, not validated standards.
- Human review and fairness monitoring are required for real HR use.
