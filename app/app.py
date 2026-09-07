"""
Employee Attrition Prediction & HR Intelligence System
Run: streamlit run app/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, for `src`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this app/ dir, for `app_utils`

from app_utils import get_dataset, get_explainer_cache, get_model_bundle, get_results_table  # noqa: E402
from src import config, predict  # noqa: E402

st.set_page_config(
    page_title="Employee Attrition Intelligence",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLORS = {"LOW": "#2E7D32", "MEDIUM": "#F9A825", "HIGH": "#C62828"}


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("HR Intelligence System")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "HR Dashboard", "Employee Risk Prediction", "Model Performance", "Employee Insights", "About Project"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Predicts employee attrition risk and explains the key drivers behind each "
    "prediction, using a model trained on the IBM HR Analytics dataset."
)

try:
    df = get_dataset()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

try:
    bundle = get_model_bundle()
    model_available = True
except FileNotFoundError:
    bundle = None
    model_available = False


# ----------------------------------------------------------------------------
# HOME
# ----------------------------------------------------------------------------
if page == "Home":
    st.title("\U0001F4CA Employee Attrition Prediction & HR Intelligence System")
    st.markdown(
        "An end-to-end data science system that predicts **which employees are at "
        "risk of leaving**, quantifies the risk, and explains **why** — so HR can "
        "act before it's too late."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", f"{len(df):,}")
    c2.metric("Attrition Cases", f"{int(df[config.TARGET_COL].sum()):,}")
    c3.metric("Attrition Rate", f"{df[config.TARGET_COL].mean()*100:.1f}%")
    if model_available:
        c4.metric("Model", bundle["model_name"])
    else:
        c4.metric("Model", "Not trained yet")

    st.markdown("### What this system does")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "- **HR Dashboard** — company-wide attrition analytics\n"
            "- **Employee Risk Prediction** — score any individual employee\n"
            "- **Model Performance** — full evaluation for technical review"
        )
    with col2:
        st.markdown(
            "- **Employee Insights** — dynamic, SHAP-based explanations\n"
            "- **About Project** — methodology, data, and limitations\n"
        )

    if not model_available:
        st.warning("No trained model found. Run `python -m src.train` from the project root first.")


# ----------------------------------------------------------------------------
# HR DASHBOARD
# ----------------------------------------------------------------------------
elif page == "HR Dashboard":
    st.title("HR Dashboard")

    probs = None
    if model_available:
        feature_cols = bundle["feature_columns"]
        probs = bundle["pipeline"].predict_proba(df[feature_cols])[:, 1]
        df_disp = df.copy()
        df_disp["AttritionProbability"] = probs
        df_disp["RiskLevel"] = df_disp["AttritionProbability"].apply(config.probability_to_risk)
    else:
        df_disp = df.copy()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Employees", f"{len(df_disp):,}")
    c2.metric("Attrition Count", f"{int(df_disp[config.TARGET_COL].sum()):,}")
    c3.metric("Attrition Rate", f"{df_disp[config.TARGET_COL].mean()*100:.1f}%")
    if probs is not None:
        c4.metric("High-Risk Employees", f"{int((df_disp['RiskLevel']=='HIGH').sum()):,}")
        c5.metric("Avg Attrition Probability", f"{probs.mean()*100:.1f}%")
    else:
        c4.metric("High-Risk Employees", "N/A")
        c5.metric("Avg Attrition Probability", "N/A")

    st.markdown("---")

    df_disp["AttritionLabel"] = df_disp[config.TARGET_COL].map({1: "Yes", 0: "No"})

    row1c1, row1c2 = st.columns(2)
    with row1c1:
        by_dept = df_disp.groupby("Department")["AttritionLabel"].value_counts(normalize=True).unstack().fillna(0)
        fig = px.bar(
            (by_dept.get("Yes", 0) * 100).reset_index(name="AttritionRate%"),
            x="Department", y="AttritionRate%", title="Attrition Rate by Department",
            color="AttritionRate%", color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Departments with a higher bar lose a larger share of their workforce.")

    with row1c2:
        by_role = df_disp.groupby("JobRole")["AttritionLabel"].value_counts(normalize=True).unstack().fillna(0)
        by_role = (by_role.get("Yes", 0) * 100).sort_values(ascending=False).reset_index(name="AttritionRate%")
        fig = px.bar(
            by_role, x="AttritionRate%", y="JobRole", orientation="h",
            title="Attrition Rate by Job Role", color="AttritionRate%", color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Sales Representatives and Lab Technicians typically show the highest exit rates.")

    row2c1, row2c2 = st.columns(2)
    with row2c1:
        age_bins = pd.cut(df_disp["Age"], bins=[17, 25, 35, 45, 55, 65], labels=["18-25", "26-35", "36-45", "46-55", "56-65"])
        by_age = df_disp.assign(AgeGroup=age_bins).groupby("AgeGroup", observed=True)["AttritionLabel"].value_counts(normalize=True).unstack().fillna(0)
        fig = px.bar(
            (by_age.get("Yes", 0) * 100).reset_index(name="AttritionRate%"),
            x="AgeGroup", y="AttritionRate%", title="Attrition Rate by Age Group", color_discrete_sequence=["#1f77b4"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Younger employees tend to show higher attrition rates, a common early-career pattern.")

    with row2c2:
        by_ot = df_disp.groupby("OverTime")["AttritionLabel"].value_counts(normalize=True).unstack().fillna(0)
        fig = px.bar(
            (by_ot.get("Yes", 0) * 100).reset_index(name="AttritionRate%"),
            x="OverTime", y="AttritionRate%", title="Attrition Rate by Overtime", color_discrete_sequence=["#d62728"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Employees who work overtime consistently show a markedly higher attrition rate.")

    row3c1, row3c2 = st.columns(2)
    with row3c1:
        fig = px.box(
            df_disp, x="AttritionLabel", y="JobSatisfaction", color="AttritionLabel",
            title="Job Satisfaction vs Attrition",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Lower job satisfaction scores are associated with a higher likelihood of leaving.")

    with row3c2:
        fig = px.box(
            df_disp, x="AttritionLabel", y="MonthlyIncome", color="AttritionLabel",
            title="Monthly Income vs Attrition",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Employees who leave tend to have noticeably lower monthly income.")

    fig = px.histogram(
        df_disp, x="YearsAtCompany", color="AttritionLabel", barmode="overlay", nbins=20,
        title="Years at Company by Attrition Status", opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Attrition is concentrated in the first few years of tenure.")


# ----------------------------------------------------------------------------
# EMPLOYEE RISK PREDICTION
# ----------------------------------------------------------------------------
elif page == "Employee Risk Prediction":
    st.title("Employee Risk Prediction")

    if not model_available:
        st.warning("No trained model found. Run `python -m src.train` from the project root first.")
        st.stop()

    st.markdown("Enter an employee's details to estimate their attrition risk.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 60, 30)
            monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000, step=100)
            job_level = st.selectbox("Job Level", sorted(df["JobLevel"].unique()))
            department = st.selectbox("Department", sorted(df["Department"].unique()))
            job_role = st.selectbox("Job Role", sorted(df["JobRole"].unique()))
            education_field = st.selectbox("Education Field", sorted(df["EducationField"].unique()))
            education = st.selectbox("Education Level (1-5)", sorted(df["Education"].unique()))
            gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
        with c2:
            job_satisfaction = st.select_slider("Job Satisfaction", [1, 2, 3, 4], value=3)
            environment_satisfaction = st.select_slider("Environment Satisfaction", [1, 2, 3, 4], value=3)
            relationship_satisfaction = st.select_slider("Relationship Satisfaction", [1, 2, 3, 4], value=3)
            work_life_balance = st.select_slider("Work Life Balance", [1, 2, 3, 4], value=3)
            job_involvement = st.select_slider("Job Involvement", [1, 2, 3, 4], value=3)
            overtime = st.radio("Overtime", ["Yes", "No"], horizontal=True)
            business_travel = st.selectbox("Business Travel", sorted(df["BusinessTravel"].unique()))
            marital_status = st.selectbox("Marital Status", sorted(df["MaritalStatus"].unique()))
        with c3:
            years_at_company = st.slider("Years at Company", 0, 40, 3)
            years_in_role = st.slider("Years in Current Role", 0, 20, 2)
            years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
            years_with_manager = st.slider("Years With Current Manager", 0, 20, 2)
            total_working_years = st.slider("Total Working Years", 0, 40, 5)
            num_companies = st.slider("Num Companies Worked", 0, 10, 1)
            distance_from_home = st.slider("Distance From Home (km)", 1, 30, 5)
            training_times = st.slider("Training Times Last Year", 0, 6, 2)

        submitted = st.form_submit_button("Predict Attrition Risk", use_container_width=True)

    if submitted:
        raw_input = {
            "Age": age, "DailyRate": int(df["DailyRate"].median()), "DistanceFromHome": distance_from_home,
            "Education": education, "EnvironmentSatisfaction": environment_satisfaction,
            "HourlyRate": int(df["HourlyRate"].median()), "JobInvolvement": job_involvement,
            "JobLevel": job_level, "JobSatisfaction": job_satisfaction, "MonthlyIncome": monthly_income,
            "MonthlyRate": int(df["MonthlyRate"].median()), "NumCompaniesWorked": num_companies,
            "PercentSalaryHike": int(df["PercentSalaryHike"].median()), "PerformanceRating": 3,
            "RelationshipSatisfaction": relationship_satisfaction, "StockOptionLevel": int(df["StockOptionLevel"].median()),
            "TotalWorkingYears": total_working_years, "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": work_life_balance, "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role, "YearsSinceLastPromotion": years_since_promotion,
            "YearsWithCurrManager": years_with_manager, "BusinessTravel": business_travel,
            "Department": department, "EducationField": education_field, "Gender": gender,
            "JobRole": job_role, "MaritalStatus": marital_status, "OverTime": overtime,
        }

        explainer_cache = get_explainer_cache()
        with st.spinner("Scoring employee..."):
            result = predict.predict_employee(raw_input, bundle, explainer_cache, df[bundle["feature_columns"]])

        st.markdown("---")
        st.subheader("ATTRITION RISK")
        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.metric("Probability", f"{result['probability']*100:.1f}%")
            st.markdown(
                f"<h2 style='color:{RISK_COLORS[result['risk']]}'>Risk: {result['risk']}</h2>",
                unsafe_allow_html=True,
            )
        with rc2:
            st.markdown("**Top Contributing Factors**")
            for i, f in enumerate(result["top_factors"], 1):
                arrow = "\U0001F53A" if f["effect"] == "increases risk" else "\U0001F53B"
                st.markdown(f"{i}. {arrow} **{f['feature']}** — {f['effect']}")
            if result["explain_mode"] == "fallback":
                st.caption("Note: SHAP unavailable for this model type; showing an approximate "
                           "importance-based explanation instead.")


# ----------------------------------------------------------------------------
# MODEL PERFORMANCE
# ----------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")

    results_df = get_results_table()
    if results_df.empty:
        st.warning("No results found. Run `python -m src.train` first.")
        st.stop()

    st.markdown("### Model Comparison")
    st.dataframe(results_df.set_index("Model").style.highlight_max(axis=0, color="#c6efce"), use_container_width=True)

    if model_available:
        st.markdown(f"**Final selected model:** `{bundle['model_name']}` — selected by mean "
                     "cross-validation F1; the held-out test set is reserved for final evaluation. "
                     "Accuracy alone is misleading on an imbalanced target.")

    from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
    import joblib

    eval_path = config.REPORTS_DIR / "eval_artifacts.pkl"
    if eval_path.exists() and model_available:
        artifacts = joblib.load(eval_path)
        X_test, y_test = artifacts["X_test"], artifacts["y_test"]
        pipeline = bundle["pipeline"]
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        col1, col2 = st.columns(2)
        with col1:
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Stay (0)", "Leave (1)"], y=["Stay (0)", "Leave (1)"],
                title="Confusion Matrix",
            )
            st.plotly_chart(fig, use_container_width=True)
            tn, fp, fn, tp = cm.ravel()
            st.markdown(
                f"- **TP={tp}**: correctly flagged leavers\n"
                f"- **TN={tn}**: correctly flagged stayers\n"
                f"- **FP={fp}**: flagged as leaving but stayed (costs an unnecessary retention effort)\n"
                f"- **FN={fn}**: predicted to stay but actually left — "
                "**the costliest error**, since HR takes no preventive action for these employees."
            )
        with col2:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig = px.area(
                x=fpr, y=tpr, title=f"ROC Curve (AUC = {roc_auc:.3f})",
                labels=dict(x="False Positive Rate", y="True Positive Rate"),
            )
            fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Classification Report")
        report = classification_report(y_test, y_pred, target_names=["Stay", "Leave"], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)


# ----------------------------------------------------------------------------
# EMPLOYEE INSIGHTS
# ----------------------------------------------------------------------------
elif page == "Employee Insights":
    st.title("Employee Insights — Explainable AI")

    if not model_available:
        st.warning("No trained model found. Run `python -m src.train` first.")
        st.stop()

    from src import explain

    feature_cols = bundle["feature_columns"]
    pipeline = bundle["pipeline"]
    explainer_cache = get_explainer_cache()

    if "global_explainer" not in explainer_cache:
        expl, feat_names, mode = explain.build_explainer(pipeline, df[feature_cols])
        explainer_cache["global_explainer"] = expl
        explainer_cache["global_feat_names"] = feat_names
        explainer_cache["global_mode"] = mode

    with st.spinner("Computing global feature importance..."):
        sample_df = df.sample(min(300, len(df)), random_state=42)
        importance_df = explain.global_feature_importance(
            pipeline, explainer_cache["global_explainer"], explainer_cache["global_feat_names"],
            sample_df[feature_cols], explainer_cache["global_mode"],
        )
    importance_df["feature"] = importance_df["feature"].apply(lambda n: explain.humanize_feature_name(n))
    top_15 = importance_df.head(15)

    st.markdown("### Global Feature Importance (What drives attrition company-wide)")
    fig = px.bar(top_15.sort_values("importance"), x="importance", y="feature", orientation="h",
                 title="Top 15 Features by Mean |SHAP value|" if explainer_cache["global_mode"] == "shap"
                       else "Top 15 Features by Model Importance")
    st.plotly_chart(fig, use_container_width=True)
    if explainer_cache["global_mode"] == "fallback":
        st.caption("SHAP unavailable for this model — showing native model feature importances instead.")

    st.markdown("---")
    st.markdown(
        "### Business Insights\n"
        "*Findings are correlational, drawn from observational HR data — not causal claims.*\n\n"
        "- **Overtime** is associated with a higher attrition rate; employees working overtime "
        "may be experiencing burnout or reduced work-life balance.\n"
        "- **Lower job/environment satisfaction** correlates with higher exit rates, suggesting "
        "engagement surveys can act as an early warning signal.\n"
        "- **Shorter tenure** (especially the first 2-3 years) is linked to elevated attrition risk, "
        "pointing to onboarding and early-career support as a retention lever.\n"
        "- **Lower relative income** (vs. experience) is associated with higher attrition, "
        "consistent with compensation-driven exits.\n"
        "- Certain **job roles** (e.g. Sales Representative, Laboratory Technician) show "
        "consistently higher attrition rates and may warrant role-specific retention programs."
    )


# ----------------------------------------------------------------------------
# ABOUT PROJECT
# ----------------------------------------------------------------------------
elif page == "About Project":
    st.title("About This Project")
    st.markdown(
        """
This is an end-to-end **Employee Attrition Prediction & HR Intelligence System**
built as a Data Science portfolio project.

**Dataset:** IBM HR Analytics Employee Attrition & Performance (1,470 employees, 35 attributes).

**Pipeline:** data cleaning → EDA → statistical hypothesis testing → feature engineering →
a scikit-learn `ColumnTransformer` + `Pipeline` preprocessing stage → baseline and 4 candidate
ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost) → stratified
cross-validation → hyperparameter tuning of the top 2 models → SHAP-based explainability →
this Streamlit application.

**Model selection criterion:** F1-score on the Attrition (minority) class on a held-out
test set, since this is an imbalanced classification problem where recall on leavers
matters more than raw accuracy.

**Risk thresholds:** LOW / MEDIUM / HIGH bands are configurable business thresholds
(0-40 / 40-70 / 70-100%), not an externally validated industry standard.

**Limitations:**
- The dataset is a single-company snapshot (IBM's synthetic/sample HR data), so findings
  may not generalize to other organizations.
- All relationships reported are **correlational**, not causal.
- Attrition timing (*when* someone will leave) is not modeled — only the likelihood.

See the repository's `README.md` for full methodology, metrics, and design
decision explanations.
"""
    )
