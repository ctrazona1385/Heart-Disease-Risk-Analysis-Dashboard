import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import joblib

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Heart Disease Risk Analyzer", layout="wide")
st.title("❤️ Heart Disease Risk Analyzer")
st.markdown("Enter your health metrics below to assess your risk level.")

# ---- LOAD MODEL ARTIFACTS ----
@st.cache_resource
def load_artifacts():
    model        = joblib.load('models/model.pkl')
    shap_model   = joblib.load('models/shap_model.pkl')
    scaler       = joblib.load('models/scaler.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    explainer    = shap.TreeExplainer(shap_model)
    return model, scaler, feature_cols, explainer

try:
    model, scaler, feature_cols, explainer = load_artifacts()
    model_ready = True
except FileNotFoundError:
    st.warning("Model not found. Run the notebook first to generate model.pkl, scaler.pkl, and feature_cols.pkl.")
    model_ready = False

# ---- LABEL MAPPINGS ----
FEATURE_LABEL_MAP = {
    'age':                           'Age',
    'trestbps':                      'Resting Blood Pressure (mmHg)',
    'chol':                          'Cholesterol (mg/dL)',
    'thalch':                        'Max Heart Rate (bpm)',
    'oldpeak':                       'ST Depression',
    'ca':                            'Major Vessels (Fluoroscopy)',
    'sex_Male':                      'Sex: Male',
    'cp_atypical angina':            'Chest Pain: Atypical Angina',
    'cp_non-anginal':                'Chest Pain: Non-Anginal Pain',
    'cp_typical angina':             'Chest Pain: Typical Angina',
    'fbs_True':                      'Fasting Blood Sugar > 120 mg/dL',
    'restecg_normal':                'Resting ECG: Normal',
    'restecg_st-t abnormality':      'Resting ECG: ST-T Wave Abnormality',
    'exang_True':                    'Exercise-Induced Angina: Yes',
    'slope_flat':                    'ST Slope: Flat',
    'slope_upsloping':               'ST Slope: Upsloping',
    'thal_normal':                   'Thalassemia: Normal',
    'thal_reversable defect':        'Thalassemia: Reversible Defect',
    'age_group_Adult (30-44)':       'Age Group: Adult (30–44)',
    'age_group_Middle-Aged (45-59)': 'Age Group: Middle-Aged (45–59)',
    'age_group_Senior (60+)':        'Age Group: Senior (60+)',
}

# Features a patient can act on (lifestyle / treatment modifiable)
ACTIONABLE_FEATURES = {'chol', 'trestbps', 'thalch', 'oldpeak', 'slope_flat', 'slope_upsloping'}

# ---- PREPROCESSING HELPER ----
def preprocess_input(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                     restecg, max_hr, exercise_angina, oldpeak, slope, ca, thal,
                     scaler, feature_cols):
    """Map sidebar inputs to the exact OHE feature layout used during training."""

    # Derive age group using same bins as notebook
    age_bins   = [0, 29, 44, 59, 120]
    age_labels = ['Young Adult (18-29)', 'Adult (30-44)', 'Middle-Aged (45-59)', 'Senior (60+)']
    age_group  = pd.cut([age], bins=age_bins, labels=age_labels)[0]

    # Map display strings to dataset values
    cp_map = {
        "Typical Angina":             "typical angina",
        "Atypical Angina":            "atypical angina",
        "Non-Anginal Pain":           "non-anginal",
        "No Symptoms (Asymptomatic)": "asymptomatic",
    }
    restecg_map = {
        "Normal":              "normal",
        "ST-T Abnormality":    "st-t abnormality",
        "LV Hypertrophy":      "lv hypertrophy",
    }
    slope_map = {
        "Upsloping":   "upsloping",
        "Flat":        "flat",
        "Downsloping": "downsloping",
    }
    thal_map = {
        "Normal":            "normal",
        "Fixed Defect":      "fixed defect",
        "Reversible Defect": "reversable defect",   # dataset column keeps original spelling
    }

    row = {
        'age':          age,
        'trestbps':     resting_bp,
        'chol':         cholesterol,
        'thalch':       max_hr,
        'oldpeak':      oldpeak,
        'ca':           ca,
        'sex':          'Male' if sex == 'Male' else 'Female',
        'cp':           cp_map[chest_pain],
        'fbs':          'TRUE' if fasting_bs == 'Yes' else 'FALSE',
        'restecg':      restecg_map[restecg],
        'exang':        'TRUE' if exercise_angina == 'Yes' else 'FALSE',
        'slope':        slope_map[slope],
        'thal':         thal_map[thal],
        'age_group':    str(age_group),
    }

    df_row = pd.DataFrame([row])

    # One-hot encode categoricals
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'age_group']
    df_row = pd.get_dummies(df_row, columns=categorical_cols, drop_first=True)

    # Align to training column layout (fills missing OHE columns with 0)
    df_row = df_row.reindex(columns=feature_cols, fill_value=0).astype(float)

    # Scale numeric columns
    numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    df_row[numeric_cols] = scaler.transform(df_row[numeric_cols].values)

    return df_row

# ---- SIDEBAR INPUTS ----
st.sidebar.header("Your Health Metrics")

age             = st.sidebar.slider("Age", 20, 80, 45)
sex             = st.sidebar.selectbox("Sex", ["Male", "Female"])
chest_pain      = st.sidebar.selectbox("Chest Pain Type",
                    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "No Symptoms (Asymptomatic)"])
resting_bp      = st.sidebar.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
cholesterol     = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs      = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
max_hr          = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["No", "Yes"])
oldpeak         = st.sidebar.number_input("ST Depression (Oldpeak)", 0.0, 6.2, 1.0, step=0.1)
restecg         = st.sidebar.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
slope           = st.sidebar.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
ca              = st.sidebar.slider("Major Vessels (Fluoroscopy)", 0, 3, 0)
thal            = st.sidebar.selectbox("Thalassemia Type", ["Normal", "Fixed Defect", "Reversible Defect"])

predict_btn = st.sidebar.button("Analyze My Risk", type="primary")

# ---- MAIN CONTENT ----
if predict_btn:
    if not model_ready:
        st.error("Cannot analyze — model artifacts not found. Run the notebook first.")
    else:
        input_df   = preprocess_input(age, sex, chest_pain, resting_bp, cholesterol,
                                      fasting_bs, restecg, max_hr, exercise_angina,
                                      oldpeak, slope, ca, thal, scaler, feature_cols)
        risk_score = model.predict_proba(input_df)[0][1]
        risk_pct   = int(risk_score * 100)

        if risk_pct < 40:
            risk_label, color = "Low Risk", "green"
        elif risk_pct < 70:
            risk_label, color = "Moderate Risk", "orange"
        else:
            risk_label, color = "High Risk", "red"

        # ---- Pre-compute all metric statuses (single source of truth) ----
        def _status(val, low, high, reverse=False):
            if reverse:
                if val >= high: return "Normal", "✅"
                if val >= low:  return "Borderline", "⚠️"
                return "Low", "🔴"
            if val <= low:  return "Normal", "✅"
            if val <= high: return "Borderline", "⚠️"
            return "High", "🔴"

        bp_status,   bp_icon   = _status(resting_bp,  120,  140)
        chol_status, chol_icon = _status(cholesterol,  200,  240)
        hr_status,   hr_icon   = _status(max_hr,        85,  100, reverse=True)
        op_status,   op_icon   = _status(oldpeak,        1.0, 2.0)
        _slope_map = {"Upsloping": ("Normal", "✅"), "Flat": ("Borderline", "⚠️"), "Downsloping": ("High Risk", "🔴")}
        sl_status,   sl_icon   = _slope_map.get(slope, ("Unknown", "❓"))

        # ---- ROW 1: Gauge + Stats + Extra Metrics ----
        col1, col2, col3 = st.columns([1.2, 1, 1])

        with col1:
            st.subheader("Your Risk Score")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40],  "color": "#d4edda"},
                        {"range": [40, 70], "color": "#fff3cd"},
                        {"range": [70, 100],"color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": risk_pct
                    }
                },
                title={"text": risk_label}
            ))
            fig.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("Your Stats vs. Thresholds")
            col2_metrics = [
                ("Blood Pressure", resting_bp, "mmHg",  bp_status,   bp_icon,   f"normal < 120 mmHg"),
                ("Cholesterol",    cholesterol, "mg/dL", chol_status, chol_icon, f"normal < 200 mg/dL"),
                ("Max Heart Rate", max_hr,      "bpm",   hr_status,   hr_icon,   f"normal ≥ 100 bpm"),
            ]
            for label, val, unit, status, icon, note in col2_metrics:
                st.metric(
                    label=f"{icon} {label}",
                    value=f"{val} {unit}",
                    delta=f"{status} ({note})",
                    delta_color="off"
                )

        with col3:
            st.subheader("Additional Risk Indicators")
            st.metric(
                label=f"{op_icon} ST Depression",
                value=f"{oldpeak:.1f} mm",
                delta=f"{op_status} (normal ≤ 1.0 mm)",
                delta_color="off"
            )
            st.metric(
                label=f"{sl_icon} ST Slope",
                value=slope,
                delta=f"{sl_status} (Upsloping is normal)",
                delta_color="off"
            )

        st.divider()

        # ---- ROW 2: SHAP (actionable features only) ----
        st.subheader("What's Driving Your Risk? (Actionable Factors)")

        shap_vals = explainer.shap_values(input_df)
        if isinstance(shap_vals, list):
            vals = np.array(shap_vals[1][0]).flatten()
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            vals = shap_vals[0, :, 1].flatten()
        else:
            vals = np.array(shap_vals[0]).flatten()

        # Build full SHAP dataframe with plain labels
        total = np.sum(np.abs(vals))
        shap_data = pd.DataFrame({
            "Feature": feature_cols,
            "Impact":  vals,
            "Pct":     np.abs(vals) / total * 100 if total > 0 else np.zeros(len(vals))
        })
        shap_data["Label"] = shap_data["Feature"].map(FEATURE_LABEL_MAP).fillna(shap_data["Feature"])

        # Filter to actionable features only for patient-facing chart
        # For OHE binary features (e.g. slope_flat, slope_upsloping), only show
        # the one that is active (value=1) — the inactive one (value=0) just means
        # "not this category" and is confusing to display.
        continuous_features = {'chol', 'trestbps', 'thalch', 'oldpeak'}
        ohe_inactive = {
            feat for feat in ACTIONABLE_FEATURES
            if feat not in continuous_features
            and feat in input_df.columns
            and input_df[feat].values[0] == 0
        }
        actionable_data = shap_data[
            shap_data["Feature"].isin(ACTIONABLE_FEATURES) &
            ~shap_data["Feature"].isin(ohe_inactive)
        ].copy()
        actionable_data = actionable_data.sort_values("Impact", ascending=True)

        fig2 = go.Figure(go.Bar(
            x=actionable_data["Impact"],
            y=actionable_data["Label"],
            orientation="h",
            text=[f"{p:.1f}%" for p in actionable_data["Pct"]],
            textposition="outside",
            marker_color=["tomato" if x > 0 else "steelblue" for x in actionable_data["Impact"]]
        ))
        fig2.update_layout(
            title="Actionable Risk Factors — What You Can Change (Red = increases risk, Blue = protective)",
            xaxis_title="Impact on Risk Score (SHAP)",
            height=350,
            margin=dict(t=60, b=20, l=260, r=80)
        )
        st.plotly_chart(fig2, width='stretch')
        st.caption(
            "This chart shows only factors you can influence through lifestyle or treatment. "
            "Other clinical factors (vessel count, thalassemia type, ECG results, sex) "
            "are still used by the model but are not shown here as they cannot be changed."
        )

        # ---- ROW 3: Advice ----
        st.subheader("📋 Personalized Recommendations")

        advice = []

        if chol_status != "Normal":
            chol_msg = (
                f"{chol_icon} **Cholesterol** is {chol_status.lower()} at {cholesterol} mg/dL. "
            )
            if chol_status == "High":
                chol_msg += "This significantly raises cardiac risk. Seek medical advice promptly about statins and switch to a heart-healthy diet (reduce saturated fats, increase fiber, omega-3s)."
            else:
                chol_msg += "Aim below 200 mg/dL through diet changes (reduce saturated fats, increase fiber) and discuss prevention options with your doctor."
            advice.append(chol_msg)

        if bp_status != "Normal":
            bp_msg = (
                f"{bp_icon} **Blood pressure** is {bp_status.lower()} at {resting_bp} mmHg. "
            )
            if bp_status == "High":
                bp_msg += "This is a significant risk factor. Reduce sodium, limit alcohol, manage stress, add cardio exercise, and discuss medication with your doctor."
            else:
                bp_msg += "Reduce sodium intake, manage stress, and add regular cardio exercise to help bring it down."
            advice.append(bp_msg)

        if hr_status != "Normal":
            hr_msg = (
                f"{hr_icon} **Max heart rate** is {hr_status.lower()} at {max_hr} bpm. "
            )
            if hr_status == "Low":
                hr_msg += "This may indicate significantly reduced cardiovascular fitness. Consult your doctor before starting any exercise program."
            else:
                hr_msg += "Gradual aerobic exercise such as walking or cycling is recommended — consult your doctor before starting."
            advice.append(hr_msg)

        if op_status != "Normal":
            op_msg = (
                f"{op_icon} **ST Depression** is {op_status.lower()} at {oldpeak:.1f} mm, which can indicate reduced blood flow during exertion. "
            )
            if op_status == "High":
                op_msg += "Seek prompt medical evaluation. Engage only in supervised cardiac rehab, avoid high-intensity effort, and discuss medication options with your cardiologist."
            else:
                op_msg += "Engage in low-intensity aerobic exercise, avoid sudden high-intensity effort, and discuss this finding with your cardiologist."
            advice.append(op_msg)

        if sl_status != "Normal":
            sl_msg = (
                f"{sl_icon} **ST Slope** is {slope.lower()}, which is associated with "
                + ("higher" if sl_status == "High Risk" else "elevated") + " cardiac risk. "
            )
            if sl_status == "High Risk":
                sl_msg += "Discuss urgent cardiac evaluation with your doctor. Sustained aerobic exercise, smoking cessation, and tight control of BP and cholesterol may help improve this over time."
            else:
                sl_msg += "Steps that may help: sustained aerobic exercise (e.g. walking, swimming), smoking cessation, weight management, and tight control of blood pressure and cholesterol."
            advice.append(sl_msg)

        if not advice:
            advice.append("✅ Your metrics look relatively healthy! Keep maintaining your current lifestyle.")

        for tip in advice:
            st.markdown(f"- {tip}")

else:
    st.info("👈 Enter your health metrics in the sidebar and click **Analyze My Risk** to get started.")
