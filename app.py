
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Customer Complaint Decision-Support Tool",
    page_icon="📋",
    layout="wide"
)

st.title("Customer Complaint Prioritization and Resolution Decision-Support Tool")

st.caption(
    "Week 10 Prototype — Predictive Model Performance and What-If Analysis"
)

APP_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    complaint_file = APP_DIR / "complaints.csv"
    decision_context_file = APP_DIR / "decision_context.csv"
    predictions_file = APP_DIR / "predictions.csv"
    departments_file = APP_DIR / "departments.csv"

    required_files = [
        complaint_file,
        decision_context_file,
        predictions_file,
        departments_file,
    ]

    missing_files = [p.name for p in required_files if not p.exists()]

    if missing_files:
        st.error(
            "Missing required file(s): "
            + ", ".join(missing_files)
        )
        st.stop()

    complaints = pd.read_csv(
        complaint_file,
        dtype={"complaint_id": str}
    )

    decision_context = pd.read_csv(
        decision_context_file,
        dtype={"complaint_id": str}
    )

    predictions = pd.read_csv(
        predictions_file,
        dtype={"complaint_id": str}
    )

    departments = pd.read_csv(
        departments_file,
        dtype={"department_id": str}
    )

    # Clean IDs
    complaints["complaint_id"] = complaints["complaint_id"].str.strip()
    decision_context["complaint_id"] = decision_context["complaint_id"].str.strip()
    predictions["complaint_id"] = predictions["complaint_id"].str.strip()

    # Dates
    complaints["date_received"] = pd.to_datetime(
        complaints["date_received"],
        errors="coerce"
    )

    # Department lookup
    department_lookup = departments.rename(
        columns={
            "department_id": "recommended_department_id",
            "department_name": "recommended_department_name"
        }
    )

    # -----------------------------------------------------
    # JOIN TABLES
    # -----------------------------------------------------

    data = complaints.merge(
        decision_context[
            [
                "complaint_id",
                "prior_complaint_count",
                "known_financial_impact",
                "current_workload",
                "available_capacity"
            ]
        ],
        on="complaint_id",
        how="left"
    )

    data = data.merge(
        predictions[
            [
                "complaint_id",
                "recommended_department_id",
                "predicted_urgency",
                "urgency_probability",
                "escalation_probability"
            ]
        ],
        on="complaint_id",
        how="left"
    )

    data = data.merge(
        department_lookup,
        on="recommended_department_id",
        how="left"
    )

    data["recommended_department"] = (
        data["recommended_department_id"].fillna("Unknown")
        + " - "
        + data["recommended_department_name"].fillna("Unknown")
    )

    # Sort High -> Medium -> Low
    priority_order = {
        "High": 1,
        "Medium": 2,
        "Low": 3
    }

    data["priority_order"] = (
        data["predicted_urgency"]
        .map(priority_order)
        .fillna(9)
    )

    data = data.sort_values(
        ["priority_order", "urgency_probability"],
        ascending=[True, False]
    )

    return data, departments


data, departments = load_data()


@st.cache_data
def load_week10_artifacts():
    """
    Load Week 10 metrics and bootstrap logistic-regression coefficients.

    The Streamlit deployment intentionally does not load the serialized
    scikit-learn .joblib file. This avoids version-specific pickle issues
    on Streamlit Cloud. The what-if prediction uses the 300 bootstrap
    logistic models created during model training.
    """

    metrics_file = APP_DIR / "week10_model_metrics.json"
    bootstrap_file = APP_DIR / "week10_bootstrap_coefficients.npz"

    required_files = [
        metrics_file,
        bootstrap_file,
    ]

    missing_files = [
        p.name for p in required_files
        if not p.exists()
    ]

    if missing_files:
        return None, None, missing_files

    with open(
        metrics_file,
        "r",
        encoding="utf-8"
    ) as f:
        metrics = json.load(f)

    bootstrap_raw = np.load(bootstrap_file)

    bootstrap = {
        "coefs": bootstrap_raw["coefs"],
        "intercepts": bootstrap_raw["intercepts"],
    }

    return metrics, bootstrap, []


def get_week10_artifacts_or_stop():
    metrics, bootstrap, missing_files = (
        load_week10_artifacts()
    )

    if missing_files:
        st.error(
            "Missing Week 10 file(s): "
            + ", ".join(missing_files)
            + ". Add them to the same GitHub folder as app.py."
        )
        st.stop()

    return metrics, bootstrap


@st.cache_data
def build_week10_preprocessing_reference(data):
    """
    Reproduce the preprocessing fitted on the 2023-2025 modeling data:
    median imputation + missing indicator + standardization for numeric
    fields, and sorted one-hot categories for categorical fields.
    """

    training_reference = data[
        data["date_received"].dt.year <= 2025
    ].copy()

    numeric_features = [
        "prior_complaint_count",
        "known_financial_impact",
        "current_workload",
        "available_capacity",
        "escalation_probability",
    ]

    categorical_features = [
        "issue",
        "submission_channel",
        "predicted_urgency",
    ]

    medians = (
        training_reference[numeric_features]
        .median()
        .to_dict()
    )

    numeric_imputed = (
        training_reference[numeric_features]
        .copy()
    )

    for col in numeric_features:
        numeric_imputed[col] = (
            numeric_imputed[col]
            .fillna(medians[col])
            .astype(float)
        )

    missing_indicator_features = [
        col
        for col in numeric_features
        if training_reference[col].isna().any()
    ]

    for col in missing_indicator_features:
        numeric_imputed[
            f"{col}__missing"
        ] = (
            training_reference[col]
            .isna()
            .astype(float)
        )

    numeric_means = (
        numeric_imputed.mean()
        .to_dict()
    )

    numeric_scales = (
        numeric_imputed.std(ddof=0)
        .replace(0, 1.0)
        .to_dict()
    )

    categories = {
        col: sorted(
            training_reference[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        for col in categorical_features
    }

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "medians": medians,
        "missing_indicator_features": missing_indicator_features,
        "numeric_columns_after_imputation": list(
            numeric_imputed.columns
        ),
        "numeric_means": numeric_means,
        "numeric_scales": numeric_scales,
        "categories": categories,
    }


def transform_what_if_row(what_if, preprocessing):
    """Transform one what-if row into the same 21-feature model space."""

    row = what_if.iloc[0]

    numeric_values = []

    for col in preprocessing["numeric_features"]:
        value = row[col]

        if pd.isna(value):
            value = preprocessing["medians"][col]

        numeric_values.append(float(value))

    for col in preprocessing["missing_indicator_features"]:
        numeric_values.append(
            1.0 if pd.isna(row[col]) else 0.0
        )

    standardized_numeric = []

    for col, value in zip(
        preprocessing["numeric_columns_after_imputation"],
        numeric_values,
    ):
        standardized_numeric.append(
            (
                value
                - preprocessing["numeric_means"][col]
            )
            / preprocessing["numeric_scales"][col]
        )

    one_hot = []

    for col in preprocessing["categorical_features"]:
        value = str(row[col])

        for category in preprocessing["categories"][col]:
            one_hot.append(
                1.0 if value == category else 0.0
            )

    return np.array(
        standardized_numeric + one_hot,
        dtype=float,
    )


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "current_screen" not in st.session_state:
    st.session_state.current_screen = "Complaint Queue"

if "selected_id" not in st.session_state:
    st.session_state.selected_id = None

if "demo_decisions" not in st.session_state:
    st.session_state.demo_decisions = {}


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Navigation")

screen_options = [
    "Complaint Queue",
    "Complaint Review",
    "Model Performance",
    "What-If Analysis"
]

if "navigation_radio" not in st.session_state:
    st.session_state.navigation_radio = (
        st.session_state.current_screen
    )


def sidebar_screen_changed():
    st.session_state.current_screen = (
        st.session_state.navigation_radio
    )


def open_selected_complaint():
    st.session_state.selected_id = (
        st.session_state.queue_selected_id
    )
    st.session_state.current_screen = "Complaint Review"
    st.session_state.navigation_radio = "Complaint Review"


def go_to_queue():
    st.session_state.current_screen = "Complaint Queue"
    st.session_state.navigation_radio = "Complaint Queue"


screen = st.sidebar.radio(
    "Select Screen",
    screen_options,
    key="navigation_radio",
    on_change=sidebar_screen_changed
)

st.session_state.current_screen = screen

st.sidebar.divider()

st.sidebar.info(
    "The model provides recommendations and risk estimates. "
    "The Customer Complaints Manager keeps the final decision."
)


# =========================================================
# SCREEN 1 — COMPLAINT QUEUE
# =========================================================

if screen == "Complaint Queue":

    st.header("Screen 1 — Complaint Queue")

    st.write(
        "This screen helps the manager identify which complaints "
        "should be reviewed first."
    )

    # -----------------------------------------------------
    # FILTERS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        selected_priority = st.multiselect(
            "Priority",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )

    with col2:

        issue_options = sorted(
            data["issue"]
            .dropna()
            .unique()
        )

        selected_issue = st.multiselect(
            "Issue",
            issue_options
        )

    with col3:

        department_options = sorted(
            data["recommended_department"]
            .dropna()
            .unique()
        )

        selected_department = st.multiselect(
            "Recommended Department",
            department_options
        )

    filtered = data.copy()

    filtered = filtered[
        filtered["predicted_urgency"]
        .isin(selected_priority)
    ]

    if selected_issue:

        filtered = filtered[
            filtered["issue"]
            .isin(selected_issue)
        ]

    if selected_department:

        filtered = filtered[
            filtered["recommended_department"]
            .isin(selected_department)
        ]

    # -----------------------------------------------------
    # SUMMARY METRICS
    # -----------------------------------------------------

    total = len(filtered)

    high_priority = (
        filtered["predicted_urgency"] == "High"
    ).sum()

    average_escalation = (
        filtered["escalation_probability"].mean()
        * 100
    )

    average_capacity = (
        filtered["available_capacity"].mean()
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Complaints Shown",
        f"{total:,}"
    )

    metric2.metric(
        "High Priority",
        f"{high_priority:,}"
    )

    metric3.metric(
        "Average Escalation Probability",
        f"{average_escalation:.1f}%"
    )

    metric4.metric(
        "Average Available Capacity",
        f"{average_capacity:.1f}"
    )

    st.divider()

    # -----------------------------------------------------
    # TABLE
    # -----------------------------------------------------

    display_data = filtered[
        [
            "complaint_id",
            "date_received",
            "issue",
            "sub_issue",
            "predicted_urgency",
            "urgency_probability",
            "recommended_department",
            "escalation_probability",
            "current_workload",
            "available_capacity"
        ]
    ].copy()

    display_data["urgency_probability"] = (
        display_data["urgency_probability"]
        * 100
    ).round(1)

    display_data["escalation_probability"] = (
        display_data["escalation_probability"]
        * 100
    ).round(1)

    display_data["current_workload"] = (
        display_data["current_workload"]
        .round(2)
    )

    display_data = display_data.rename(
        columns={
            "complaint_id": "Complaint ID",
            "date_received": "Date Received",
            "issue": "Issue",
            "sub_issue": "Sub-Issue",
            "predicted_urgency": "Priority",
            "urgency_probability": "Urgency %",
            "recommended_department": "Recommended Department",
            "escalation_probability": "Escalation %",
            "current_workload": "Current Workload",
            "available_capacity": "Available Capacity"
        }
    )

    st.dataframe(
        display_data.head(200),
        use_container_width=True,
        hide_index=True,
        height=450
    )

    if len(display_data) > 200:

        st.caption(
            "Showing the first 200 complaints."
        )

    # -----------------------------------------------------
    # SELECT COMPLAINT
    # -----------------------------------------------------

    st.subheader("Open Complaint")

    complaint_ids = (
        filtered["complaint_id"]
        .astype(str)
        .tolist()
    )

    if complaint_ids:

        default_index = 0

        if (
            st.session_state.selected_id
            in complaint_ids
        ):
            default_index = complaint_ids.index(
                st.session_state.selected_id
            )

        st.selectbox(
            "Select Complaint ID",
            complaint_ids,
            index=default_index,
            key="queue_selected_id"
        )

        st.button(
            "Review Selected Complaint",
            type="primary",
            on_click=open_selected_complaint
        )

    else:

        st.warning(
            "No complaints match the selected filters."
        )
# =========================================================
# SCREEN 2 — COMPLAINT REVIEW
# =========================================================

elif screen == "Complaint Review":

    st.header(
        "Screen 2 — Complaint Review and Manager Decision"
    )

    all_ids = (
        data["complaint_id"]
        .astype(str)
        .tolist()
    )

    if (
        st.session_state.selected_id
        not in all_ids
    ):

        st.session_state.selected_id = (
            all_ids[0]
        )

    selected_id = st.selectbox(
        "Complaint ID",
        all_ids,
        index=all_ids.index(
            st.session_state.selected_id
        )
    )

    st.session_state.selected_id = (
        selected_id
    )

    record = data[
        data["complaint_id"]
        .astype(str)
        == selected_id
    ].iloc[0]

    # -----------------------------------------------------
    # THREE INFORMATION PANELS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    # ------------------
    # Complaint Details
    # ------------------

    with col1:

        st.subheader("Complaint Details")

        st.write(
            "**Complaint ID:**",
            record["complaint_id"]
        )

        st.write(
            "**Date Received:**",
            record["date_received"].date()
        )

        st.write(
            "**Product:**",
            record["product"]
        )

        st.write(
            "**Sub-product:**",
            record["sub_product"]
        )

        st.write(
            "**Issue:**",
            record["issue"]
        )

        if pd.notna(
            record["sub_issue"]
        ):

            st.write(
                "**Sub-Issue:**",
                record["sub_issue"]
            )

        else:

            st.write(
                "**Sub-Issue:** Not Available"
            )

        st.write(
            "**Submission Channel:**",
            record["submission_channel"]
        )

    # ------------------
    # Decision Context
    # ------------------

    with col2:

        st.subheader("Decision Context")

        st.metric(
            "Prior Complaints",
            int(
                record[
                    "prior_complaint_count"
                ]
            )
        )

        if pd.notna(
            record[
                "known_financial_impact"
            ]
        ):

            st.metric(
                "Known Financial Impact",
                f"${record['known_financial_impact']:,.2f}"
            )

        else:

            st.metric(
                "Known Financial Impact",
                "Not Available"
            )

        st.metric(
            "Current Workload",
            f"{record['current_workload']:.2f}"
        )

        st.metric(
            "Available Capacity",
            int(
                record[
                    "available_capacity"
                ]
            )
        )

    # ------------------
    # Model Recommendation
    # ------------------

    with col3:

        st.subheader(
            "Model Recommendation"
        )

        st.metric(
            "Predicted Priority",
            record[
                "predicted_urgency"
            ]
        )

        st.metric(
            "Urgency Probability",
            f"{record['urgency_probability'] * 100:.1f}%"
        )

        st.metric(
            "Recommended Department",
            record[
                "recommended_department"
            ]
        )

        st.metric(
            "Escalation Probability",
            f"{record['escalation_probability'] * 100:.1f}%"
        )

    st.divider()

    # -----------------------------------------------------
    # MANAGER DECISION
    # -----------------------------------------------------

    st.subheader(
        "Manager Decision"
    )

    st.write(
        "The manager can accept or override "
        "the model recommendation."
    )

    department_dict = dict(
        zip(
            departments[
                "department_id"
            ].astype(str),

            departments[
                "department_name"
            ].astype(str)
        )
    )

    department_ids = list(
        department_dict.keys()
    )

    decision1, decision2, decision3 = (
        st.columns(3)
    )

    with decision1:

        priorities = [
            "Low",
            "Medium",
            "High"
        ]

        default_priority = (
            priorities.index(
                record[
                    "predicted_urgency"
                ]
            )
        )

        final_priority = st.selectbox(
            "Final Priority",
            priorities,
            index=default_priority
        )

    with decision2:

        recommended_id = str(
            record[
                "recommended_department_id"
            ]
        )

        if recommended_id in department_ids:

            default_department = (
                department_ids.index(
                    recommended_id
                )
            )

        else:

            default_department = 0

        final_department = (
            st.selectbox(
                "Final Department",
                department_ids,
                index=default_department,
                format_func=lambda x:
                f"{x} - {department_dict[x]}"
            )
        )

    with decision3:

        escalate = st.radio(
            "Escalate Complaint?",
            [
                "No",
                "Yes"
            ],
            horizontal=True
        )

    reason = st.text_area(
        "Manager Note",
        placeholder=(
            "Optional: explain why "
            "the recommendation was changed."
        )
    )

    if st.button(
        "Confirm Decision",
        type="primary"
    ):

        st.session_state.demo_decisions[
            selected_id
        ] = {

            "priority":
                final_priority,

            "department":
                final_department,

            "escalation":
                escalate,

            "reason":
                reason
        }

        st.success(
            f"""
            Decision saved for Complaint {selected_id}

            Priority: {final_priority}

            Department: {final_department} -
            {department_dict[final_department]}

            Escalate: {escalate}
            """
        )

    # -----------------------------------------------------
    # SHOW SAVED DECISION
    # -----------------------------------------------------

    if (
        selected_id
        in st.session_state.demo_decisions
    ):

        saved = (
            st.session_state
            .demo_decisions[
                selected_id
            ]
        )

        st.info(
            f"""
            Current Demo Decision

            Priority: {saved['priority']}

            Department: {saved['department']} -
            {department_dict[saved['department']]}

            Escalation: {saved['escalation']}
            """
        )

    st.divider()

    st.button(
        "Back to Complaint Queue",
        on_click=go_to_queue
    )

# =========================================================
# SCREEN 3 — MODEL PERFORMANCE
# =========================================================

elif screen == "Model Performance":

    metrics, bootstrap = (
        get_week10_artifacts_or_stop()
    )

    st.header(
        "Screen 3 — Escalation Model Performance"
    )

    st.write(
        "This screen shows how well the Week 10 Logistic Regression "
        "model identifies complaints that eventually escalate."
    )

    st.caption(
        "The decision threshold (35%) was selected using the 2025 "
        "development set. The performance below is from the held-out "
        "2026 test set."
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.1f}%"
    )

    m2.metric(
        "Precision",
        f"{metrics['precision'] * 100:.1f}%"
    )

    m3.metric(
        "Recall",
        f"{metrics['recall'] * 100:.1f}%"
    )

    m4.metric(
        "F1 Score",
        f"{metrics['f1'] * 100:.1f}%"
    )

    m5.metric(
        "ROC-AUC",
        f"{metrics['roc_auc']:.3f}"
    )

    st.divider()

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Confusion Matrix")

        confusion_df = pd.DataFrame(
            [
                [
                    metrics["true_negatives"],
                    metrics["false_positives"]
                ],
                [
                    metrics["false_negatives"],
                    metrics["true_positives"]
                ]
            ],
            index=[
                "Actual: No escalation",
                "Actual: Escalation"
            ],
            columns=[
                "Predicted: No",
                "Predicted: Yes"
            ]
        )

        st.dataframe(
            confusion_df,
            use_container_width=True
        )

        test_rows = (
            metrics["true_negatives"]
            + metrics["false_positives"]
            + metrics["false_negatives"]
            + metrics["true_positives"]
        )

        st.caption(
            f"Held-out 2026 test cases: {test_rows:,}"
        )

    with right:
        st.subheader("What this means for the manager")

        st.metric(
            "Escalations correctly identified",
            f"{metrics['true_positives']:,}"
        )

        st.metric(
            "Escalations missed",
            f"{metrics['false_negatives']:,}"
        )

        st.metric(
            "Decision threshold",
            f"{metrics['threshold'] * 100:.0f}%"
        )

    st.info(
        "Recall is the main screening metric for this prototype. "
        f"At the {metrics['threshold'] * 100:.0f}% threshold, the model "
        f"identified {metrics['recall'] * 100:.1f}% of complaints that "
        "actually escalated. The trade-off is more false positives, so "
        "the model is intended to support review rather than make an "
        "automatic escalation decision."
    )

    with st.expander("Model inputs and approach"):
        st.write(
            "Approach: Logistic Regression. The model combines complaint "
            "and operational information available around decision time."
        )

        st.write(
            "**Predictors:** prior complaint count, known financial impact, "
            "current workload, available capacity, issue, submission channel, "
            "predicted urgency, and the existing Week 9 escalation-risk signal."
        )

        st.write(
            "**Training design:** 2023–2024 was used for initial training, "
            "2025 for development and threshold selection, and the final "
            "model was fit on 2023–2025 before one evaluation on 2026."
        )


# =========================================================
# SCREEN 4 — WHAT-IF ANALYSIS
# =========================================================

elif screen == "What-If Analysis":

    metrics, bootstrap = (
        get_week10_artifacts_or_stop()
    )

    preprocessing = (
        build_week10_preprocessing_reference(
            data
        )
    )

    st.header(
        "Screen 4 — What-If Escalation Analysis"
    )

    st.write(
        "Change the predictor values below to see how the estimated "
        "probability of eventual escalation changes."
    )

    st.caption(
        "The result is a decision-support estimate, not an automatic "
        "manager decision."
    )

    # Use observed data to provide realistic defaults and options.
    issue_options = sorted(
        data["issue"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    channel_options = sorted(
        data["submission_channel"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        prior_complaints = st.number_input(
            "Prior complaints",
            min_value=0,
            max_value=20,
            value=2,
            step=1
        )

        impact_unknown = st.checkbox(
            "Financial impact is not known",
            value=False
        )

        financial_impact = st.number_input(
            "Known financial impact ($)",
            min_value=0.0,
            max_value=50000.0,
            value=1500.0,
            step=100.0,
            disabled=impact_unknown
        )

    with c2:
        current_workload = st.number_input(
            "Current workload",
            min_value=0.0,
            max_value=50.0,
            value=8.0,
            step=0.25
        )

        available_capacity = st.number_input(
            "Available capacity",
            min_value=0,
            max_value=100,
            value=12,
            step=1
        )

        week9_escalation_signal = st.slider(
            "Existing Week 9 escalation signal",
            min_value=0,
            max_value=100,
            value=60,
            step=1,
            help=(
                "The intake risk signal already used in the Week 9 "
                "prototype. The Week 10 model combines it with context."
            )
        )

    with c3:
        selected_issue = st.selectbox(
            "Issue",
            issue_options
        )

        selected_channel = st.selectbox(
            "Submission channel",
            channel_options
        )

        selected_urgency = st.selectbox(
            "Predicted urgency",
            ["Low", "Medium", "High"],
            index=2
        )

    st.divider()

    if st.button(
        "Run What-If Prediction",
        type="primary"
    ):

        impact_value = (
            np.nan
            if impact_unknown
            else float(financial_impact)
        )

        what_if = pd.DataFrame(
            {
                "prior_complaint_count": [
                    int(prior_complaints)
                ],
                "known_financial_impact": [
                    impact_value
                ],
                "current_workload": [
                    float(current_workload)
                ],
                "available_capacity": [
                    int(available_capacity)
                ],
                "escalation_probability": [
                    float(
                        week9_escalation_signal
                    ) / 100.0
                ],
                "issue": [
                    selected_issue
                ],
                "submission_channel": [
                    selected_channel
                ],
                "predicted_urgency": [
                    selected_urgency
                ]
            }
        )

        transformed_row = (
            transform_what_if_row(
                what_if,
                preprocessing
            )
        )

        coefs = bootstrap["coefs"]
        intercepts = bootstrap["intercepts"]

        if coefs.shape[1] != len(
            transformed_row
        ):
            st.error(
                "The bootstrap coefficients do not match the "
                "expected transformed feature count."
            )
            st.stop()

        linear_scores = (
            coefs @ transformed_row
            + intercepts
        )

        linear_scores = np.clip(
            linear_scores,
            -50,
            50
        )

        bootstrap_probabilities = (
            1.0
            / (
                1.0
                + np.exp(
                    -linear_scores
                )
            )
        )

        # Use the bootstrap ensemble mean as the point estimate.
        probability = float(
            np.mean(
                bootstrap_probabilities
            )
        )

        lower = float(
            np.percentile(
                bootstrap_probabilities,
                2.5
            )
        )

        upper = float(
            np.percentile(
                bootstrap_probabilities,
                97.5
            )
        )

        threshold = float(
            metrics["threshold"]
        )

        flagged = (
            probability >= threshold
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Predicted escalation probability",
            f"{probability * 100:.1f}%"
        )

        r2.metric(
            "95% bootstrap confidence interval",
            (
                f"{lower * 100:.1f}% "
                f"to {upper * 100:.1f}%"
            )
        )

        r3.metric(
            "Screening result",
            (
                "Flag for review"
                if flagged
                else "Below review threshold"
            )
        )

        if lower > threshold:
            st.success(
                "The full 95% confidence interval is above the "
                f"{threshold * 100:.0f}% review threshold. "
                "This scenario shows a relatively stable higher-risk signal."
            )

        elif upper < threshold:
            st.info(
                "The full 95% confidence interval is below the "
                f"{threshold * 100:.0f}% review threshold. "
                "This scenario shows a relatively stable lower-risk signal."
            )

        else:
            st.warning(
                "The 95% confidence interval crosses the "
                f"{threshold * 100:.0f}% review threshold. "
                "The manager should treat this prediction with extra caution."
            )

        st.caption(
            "The point estimate is the mean prediction across 300 bootstrap "
            "logistic-regression models. The 95% interval shows bootstrap "
            "uncertainty in the estimated escalation probability; it does not "
            "mean that the individual complaint's outcome itself lies inside "
            "an interval."
        )

        with st.expander(
            "What-if values used"
        ):
            st.dataframe(
                what_if,
                use_container_width=True,
                hide_index=True
            )

