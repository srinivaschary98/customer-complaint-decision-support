
import streamlit as st
import pandas as pd
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
    "Week 9 Prototype — Customer Complaints Manager Decision Support"
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    app_dir = Path(__file__).resolve().parent

    complaint_file = app_dir / "complaints.csv"
    decision_context_file = app_dir / "decision_context.csv"
    predictions_file = app_dir / "predictions.csv"
    departments_file = app_dir / "departments.csv"

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
    "Complaint Review"
]

screen = st.sidebar.radio(
    "Select Screen",
    screen_options,
    index=screen_options.index(
        st.session_state.current_screen
    ),
    key="navigation_radio"
)

st.session_state.current_screen = screen

st.sidebar.divider()

st.sidebar.info(
    "The model provides recommendations. "
    "The Customer Complaints Manager makes the final decision."
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

        selected_id = st.selectbox(
            "Select Complaint ID",
            complaint_ids
        )

        if st.button(
    "Review Selected Complaint",
    type="primary"
):

    st.session_state.selected_id = selected_id

    st.session_state.current_screen = (
        "Complaint Review"
    )

    st.rerun()

else:

    st.warning(
        "No complaints match the selected filters."
    )

# =========================================================
# SCREEN 2 — COMPLAINT REVIEW
# =========================================================

else:

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

    if st.button(
        "Back to Complaint Queue"
    ):

        st.session_state.current_screen = (
    "Complaint Queue"
)

        st.rerun()
