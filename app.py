import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HR Analytics & Churn Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL UI STYLING
# =========================================================

st.markdown("""
<style>

    /* ================================
       MAIN APPLICATION
    ================================= */

    .main {
        padding-top: 1rem;
    }

    /* ================================
       HEADINGS
    ================================= */

    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    h2 {
        font-weight: 700 !important;
    }

    h3 {
        font-weight: 650 !important;
    }

    /* ================================
       METRIC CARDS
    ================================= */

    [data-testid="stMetric"] {
        background: linear-gradient(
            135deg,
            rgba(255,255,255,0.06),
            rgba(255,255,255,0.02)
        );

        border: 1px solid rgba(255,255,255,0.10);

        padding: 20px;

        border-radius: 16px;

        box-shadow:
            0 8px 24px rgba(0,0,0,0.18);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);

        box-shadow:
            0 12px 30px rgba(0,0,0,0.28);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 750 !important;
    }

    /* ================================
       BUTTONS
    ================================= */

    .stButton > button,
    .stDownloadButton > button {

        border-radius: 10px;

        min-height: 45px;

        font-weight: 650;

        border: 1px solid rgba(255,255,255,0.12);

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {

        transform: translateY(-2px);

        box-shadow:
            0 8px 20px rgba(0,0,0,0.25);
    }

    /* ================================
       DATAFRAME
    ================================= */

    [data-testid="stDataFrame"] {

        border-radius: 14px;

        overflow: hidden;

        border: 1px solid rgba(255,255,255,0.10);
    }

    /* ================================
       INPUT BOXES
    ================================= */

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox,
    .stMultiSelect {

        border-radius: 10px;
    }

    /* ================================
       ALERT / INFO BOXES
    ================================= */

    [data-testid="stAlert"] {

        border-radius: 12px;

        border: 1px solid rgba(255,255,255,0.10);
    }

    /* ================================
       DIVIDERS
    ================================= */

    hr {

        margin-top: 1.5rem;

        margin-bottom: 1.5rem;

        border: none;

        border-top:
            1px solid rgba(255,255,255,0.10);
    }

    /* ================================
       SIDEBAR
    ================================= */

    [data-testid="stSidebar"] {

        border-right:
            1px solid rgba(255,255,255,0.08);
    }

    /* ================================
       PLOTLY CONTAINERS
    ================================= */

    [data-testid="stPlotlyChart"] {

        border-radius: 14px;

        padding: 4px;

        background:
            rgba(255,255,255,0.015);
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    font-size: 14px;
}

[data-testid="stMetricValue"] {
    font-size: 27px;
    font-weight: 700;
}

hr {
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 10px;
}

.insight-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("Employee_HR.csv")


@st.cache_resource
def load_model():
    return joblib.load("employee_churn_model.pkl")


df = load_data()
model = load_model()


# =========================================================
# EMPLOYEE RISK SCORING
# =========================================================

@st.cache_data
def calculate_employee_risk(data):

    prediction_data = data.drop(
        columns=["EmpId", "Churn"],
        errors="ignore"
    ).copy()

    prediction_data = pd.get_dummies(
        prediction_data,
        drop_first=True
    )

    prediction_data = prediction_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    probabilities = model.predict_proba(
        prediction_data
    )[:, 1]

    result = data.copy()

    result["Churn Probability"] = probabilities * 100

    result["Risk Level"] = result[
        "Churn Probability"
    ].apply(
        lambda x:
        "🔴 High Risk" if x >= 70
        else "🟠 Medium Risk" if x >= 40
        else "🟢 Low Risk"
    )

    return result


# =========================================================
# GLOBAL CALCULATIONS
# =========================================================

total_employees = len(df)

employees_left = int(df["Churn"].sum())

employees_stayed = total_employees - employees_left

attrition_rate = df["Churn"].mean() * 100

avg_salary = df["Salary_INR"].mean()

avg_satisfaction = df["Satisfaction"].mean()

avg_evaluation = df["Evaluation"].mean()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("👥 HR Intelligence")

st.sidebar.caption("Employee Analytics & Churn Prediction")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Executive Dashboard",
        "🤖 Churn Prediction",
        "📊 HR Analytics",
        "🎯 Risk Center",
        "💡 Executive Insights",
        "📋 Employee Explorer"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "**Project:** HR Analytics & Employee Churn Prediction"
)

st.sidebar.markdown(
    "**Developed by:** Khemchand Prajapat"
)


# ==================================================================================================================================
# EXECUTIVE DASHBOARD
# =================================================================================================================================
if page == "🏠 Executive Dashboard":

    st.title("🏠 HR Intelligence Dashboard")

    st.caption(
    "Executive workforce analytics, employee attrition insights "
    "and machine-learning based churn intelligence."
)

    st.markdown(
    """
    <div style="
        padding: 12px 18px;
        border-radius: 10px;
        background-color: rgba(49, 51, 63, 0.45);
        margin-bottom: 20px;
    ">
        <b>📌 Executive View</b><br>
        Monitor workforce health, identify attrition patterns,
        and prioritize HR interventions using data-driven insights.
    </div>
    """,
    unsafe_allow_html=True

     )
    

    st.subheader("📊 Workforce Overview")

    st.caption(
    "Key workforce indicators providing a high-level view of organizational health."
     )

    # =====================================================
    # KPI ROW 1
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Employees",
        f"{total_employees:,}"
    )

    c2.metric(
        "Employees Left",
        f"{employees_left:,}"
    )

    c3.metric(
        "Employees Stayed",
        f"{employees_stayed:,}"
    )

    c4.metric(
        "Attrition Rate",
        f"{attrition_rate:.2f}%"
    )

    st.markdown("")

    # =====================================================
    # KPI ROW 2
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average Salary",
        f"₹{avg_salary:,.0f}"
    )

    c2.metric(
        "Avg Satisfaction",
        f"{avg_satisfaction:.2f}/10"
    )

    c3.metric(
        "Avg Evaluation",
        f"{avg_evaluation:.2f}/10"
    )

    c4.metric(
        "Departments",
        df["Department"].nunique()
    )

    st.markdown("---")

    # =====================================================
    # CHURN PIE + DEPARTMENT
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Employee Status")

        churn_df = pd.DataFrame({
            "Status": ["Stayed", "Left"],
            "Employees": [
                employees_stayed,
                employees_left
            ]
        })

        fig = px.pie(
            churn_df,
            names="Status",
            values="Employees",
            hole=0.55
        )

        fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            legend_title_text=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Workforce by Department")

        dept_df = (
            df["Department"]
            .value_counts()
            .reset_index()
        )

        dept_df.columns = [
            "Department",
            "Employees"
        ]

        fig = px.bar(
            dept_df,
            x="Employees",
            y="Department",
            orientation="h",
            text="Employees"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
    "Sales represents the largest workforce segment, followed by Technical and Support departments."
)
         
        
    # =====================================================
    # DEPARTMENT ATTRITION
    # =====================================================

    st.subheader("📉 Attrition Rate by Department")

    st.caption(
        "Department-wise employee attrition comparison to identify "
        "areas requiring HR attention."
    )

    dept_attrition = (
        df.groupby("Department")["Churn"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index(name="Attrition Rate")
        .sort_values(
            "Attrition Rate",
            ascending=False
        )
    )

    # Create readable department names
    dept_attrition["Department"] = (
        dept_attrition["Department"]
        .str.replace("_", " ")
        .str.title()
    )

    fig = px.bar(
        dept_attrition,
        x="Department",
        y="Attrition Rate",
        text="Attrition Rate",
        hover_data={
            "Department": True,
            "Attrition Rate": ":.2f"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Attrition Rate: %{y:.2f}%"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title={
            "text": "Department-wise Attrition",
            "x": 0.5
        },
        xaxis_title="Department",
        yaxis_title="Attrition Rate (%)",
        yaxis=dict(
            ticksuffix="%"
        ),
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20
        ),
        font=dict(
            size=13
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
    "Higher attrition indicates departments requiring stronger retention and employee-engagement strategies."
         )


    # =====================================================
    # MANAGEMENT INSIGHTS
    # =====================================================

    st.subheader("💡 Key HR Insights")

    highest_department = dept_attrition.iloc[0]

    lowest_department = dept_attrition.iloc[-1]

    high_hours = df[
        df["average_montly_hours"] > 250
    ]

    high_hours_attrition = (
        high_hours["Churn"].mean() * 100
        if len(high_hours) > 0
        else 0
    )

    low_satisfaction = df[
        df["Satisfaction"] < 4
    ]

    low_satisfaction_attrition = (
        low_satisfaction["Churn"].mean() * 100
        if len(low_satisfaction) > 0
        else 0
    )

        # =====================================================
    # MANAGEMENT ACTION SUMMARY
    # =====================================================

    st.markdown("---")

    st.subheader("🎯 Management Action Summary")

    st.caption(
        "Recommended actions based on workforce attrition, "
        "employee satisfaction and workload indicators."
    )

    a1, a2 = st.columns(2)

    with a1:

        st.error(
            f"""
            ### 🔴 Priority Action

            **Focus on {highest_department['Department']} department**

            This department has the highest attrition rate of
            **{highest_department['Attrition Rate']:.2f}%**.

            **Recommended Action:**
            Conduct retention discussions and identify the major
            causes of employee turnover.
            """
        )

        st.warning(
            f"""
            ### 🟡 Satisfaction Improvement

            Employees with satisfaction below **4/10** have an
            attrition rate of **{low_satisfaction_attrition:.2f}%**.

            **Recommended Action:**
            Improve employee engagement, feedback and satisfaction.
            """
        )

    with a2:

        st.warning(
            f"""
            ### 🟡 Workload Review

            Employees working more than **250 hours/month** have an
            attrition rate of **{high_hours_attrition:.2f}%**.

            **Recommended Action:**
            Review workload distribution and working-hour pressure.
            """
        )

        st.success(
            f"""
            ### 🟢 Workforce Monitoring

            **{len(df) - int(df["Churn"].sum()):,} employees**
            currently remain with the organization.

            **Recommended Action:**
            Continue regular engagement, performance reviews and
            early-risk monitoring.
            """
        )

    st.markdown("---")

    st.info(
        "💡 **Management Focus:** Use attrition trends, employee "
        "satisfaction, workload indicators and ML churn-risk scores "
        "together when making workforce decisions."
    )

    st.markdown("---")

    st.success(
    "✅ **Executive Takeaway:** "
    "The dashboard combines workforce metrics, attrition analytics "
    "and ML-based risk intelligence to support proactive HR decision-making."
)

     
    # =====================================================
    # INSIGHT CARDS
    # =====================================================

    i1, i2 = st.columns(2)

    with i1:

        st.warning(
            f"🏢 **Highest Attrition Department**\n\n"
            f"**{highest_department['Department']}**\n\n"
            f"Attrition Rate: "
            f"**{highest_department['Attrition Rate']:.2f}%**"
        )

        st.warning(
            f"😊 **Low Satisfaction Risk**\n\n"
            f"Employees with satisfaction below 4/10 "
            f"have an attrition rate of "
            f"**{low_satisfaction_attrition:.2f}%**."
        )


    with i2:

        st.success(
            f"🏢 **Lowest Attrition Department**\n\n"
            f"**{lowest_department['Department']}**\n\n"
            f"Attrition Rate: "
            f"**{lowest_department['Attrition Rate']:.2f}%**"
        )

        st.warning(
            f"⏱️ **High Workload Risk**\n\n"
            f"Employees working over 250 hours/month "
            f"have an attrition rate of "
            f"**{high_hours_attrition:.2f}%**."
        )


# =========================================================
# CHURN PREDICTION
# =========================================================

elif page == "🤖 Churn Prediction":

    st.title("🤖 Employee Churn Prediction")

    st.caption(
        "Use employee information to estimate the probability of churn."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    # =====================================================
    # EMPLOYEE PROFILE
    # =====================================================

    with col1:

        st.subheader("👤 Employee Profile")

        satisfaction = st.slider(
            "Satisfaction",
            0.0,
            10.0,
            5.0,
            0.1
        )

        evaluation = st.slider(
            "Evaluation",
            0.0,
            10.0,
            7.0,
            0.1
        )

        projects = st.number_input(
            "Number of Projects",
            min_value=int(df["number_of_projects"].min()),
            max_value=int(df["number_of_projects"].max()),
            value=int(df["number_of_projects"].median())
        )

        hours = st.number_input(
            "Average Monthly Hours",
            min_value=int(df["average_montly_hours"].min()),
            max_value=int(df["average_montly_hours"].max()),
            value=int(df["average_montly_hours"].median())
        )

    # =====================================================
    # JOB PROFILE
    # =====================================================

    with col2:

        st.subheader("💼 Job Profile")

        experience = st.number_input(
            "Years at Company",
            min_value=int(df["time_spent_company"].min()),
            max_value=int(df["time_spent_company"].max()),
            value=int(df["time_spent_company"].median())
        )

        accident = st.selectbox(
            "Work Accident",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        promotion = st.selectbox(
            "Promotion",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        department = st.selectbox(
            "Department",
            sorted(df["Department"].unique())
        )

        salary = st.number_input(
            "Salary (₹)",
            min_value=int(df["Salary_INR"].min()),
            max_value=int(df["Salary_INR"].max()),
            value=int(df["Salary_INR"].median()),
            step=1000
        )

    st.markdown("---")

    # =====================================================
    # PREDICT
    # =====================================================

    if st.button(
        "🔮 Predict Churn Risk",
        type="primary",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "Satisfaction": [satisfaction],
            "Evaluation": [evaluation],
            "number_of_projects": [projects],
            "average_montly_hours": [hours],
            "time_spent_company": [experience],
            "work_accident": [accident],
            "Promotion": [promotion],
            "Department": [department],
            "Salary_INR": [salary]
        })

        input_data = pd.get_dummies(
            input_data,
            drop_first=True
        )

        input_data = input_data.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        risk_percentage = probability * 100

        # =================================================
        # RISK LEVEL
        # =================================================

        if risk_percentage >= 70:
            risk_level = "HIGH RISK"
            risk_icon = "🔴"

        elif risk_percentage >= 40:
            risk_level = "MEDIUM RISK"
            risk_icon = "🟠"

        else:
            risk_level = "LOW RISK"
            risk_icon = "🟢"

        st.markdown("---")

        st.subheader("🎯 Prediction Result")

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Churn Probability",
            f"{risk_percentage:.2f}%"
        )

        r2.metric(
            "Risk Level",
            f"{risk_icon} {risk_level}"
        )

        r3.metric(
            "Prediction",
            "Will Leave"
            if prediction == 1
            else "Will Stay"
        )

        st.progress(
            min(probability, 1.0)
        )

        # =================================================
        # RESULT MESSAGE
        # =================================================

        if prediction == 1:

            st.error(
                "⚠️ Employee is predicted to have a higher "
                "likelihood of leaving the organization."
            )

        else:

            st.success(
                "✅ Employee is predicted to have a higher "
                "likelihood of staying with the organization."
            )

        # =================================================
        # RECOMMENDATIONS
        # =================================================

        st.subheader("💡 Recommended HR Actions")

        recommendations = []

        if satisfaction < 4:
            recommendations.append(
                "Conduct an employee satisfaction discussion."
            )

        if hours > 250:
            recommendations.append(
                "Review workload and working-hour pressure."
            )

        if projects >= 6:
            recommendations.append(
                "Consider redistributing project workload."
            )

        if promotion == 0 and experience >= 5:
            recommendations.append(
                "Review career progression and promotion opportunities."
            )

        if salary < df["Salary_INR"].median():
            recommendations.append(
                "Review compensation against comparable employees."
            )

        if not recommendations:
            recommendations.append(
                "Continue regular employee engagement and performance reviews."
            )

        for item in recommendations:
            st.write("•", item)


# =========================================================
# HR ANALYTICS
# =========================================================

elif page == "📊 HR Analytics":

    st.title("📊 Advanced HR Analytics")

    st.caption(
        "Explore employee attrition patterns and workforce behavior."
    )

    st.markdown("---")

    # =====================================================
    # FILTERS
    # =====================================================

    st.subheader("🔎 Analytics Filters")

    f1, f2, f3 = st.columns(3)

    with f1:

        selected_departments = st.multiselect(
            "Department",
            sorted(df["Department"].unique())
        )

    with f2:

        selected_status = st.multiselect(
            "Employee Status",
            [0, 1],
            format_func=lambda x:
                "Stayed" if x == 0 else "Left"
        )

    with f3:

        satisfaction_filter = st.slider(
            "Minimum Satisfaction",
            0.0,
            10.0,
            0.0,
            0.1
        )

    filtered_df = df.copy()

    if selected_departments:

        filtered_df = filtered_df[
            filtered_df["Department"].isin(
                selected_departments
            )
        ]

    if selected_status:

        filtered_df = filtered_df[
            filtered_df["Churn"].isin(
                selected_status
            )
        ]

    filtered_df = filtered_df[
        filtered_df["Satisfaction"] >= satisfaction_filter
    ]

    # =====================================================
    # FILTERED KPIs
    # =====================================================

    st.markdown("---")

    if len(filtered_df) > 0:

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Filtered Employees",
            f"{len(filtered_df):,}"
        )

        k2.metric(
            "Employees Left",
            f"{int(filtered_df['Churn'].sum()):,}"
        )

        k3.metric(
            "Attrition Rate",
            f"{filtered_df['Churn'].mean() * 100:.2f}%"
        )

        k4.metric(
            "Average Salary",
            f"₹{filtered_df['Salary_INR'].mean():,.0f}"
        )

    st.markdown("---")

    # =====================================================
    # SATISFACTION VS CHURN
    # =====================================================

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("😊 Satisfaction vs Attrition")

        satisfaction_df = (
            filtered_df.groupby("Satisfaction")["Churn"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index(name="Attrition Rate")
        )

        fig = px.line(
            satisfaction_df,
            x="Satisfaction",
            y="Attrition Rate",
            markers=True
        )

        fig.update_layout(
            yaxis_title="Attrition Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        st.subheader("📊 Evaluation vs Attrition")

        evaluation_df = (
            filtered_df.groupby("Evaluation")["Churn"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index(name="Attrition Rate")
        )

        fig = px.line(
            evaluation_df,
            x="Evaluation",
            y="Attrition Rate",
            markers=True
        )

        fig.update_layout(
            yaxis_title="Attrition Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # PROJECTS VS CHURN
    # =====================================================

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("📁 Projects vs Attrition")

        projects_df = (
            filtered_df.groupby("number_of_projects")["Churn"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index(name="Attrition Rate")
        )

        fig = px.bar(
            projects_df,
            x="number_of_projects",
            y="Attrition Rate",
            text="Attrition Rate"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        st.subheader("⏱️ Monthly Hours vs Attrition")

        hours_df = (
            filtered_df.groupby("average_montly_hours")["Churn"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index(name="Attrition Rate")
        )

        fig = px.scatter(
            hours_df,
            x="average_montly_hours",
            y="Attrition Rate"
        )

        fig.update_layout(
            xaxis_title="Average Monthly Hours",
            yaxis_title="Attrition Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # EXPERIENCE VS CHURN
    # =====================================================

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("⌛ Experience vs Attrition")

        experience_df = (
            filtered_df.groupby("time_spent_company")["Churn"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index(name="Attrition Rate")
        )

        fig = px.bar(
            experience_df,
            x="time_spent_company",
            y="Attrition Rate",
            text="Attrition Rate"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        st.subheader("💰 Salary vs Attrition")

        salary_bins = pd.cut(
            filtered_df["Salary_INR"],
            bins=5
        )

        salary_df = (
            filtered_df.groupby(
                salary_bins,
                observed=True
            )["Churn"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index(name="Attrition Rate")
        )

        salary_df["Salary Range"] = salary_df[
            "Salary_INR"
        ].apply(
            lambda x:
            f"₹{x.left:,.0f} - ₹{x.right:,.0f}"
        )

        fig = px.bar(
            salary_df,
            x="Salary Range",
            y="Attrition Rate",
            text="Attrition Rate"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # =====================================================
    # PROMOTION VS CHURN
    # =====================================================

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("📈 Promotion vs Attrition")

        promotion_df = (
            filtered_df.groupby("Promotion")["Churn"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index(name="Attrition Rate")
        )

        promotion_df["Status"] = promotion_df[
            "Promotion"
        ].map({
            0: "No Promotion",
            1: "Promoted"
        })

        fig = px.bar(
            promotion_df,
            x="Status",
            y="Attrition Rate",
            text="Attrition Rate"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="Attrition Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        st.subheader("⚠️ Work Accident vs Attrition")

        accident_df = (
            filtered_df.groupby("work_accident")["Churn"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index(name="Attrition Rate")
        )

        accident_df["Status"] = accident_df[
            "work_accident"
        ].map({
            0: "No Accident",
            1: "Accident"
        })

        fig = px.bar(
            accident_df,
            x="Status",
            y="Attrition Rate",
            text="Attrition Rate"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="Attrition Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# RISK CENTER
# =========================================================

elif page == "🎯 Risk Center":

    st.title("🎯 Employee Risk Center")

    st.caption(
        "ML-powered employee churn risk analysis and HR intervention center."
    )

    st.markdown("---")

    # =====================================================
    # CALCULATE RISK
    # =====================================================

    with st.spinner("Analyzing employee churn risk..."):

        risk_df = calculate_employee_risk(df)


            # =====================================================
    # INDIVIDUAL EMPLOYEE RISK PROFILE
    # =====================================================

    st.markdown("---")

    st.subheader("🔍 Individual Employee Risk Profile")

    st.caption(
        "Enter an Employee ID to view the employee's "
        "ML-based churn risk profile."
    )

    employee_id = st.number_input(
        "Employee ID",
        min_value=int(df["EmpId"].min()),
        max_value=int(df["EmpId"].max()),
        value=int(df["EmpId"].iloc[0]),
        step=1
    )

    if st.button(
        "🔎 Analyze Employee",
        type="primary",
        use_container_width=True
    ):

        employee_profile = risk_df[
            risk_df["EmpId"] == employee_id
        ]

        if employee_profile.empty:

            st.error(
                f"❌ Employee ID {employee_id} was not found."
            )

        else:

            employee = employee_profile.iloc[0]

            probability = employee[
                "Churn Probability"
            ]

            risk_level = employee[
                "Risk Level"
            ]

            st.markdown("---")

            # =================================================
            # EMPLOYEE HEADER
            # =================================================

            st.subheader(
                f"👤 Employee {int(employee['EmpId'])}"
            )

            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "Department",
                str(employee["Department"])
            )

            p2.metric(
                "Churn Probability",
                f"{probability:.2f}%"
            )

            p3.metric(
                "Risk Level",
                risk_level
            )

            p4.metric(
                "Current Status",
                "Previously Left"
                if employee["Churn"] == 1
                else "Currently Stayed"
            )

            st.markdown("")

            # =================================================
            # RISK PROGRESS
            # =================================================

            st.write("### 🎯 Churn Risk Score")

            st.progress(
                min(probability / 100, 1.0)
            )

            # =================================================
            # EMPLOYEE DETAILS
            # =================================================

            st.markdown("---")

            st.subheader("📋 Employee Details")

            d1, d2, d3, d4 = st.columns(4)

            d1.metric(
                "Satisfaction",
                f"{employee['Satisfaction']:.1f}/10"
            )

            d2.metric(
                "Evaluation",
                f"{employee['Evaluation']:.1f}/10"
            )

            d3.metric(
                "Projects",
                int(employee["number_of_projects"])
            )

            d4.metric(
                "Monthly Hours",
                int(employee["average_montly_hours"])
            )

            d1, d2, d3, d4 = st.columns(4)

            d1.metric(
                "Years at Company",
                int(employee["time_spent_company"])
            )

            d2.metric(
                "Promotion",
                "Yes"
                if employee["Promotion"] == 1
                else "No"
            )

            d3.metric(
                "Work Accident",
                "Yes"
                if employee["work_accident"] == 1
                else "No"
            )

            d4.metric(
                "Salary",
                f"₹{employee['Salary_INR']:,.0f}"
            )

            # =================================================
            # POTENTIAL RISK INDICATORS
            # =================================================

            st.markdown("---")

            st.subheader("⚠️ Potential Risk Indicators")

            indicators = []

            median_salary = df["Salary_INR"].median()

            if employee["Satisfaction"] < 4:

                indicators.append(
                    "Low employee satisfaction"
                )

            if employee["average_montly_hours"] > 250:

                indicators.append(
                    "High monthly workload"
                )

            if employee["number_of_projects"] >= 6:

                indicators.append(
                    "High number of projects"
                )

            if (
                employee["time_spent_company"] >= 5
                and employee["Promotion"] == 0
            ):

                indicators.append(
                    "Long tenure without promotion"
                )

            if employee["Salary_INR"] < median_salary:

                indicators.append(
                    "Salary below dataset median"
                )

            if len(indicators) == 0:

                st.success(
                    "No major rule-based risk indicators "
                    "were identified."
                )

            else:

                for indicator in indicators:

                    st.warning(
                        f"• {indicator}"
                    )

            # =================================================
            # HR RECOMMENDATIONS
            # =================================================

            st.markdown("---")

            st.subheader("💡 Recommended HR Actions")

            recommendations = []

            if employee["Satisfaction"] < 4:

                recommendations.append(
                    "Conduct a satisfaction and engagement discussion."
                )

            if employee["average_montly_hours"] > 250:

                recommendations.append(
                    "Review workload and working-hour pressure."
                )

            if (
                employee["time_spent_company"] >= 5
                and employee["Promotion"] == 0
            ):

                recommendations.append(
                    "Discuss career progression and promotion opportunities."
                )

            if employee["Salary_INR"] < median_salary:

                recommendations.append(
                    "Review compensation against comparable employees."
                )

            if employee["number_of_projects"] >= 6:

                recommendations.append(
                    "Consider redistributing project responsibilities."
                )

            if not recommendations:

                recommendations.append(
                    "Continue regular engagement, recognition "
                    "and performance discussions."
                )

            for recommendation in recommendations:

                st.write(
                    f"✅ {recommendation}"
                )

                # =================================================
            # DEPARTMENT BENCHMARKING
            # =================================================

            st.markdown("---")

            st.subheader("📊 Department Benchmark")

            st.caption(
                "Compare this employee with the average employee "
                "in the same department."
            )

            employee_department = employee["Department"]

            department_data = df[
                df["Department"] == employee_department
            ]

            # Department averages
            department_avg_satisfaction = (
                department_data["Satisfaction"].mean()
            )

            department_avg_evaluation = (
                department_data["Evaluation"].mean()
            )

            department_avg_projects = (
                department_data["number_of_projects"].mean()
            )

            department_avg_hours = (
                department_data["average_montly_hours"].mean()
            )

            department_avg_salary = (
                department_data["Salary_INR"].mean()
            )

            department_avg_churn = (
                department_data["Churn"].mean() * 100
            )

            # =================================================
            # COMPARISON DATA
            # =================================================

            benchmark_df = pd.DataFrame({
                "Metric": [
                    "Satisfaction",
                    "Evaluation",
                    "Projects",
                    "Monthly Hours",
                    "Salary",
                    "Churn Rate"
                ],

                "Employee": [
                    employee["Satisfaction"],
                    employee["Evaluation"],
                    employee["number_of_projects"],
                    employee["average_montly_hours"],
                    employee["Salary_INR"],
                    probability
                ],

                "Department Average": [
                    department_avg_satisfaction,
                    department_avg_evaluation,
                    department_avg_projects,
                    department_avg_hours,
                    department_avg_salary,
                    department_avg_churn
                ]
            })

            # =================================================
            # DISPLAY COMPARISON TABLE
            # =================================================

            display_benchmark = benchmark_df.copy()

            display_benchmark["Employee"] = (
                display_benchmark["Employee"].round(2)
            )

            display_benchmark["Department Average"] = (
                display_benchmark["Department Average"].round(2)
            )

            st.dataframe(
                display_benchmark,
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # VISUAL COMPARISON
            # =================================================

            comparison_chart = benchmark_df.copy()

            comparison_chart = comparison_chart[
                comparison_chart["Metric"].isin([
                    "Satisfaction",
                    "Evaluation",
                    "Projects",
                    "Monthly Hours"
                ])
            ]

            comparison_chart = comparison_chart.melt(
                id_vars="Metric",
                value_vars=[
                    "Employee",
                    "Department Average"
                ],
                var_name="Comparison",
                value_name="Value"
            )

            fig = px.bar(
                comparison_chart,
                x="Metric",
                y="Value",
                color="Comparison",
                barmode="group",
                text="Value"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )

            fig.update_layout(
                title=f"Employee vs {employee_department} Department Average",
                xaxis_title="Metric",
                yaxis_title="Value"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # =================================================
            # BENCHMARK INSIGHTS
            # =================================================

            st.subheader("💡 Benchmark Insights")

            benchmark_insights = []

            if (
                employee["Satisfaction"]
                < department_avg_satisfaction
            ):
                benchmark_insights.append(
                    "Employee satisfaction is below the department average."
                )
            else:
                benchmark_insights.append(
                    "Employee satisfaction is at or above the department average."
                )

            if (
                employee["Evaluation"]
                < department_avg_evaluation
            ):
                benchmark_insights.append(
                    "Employee evaluation is below the department average."
                )
            else:
                benchmark_insights.append(
                    "Employee evaluation is at or above the department average."
                )

            if (
                employee["number_of_projects"]
                > department_avg_projects
            ):
                benchmark_insights.append(
                    "Employee has more projects than the department average."
                )

            if (
                employee["average_montly_hours"]
                > department_avg_hours
            ):
                benchmark_insights.append(
                    "Employee's monthly workload is above the department average."
                )

            if (
                employee["Salary_INR"]
                < department_avg_salary
            ):
                benchmark_insights.append(
                    "Employee salary is below the department average."
                )

            if not benchmark_insights:

                benchmark_insights.append(
                    "Employee metrics are generally aligned with "
                    "the department averages."
                )

            for insight in benchmark_insights:

                st.info(
                    f"• {insight}"
                )            

    # =====================================================
    # RISK COUNTS
    # =====================================================

    high_risk = risk_df[
        risk_df["Churn Probability"] >= 70
    ]

    medium_risk = risk_df[
        (risk_df["Churn Probability"] >= 40)
        & (risk_df["Churn Probability"] < 70)
    ]

    low_risk = risk_df[
        risk_df["Churn Probability"] < 40
    ]

    # =====================================================
    # KPI CARDS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🔴 High Risk",
        f"{len(high_risk):,}"
    )

    c2.metric(
        "🟠 Medium Risk",
        f"{len(medium_risk):,}"
    )

    c3.metric(
        "🟢 Low Risk",
        f"{len(low_risk):,}"
    )

    c4.metric(
        "Average Risk",
        f"{risk_df['Churn Probability'].mean():.2f}%"
    )

    st.markdown("---")

    # =====================================================
    # RISK DISTRIBUTION
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Employee Risk Distribution")

        risk_chart = pd.DataFrame({
            "Risk Level": [
                "High Risk",
                "Medium Risk",
                "Low Risk"
            ],
            "Employees": [
                len(high_risk),
                len(medium_risk),
                len(low_risk)
            ]
        })

        fig = px.pie(
            risk_chart,
            names="Risk Level",
            values="Employees",
            hole=0.5
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("📈 Risk Probability Distribution")

        fig = px.histogram(
            risk_df,
            x="Churn Probability",
            nbins=20,
            title="Employee Churn Probability"
        )

        fig.update_layout(
            xaxis_title="Churn Probability (%)",
            yaxis_title="Employees"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # =====================================================
    # RISK FILTER
    # =====================================================

    st.subheader("🔎 Risk Filters")

    f1, f2 = st.columns(2)

    with f1:

        risk_filter = st.multiselect(
            "Select Risk Level",
            [
                "🔴 High Risk",
                "🟠 Medium Risk",
                "🟢 Low Risk"
            ],
            default=["🔴 High Risk"]
        )

    with f2:

        department_filter = st.multiselect(
            "Select Department",
            sorted(df["Department"].unique())
        )

    filtered_risk = risk_df.copy()

    if risk_filter:

        filtered_risk = filtered_risk[
            filtered_risk["Risk Level"].isin(
                risk_filter
            )
        ]

    if department_filter:

        filtered_risk = filtered_risk[
            filtered_risk["Department"].isin(
                department_filter
            )
        ]

    # =====================================================
    # TOP AT-RISK EMPLOYEES
    # =====================================================

    st.markdown("---")

    st.subheader("🚨 Top At-Risk Employees")

    display_columns = [
        "EmpId",
        "Department",
        "Churn Probability",
        "Risk Level",
        "Satisfaction",
        "Evaluation",
        "number_of_projects",
        "average_montly_hours",
        "time_spent_company",
        "Promotion",
        "Salary_INR"
    ]

    top_risk = (
        filtered_risk[
            display_columns
        ]
        .sort_values(
            "Churn Probability",
            ascending=False
        )
        .head(20)
        .copy()
    )

    top_risk["Churn Probability"] = (
        top_risk["Churn Probability"]
        .round(2)
    )

    top_risk["Salary_INR"] = (
        top_risk["Salary_INR"]
        .round(0)
    )

    st.dataframe(
        top_risk,
        use_container_width=True,
        height=500
    )

    st.markdown("---")

    # =====================================================
    # HIGH RISK ANALYSIS
    # =====================================================

    st.subheader("🔴 High-Risk Employee Analysis")

    if len(high_risk) > 0:

        high_col1, high_col2, high_col3 = st.columns(3)

        high_col1.metric(
            "High-Risk Employees",
            f"{len(high_risk):,}"
        )

        high_col2.metric(
            "Avg Satisfaction",
            f"{high_risk['Satisfaction'].mean():.2f}"
        )

        high_col3.metric(
            "Avg Monthly Hours",
            f"{high_risk['average_montly_hours'].mean():.0f}"
        )

        st.markdown("")

        # Department concentration
        high_dept = (
            high_risk["Department"]
            .value_counts()
            .reset_index()
        )

        high_dept.columns = [
            "Department",
            "High Risk Employees"
        ]

        fig = px.bar(
            high_dept,
            x="Department",
            y="High Risk Employees",
            text="High Risk Employees",
            title="High-Risk Employees by Department"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.success(
            "No employees currently fall into the high-risk category."
        )

    # =====================================================
    # HR ACTION CENTER
    # =====================================================

    st.markdown("---")

    st.subheader("💡 HR Action Center")

    action1, action2, action3 = st.columns(3)

    with action1:

        st.error(
            "### 🔴 High Risk\n\n"
            "**Immediate Intervention**\n\n"
            "• Conduct retention discussion\n"
            "• Review workload\n"
            "• Check career growth\n"
            "• Review compensation"
        )

    with action2:

        st.warning(
            "### 🟠 Medium Risk\n\n"
            "**Monitor & Engage**\n\n"
            "• Schedule regular check-ins\n"
            "• Monitor satisfaction\n"
            "• Discuss career goals\n"
            "• Track workload"
        )

    with action3:

        st.success(
            "### 🟢 Low Risk\n\n"
            "**Maintain Engagement**\n\n"
            "• Continue recognition\n"
            "• Maintain engagement\n"
            "• Encourage development\n"
            "• Regular feedback"
        )

    # =====================================================
    # DOWNLOAD HIGH RISK DATA
    # =====================================================

    st.markdown("---")

    high_risk_download = high_risk.copy()

    csv = high_risk_download.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download High-Risk Employee Report",
        data=csv,
        file_name="high_risk_employees.csv",
        mime="text/csv",
        use_container_width=True
    )




# =========================================================
# EXECUTIVE INSIGHTS
# =========================================================

elif page == "💡 Executive Insights":

    st.title("💡 Executive HR Insights")

    st.caption(
        "Automated management insights generated from HR analytics "
        "and employee churn risk data."
    )

    st.markdown("---")

    # =====================================================
    # BASIC METRICS
    # =====================================================

    total_employees = len(df)

    employees_left = int(df["Churn"].sum())

    attrition_rate = (
        df["Churn"].mean() * 100
    )

    avg_satisfaction = (
        df["Satisfaction"].mean()
    )

    avg_salary = (
        df["Salary_INR"].mean()
    )

    avg_hours = (
        df["average_montly_hours"].mean()
    )

    # =====================================================
    # ML RISK DATA
    # =====================================================

    with st.spinner("Generating executive insights..."):

        risk_df = calculate_employee_risk(df)

    high_risk_count = len(
        risk_df[
            risk_df["Churn Probability"] >= 70
        ]
    )

    medium_risk_count = len(
        risk_df[
            (risk_df["Churn Probability"] >= 40)
            & (risk_df["Churn Probability"] < 70)
        ]
    )

    low_risk_count = len(
        risk_df[
            risk_df["Churn Probability"] < 40
        ]
    )

    avg_risk = (
        risk_df["Churn Probability"].mean()
    )

    # =====================================================
    # EXECUTIVE KPI CARDS
    # =====================================================

    st.subheader("📊 Executive Summary")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Total Employees",
        f"{total_employees:,}"
    )

    k2.metric(
        "Attrition Rate",
        f"{attrition_rate:.2f}%"
    )

    k3.metric(
        "High-Risk Employees",
        f"{high_risk_count:,}"
    )

    k4.metric(
        "Average Churn Risk",
        f"{avg_risk:.2f}%"
    )

    st.markdown("---")

    # =====================================================
    # TOP HR ISSUES
    # =====================================================

    st.subheader("🚨 Top HR Issues")

    issues = []

    # Satisfaction
    if avg_satisfaction < 5:

        issues.append(
            (
                "🔴",
                "Low Employee Satisfaction",
                f"Average satisfaction is only "
                f"{avg_satisfaction:.2f}/10."
            )
        )

    else:

        issues.append(
            (
                "🟢",
                "Employee Satisfaction",
                f"Average satisfaction is "
                f"{avg_satisfaction:.2f}/10."
            )
        )

    # Attrition
    if attrition_rate >= 15:

        issues.append(
            (
                "🔴",
                "High Attrition",
                f"Overall attrition rate is "
                f"{attrition_rate:.2f}%."
            )
        )

    else:

        issues.append(
            (
                "🟢",
                "Attrition",
                f"Overall attrition rate is "
                f"{attrition_rate:.2f}%."
            )
        )

    # Workload
    if avg_hours > 220:

        issues.append(
            (
                "🔴",
                "High Workload",
                f"Average monthly working hours are "
                f"{avg_hours:.0f}."
            )
        )

    else:

        issues.append(
            (
                "🟢",
                "Workload",
                f"Average monthly working hours are "
                f"{avg_hours:.0f}."
            )
        )

    # High risk
    high_risk_percentage = (
        high_risk_count / total_employees * 100
    )

    if high_risk_percentage >= 10:

        issues.append(
            (
                "🔴",
                "High Churn Risk Population",
                f"{high_risk_percentage:.2f}% of employees "
                f"are classified as high risk."
            )
        )

    else:

        issues.append(
            (
                "🟢",
                "Churn Risk Population",
                f"{high_risk_percentage:.2f}% of employees "
                f"are classified as high risk."
            )
        )

    # =====================================================
    # DISPLAY ISSUES
    # =====================================================

    for icon, title, description in issues:

        st.info(
            f"{icon} **{title}**  \n"
            f"{description}"
        )

    st.markdown("---")

    # =====================================================
    # DEPARTMENT INSIGHTS
    # =====================================================

    st.subheader("🏢 Department Insights")

    department_analysis = (
        df.groupby("Department")
        .agg(
            Employees=("EmpId", "count"),
            Attrition_Rate=("Churn", "mean"),
            Avg_Satisfaction=("Satisfaction", "mean"),
            Avg_Salary=("Salary_INR", "mean"),
            Avg_Hours=("average_montly_hours", "mean")
        )
        .reset_index()
    )

    department_analysis[
        "Attrition_Rate"
    ] *= 100

    department_analysis[
        "Attrition_Rate"
    ] = department_analysis[
        "Attrition_Rate"
    ].round(2)

    department_analysis[
        "Avg_Satisfaction"
    ] = department_analysis[
        "Avg_Satisfaction"
    ].round(2)

    department_analysis[
        "Avg_Salary"
    ] = department_analysis[
        "Avg_Salary"
    ].round(0)

    department_analysis[
        "Avg_Hours"
    ] = department_analysis[
        "Avg_Hours"
    ].round(0)

    highest_attrition_department = (
        department_analysis
        .sort_values(
            "Attrition_Rate",
            ascending=False
        )
        .iloc[0]
    )

    lowest_satisfaction_department = (
        department_analysis
        .sort_values(
            "Avg_Satisfaction"
        )
        .iloc[0]
    )

    highest_workload_department = (
        department_analysis
        .sort_values(
            "Avg_Hours",
            ascending=False
        )
        .iloc[0]
    )

    d1, d2, d3 = st.columns(3)

    with d1:

        st.warning(
            f"🏢 **Highest Attrition**\n\n"
            f"{highest_attrition_department['Department']}\n\n"
            f"Attrition: "
            f"{highest_attrition_department['Attrition_Rate']:.2f}%"
        )

    with d2:

        st.warning(
            f"😊 **Lowest Satisfaction**\n\n"
            f"{lowest_satisfaction_department['Department']}\n\n"
            f"Satisfaction: "
            f"{lowest_satisfaction_department['Avg_Satisfaction']:.2f}/10"
        )

    with d3:

        st.warning(
            f"⏱️ **Highest Workload**\n\n"
            f"{highest_workload_department['Department']}\n\n"
            f"Hours: "
            f"{highest_workload_department['Avg_Hours']:.0f}"
        )

    st.markdown("---")

    # =====================================================
    # MANAGEMENT RECOMMENDATIONS
    # =====================================================

    st.subheader("🎯 Management Recommendations")

    recommendations = [
        (
            "1️⃣",
            "Focus on high-risk employees",
            f"Prioritize the {high_risk_count:,} employees "
            "with churn probability above 70%."
        ),

        (
            "2️⃣",
            "Improve employee engagement",
            f"Review employees with satisfaction below "
            f"4/10. Current overall satisfaction is "
            f"{avg_satisfaction:.2f}/10."
        ),

        (
            "3️⃣",
            "Review workload",
            f"Monitor departments and employees with "
            f"high monthly working hours. Overall average "
            f"is {avg_hours:.0f} hours."
        ),

        (
            "4️⃣",
            "Review department-specific issues",
            f"The {highest_attrition_department['Department']} "
            "department currently has the highest attrition rate."
        ),

        (
            "5️⃣",
            "Use ML risk scores for prioritization",
            "Combine churn probability with employee-level "
            "risk indicators before taking HR action."
        )
    ]

    for number, title, description in recommendations:

        st.success(
            f"{number} **{title}**  \n"
            f"{description}"
        )

    st.markdown("---")

    # =====================================================
    # DEPARTMENT TABLE
    # =====================================================

    st.subheader("📋 Department Performance Summary")

    display_department = department_analysis.copy()

    display_department.columns = [
        "Department",
        "Employees",
        "Attrition Rate (%)",
        "Avg Satisfaction",
        "Avg Salary (₹)",
        "Avg Monthly Hours"
    ]

    st.dataframe(
        display_department,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # =====================================================
    # EXECUTIVE REPORT
    # =====================================================

    st.subheader("📄 Executive HR Report")

    report_text = f"""
HR ANALYTICS — EXECUTIVE REPORT
================================

WORKFORCE SUMMARY
-----------------
Total Employees: {total_employees:,}
Employees Left: {employees_left:,}
Attrition Rate: {attrition_rate:.2f}%
Average Satisfaction: {avg_satisfaction:.2f}/10
Average Salary: ₹{avg_salary:,.0f}
Average Monthly Hours: {avg_hours:.0f}

ML CHURN RISK
-------------
High Risk Employees: {high_risk_count:,}
Medium Risk Employees: {medium_risk_count:,}
Low Risk Employees: {low_risk_count:,}
Average Churn Risk: {avg_risk:.2f}%

KEY DEPARTMENT FINDINGS
-----------------------
Highest Attrition Department:
{highest_attrition_department['Department']}
Attrition Rate:
{highest_attrition_department['Attrition_Rate']:.2f}%

Lowest Satisfaction Department:
{lowest_satisfaction_department['Department']}
Average Satisfaction:
{lowest_satisfaction_department['Avg_Satisfaction']:.2f}/10

Highest Workload Department:
{highest_workload_department['Department']}
Average Monthly Hours:
{highest_workload_department['Avg_Hours']:.0f}

MANAGEMENT RECOMMENDATIONS
--------------------------
1. Prioritize high-risk employees for retention actions.
2. Improve employee engagement and satisfaction.
3. Review workload in high-hour departments.
4. Investigate departments with high attrition.
5. Use ML risk scores together with HR indicators.

Generated by:
HR Analytics & Employee Churn Intelligence
Developed by Khemchand Prajapat
"""

    st.download_button(
        "⬇️ Download Executive HR Report",
        data=report_text,
        file_name="HR_Executive_Report.txt",
        mime="text/plain",
        use_container_width=True
    )

# =========================================================
# EMPLOYEE EXPLORER
# =========================================================

elif page == "📋 Employee Explorer":

    st.title("📋 Employee Data Explorer")

    st.caption(
        "Filter, analyze and download employee records."
    )

    st.markdown("---")

    f1, f2, f3 = st.columns(3)

    with f1:

        departments = st.multiselect(
            "Department",
            sorted(df["Department"].unique())
        )

    with f2:

        status = st.multiselect(
            "Employee Status",
            [0, 1],
            format_func=lambda x:
                "Stayed" if x == 0 else "Left"
        )

    with f3:

        min_satisfaction = st.slider(
            "Minimum Satisfaction",
            0.0,
            10.0,
            0.0,
            0.1
        )

    result = df.copy()

    if departments:

        result = result[
            result["Department"].isin(departments)
        ]

    if status:

        result = result[
            result["Churn"].isin(status)
        ]

    result = result[
        result["Satisfaction"] >= min_satisfaction
    ]

    st.markdown("---")

    st.write(
        f"### {len(result):,} Employees Found"
    )

    st.dataframe(
        result,
        use_container_width=True,
        height=520
    )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    csv = result.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Employee Data",
        data=csv,
        file_name="HR_filtered_data.csv",
        mime="text/csv",
        use_container_width=True
    )