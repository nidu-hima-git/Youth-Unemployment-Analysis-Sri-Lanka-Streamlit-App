import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.metrics import roc_curve, roc_auc_score

st.set_page_config(page_title="Youth Unemployment App", layout="wide")

st.markdown("""
<div style='
    background-color:#D6EAF8;
    padding:0px;
    border-radius:12px;
    text-align:center;
    margin-bottom:30px;
'>
    <h1 style='
        color:black;
        margin:0;
        font-weight:700;
        font-size:38px;
    '>
        Youth Unemployment Analysis in Sri Lanka
    </h1>
</div>
""", unsafe_allow_html=True)



#####Navigation#####
from streamlit_option_menu import option_menu

st.sidebar.title("Youth Unemployment App")
st.sidebar.markdown("---")

with st.sidebar:
    menu = option_menu(
        menu_title="Navigation",
        options=[
            "Home",
            "Data Explorer",
            "Exploratory Analysis",
            "Hypothesis Testing",
            "Logistic Regression",
            "Conclusion"
        ],
        icons=[
            "house",
            "table",
            "bar-chart",
            "clipboard-data",
            "graph-up",
            "check-circle"
        ],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5!important", "background-color": "#f8f9fa"},
            "icon": {"color": "#1f77b4", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#e9ecef",
            },
            "nav-link-selected": {"background-color": "#1f77b4", "color": "white"},
        }
    )


#####Load dataset#####
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\s16905\\unemp_youth.csv")
    return df

df = load_data()

df["Q4"] = df["Q4"].fillna("Not Applicable")
df["Q6_A"] = df["Q6_A"].fillna("Not Applicable")
df["Q47"] = df["Q47"].fillna("Not Applicable")
df["Q48"] = df["Q48"].fillna("Not Applicable")
df["Q50"] = df["Q50"].fillna("Not Applicable")
df["Q51"] = df["Q51"].fillna("Not Applicable")
df["Q52"] = df["Q52"].fillna("Not Applicable")
df.head()

df = df.drop(columns=["BMONTH"]) #The variable “bmonth” was initially retained for duplicate identification purposes. After duplicate removal, it was excluded from further analysis as it was not relevant to the study objectives. Therefore, its missing values were not considered in the final analysis.



#####About#####
if menu == "Home":

    st.markdown("## Overview")

    st.markdown("""
    This interactive application analyses **Youth Unemployment in Sri Lanka (Ages 15–29)** 
    using statistical inference and predictive modelling techniques.
    """)

    st.divider()


    st.markdown("###  Key Indicators")

    total_obs = df.shape[0]
    total_vars = df.shape[1]
    unemployed_count = len(df[df["employment_status"] == "Unemployed"])
    employed_count = len(df[df["employment_status"] == "Employed"])
    unemployment_rate = unemployed_count / total_obs * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Youth Observed", total_obs)
    col2.metric("Total Variables", total_vars)
    col3.metric("Total Unemployed", unemployed_count)
    col4.metric("Unemployment Rate", f"{unemployment_rate:.1f}%")

    st.divider()


# What This App Does

    st.markdown("###  What This Application Provides")

    st.markdown("""
    ✔ Interactive data exploration with filters  
    ✔ Visual analysis of unemployment patterns  
    ✔ Statistical hypothesis testing  
    ✔ Logistic regression prediction model  
    ✔ Policy-relevant interpretation in simple language  
    """)

    st.divider()


# Data Source

    st.markdown("###  Data Source")

    st.info("""
    Sri Lanka Labour Force Survey (LFS) – 2023  
    Department of Census and Statistics (DCS), Sri Lanka
    """)

    st.divider()

 
# Definitions Section

    st.markdown("###  Key Definitions")

    st.markdown("""
    **Youth:** Individuals aged 15–29 years.  

    **Unemployed:** Persons actively seeking work and available for employment.  

    **Labour Force:** Economically active population (employed + unemployed).
    """)



#####Data Explorer#####
if menu == "Data Explorer":

    st.header("Data Explorer")
    st.subheader("Data & Preprocessing")

    st.info("""
    Data preprocessing including handling missing values, handling duplicates, variable recoding, 
    and feature engineering (EDU_group and PROVINCE creation) 
    was performed in Jupyter Notebook.

    The cleaned dataset was exported and loaded into this Streamlit application.

    Additional preprocessing such as encoding and dummy variable creation 
    is performed dynamically within the application for modelling purposes.
    """)


    st.divider()

    st.markdown("### Data Quality Check")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###  Missing Values")
        st.dataframe(df.isnull().sum())

    with col2:
        st.markdown("###  Data Types")
        st.dataframe(df.dtypes)

    st.divider()

    st.markdown("## Filter Data")

    sector_filter = st.selectbox("Select Sector", df["SECTOR"].unique())
    sex_filter = st.selectbox("Select Sex", df["SEX"].unique())

    filtered_df = df[(df["SECTOR"] == sector_filter) &
                     (df["SEX"] == sex_filter)]

    st.dataframe(filtered_df.head())




####Exploratory Analysis
if menu == "Exploratory Analysis":

    st.header(" Exploratory Analysis")
    
    st.subheader("Sex-wise Distribution of Unemployed Youth (%)")

    # Filter only unemployed
    unemp_df = df[df["employment_status"] == "Unemployed"]

    # Count by SEX
    sex_unemp = unemp_df["SEX"].value_counts().reset_index()
    sex_unemp.columns = ["SEX", "Count"]

    # Calculate percentage
    sex_unemp["Percentage"] = (
        sex_unemp["Count"] / sex_unemp["Count"].sum() * 100
    )

    # Create Pie Chart
    fig = px.pie(
        sex_unemp,
        names="SEX",
        values="Percentage",
        color_discrete_sequence=["#0B3C5D", "#6BAED6"]  
    )

    fig.update_traces(
        textinfo="percent+label",
        textfont_size=14
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
                     **Interpretation:**
                         The pie chart shows that youth unemployment is slightly higher among females (53.4%)
                         compared to males (46.6%). This suggests that young women may experience more difficulty
                         finding jobs than young men, possibly due to social expectations, family responsibilities,
                         restricted mobility, or limited access to suitable employment opportunities.
                     """)



## Age vs Employment
    st.subheader("Age Distribution of Unemployed Youth (%)")

# Age slider
    age_slider = st.slider("Select Age Range", 15, 29, (15, 29))

# Filter unemployed within selected age range
    unemp_df = df[
         (df["employment_status"] == "Unemployed") &
         (df["AGE"] >= age_slider[0]) &
         (df["AGE"] <= age_slider[1])
         ]

# Count age frequencies
    age_unemp =  unemp_df["AGE"].value_counts().sort_index()

# Convert to percentage
    age_unemp_pct = age_unemp / age_unemp.sum() * 100

# Convert to dataframe for plotly
    age_df = age_unemp_pct.reset_index()
    age_df.columns = ["AGE", "Percentage"]

# Create bar chart
    fig2 = px.bar(
      age_df,
      x="AGE",
      y="Percentage",
      text=age_df["Percentage"].round(1),
      color_discrete_sequence=["#0B3C5D"],
      title="Age Distribution of Unemployed Youth"
      )

# Show percentage labels properly
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')

    fig2.update_layout(
         yaxis_title="Percentage of Unemployed Youth (%)",
         xaxis_title="Age",
         title_x=0.5
         )

    st.plotly_chart(fig2, use_container_width=True)


    st.markdown("""
                     **Interpretation:**
                         The graph shows that youth unemployment in Sri Lanka is highest among the youngest age group, especially those
                         aged 19–24. This is mainly because many young people enter the labour market during this period after completing
                         school or higher education and often struggle to find jobs that match their skills. As age increases toward 29,  
                         unemployment steadily declines, suggesting that with time, experience, and skill development, young people are
                         more likely to secure employment or adjust their expectations to available opportunities.
                     """)




# Sector-wise Unemployment
    st.subheader("Youth Unemployment by Sector (%)")

# Filter only unemployed
    unemp_df = df[df["employment_status"] == "Unemployed"]

# Count sector frequencies
    sector_unemp = unemp_df["SECTOR"].value_counts()

# Convert to percentage
    sector_unemp_pct = sector_unemp / sector_unemp.sum() * 100

# Convert to dataframe
    sector_df = sector_unemp_pct.reset_index()
    sector_df.columns = ["SECTOR", "Percentage"]

# Plot bar chart
    fig_sector = px.bar(
         sector_df,
         x="SECTOR",
         y="Percentage",
         text=sector_df["Percentage"].round(1),
         color_discrete_sequence=["#0B3C5D"],
         title="Youth Unemployment by Sector"
         )

    fig_sector.update_traces(
         texttemplate='%{text:.1f}%',
         textposition='outside'
         )

    fig_sector.update_layout(
         yaxis_title="Percentage of Unemployed Youth (%)",
         xaxis_title="Sector",
         title_x=0.5,
         template="plotly_white"
         )

    st.plotly_chart(fig_sector, use_container_width=True)


    st.markdown("""
                     **Interpretation:**
                    The sector-wise bar chart shows that the majority of unemployed youth are from the rural sector (79.9%), while a much                     smaller proportion is from urban areas (14.8%) and estate sectors (5.3%).  This shows a clear gap between rural and                       urban areas, where young people in rural communities struggle more to find jobs because there are fewer industries
                    and limited employment opportunities available to them.
                     """)




# Province-wise Unemployment
    st.subheader("Youth Unemployment by Province (%)")

# Count province frequencies
    province_unemp = unemp_df["PROVINCE"].value_counts()

# Convert to percentage
    province_unemp_pct = province_unemp / province_unemp.sum() * 100

# Convert to dataframe
    province_df = province_unemp_pct.reset_index()
    province_df.columns = ["PROVINCE", "Percentage"]
 
# Sort by percentage (optional but cleaner)
    province_df = province_df.sort_values(by="Percentage", ascending=False)

# Plot bar chart
    fig_province = px.bar(
         province_df,
         x="PROVINCE",
         y="Percentage",
         text=province_df["Percentage"].round(1),
         color_discrete_sequence=["#0B3C5D"],
         title="Youth Unemployment by Province"
         )

    fig_province.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
        )

    fig_province.update_layout(
       yaxis_title="Percentage of Unemployed Youth (%)",
       xaxis_title="Province",
       title_x=0.5,
       template="plotly_white"
       )

    st.plotly_chart(fig_province, use_container_width=True)


    st.markdown(""" 
                From the information presented in the bar chart provided above, it can be observed that there is a significant disparity in the unemployment rate of the youth in the different provinces of Sri Lanka, where there is a large percentage of unemployment in the southern and western parts of the country. The Southern province is facing the most critical unemployment situation with a 26.5% unemployment rate, while the Western province is second in the list with a 22.2% unemployment rate. On the other hand, the unemployment rate in the Eastern and Northern provinces is relatively very low, as low as 3.7%.
                """)

    st.markdown("## 🔎 Filter Options")

    sector_option = st.radio(
        "Select Sector for Gender and Age",
        list(df["SECTOR"].unique()),
        horizontal=True
    )

    if sector_option == "All":
        df_filtered = df.copy()
    else:
        df_filtered = df[df["SECTOR"] == sector_option]


    st.divider()


col1, col2 = st.columns(2)
# COLUMN 1 — Gender Pie Chart
if menu == "Exploratory Analysis":

   with col1:

    st.subheader("Sex-wise Distribution of Unemployed Youth (%)")

    unemp_df = df_filtered[df_filtered["employment_status"] == "Unemployed"]

    sex_unemp = unemp_df["SEX"].value_counts().reset_index()
    sex_unemp.columns = ["SEX", "Count"]

    sex_unemp["Percentage"] = (
        sex_unemp["Count"] / sex_unemp["Count"].sum() * 100
    )

    fig_gender = px.pie(
        sex_unemp,
        names="SEX",
        values="Percentage",
        color_discrete_sequence=["#0B3C5D", "#6BAED6"]
    )

    fig_gender.update_traces(
        textinfo="percent+label",
        textfont_size=14
    )

    st.plotly_chart(fig_gender, use_container_width=True)

    # Dynamic Interpretation
    male_pct = sex_unemp.loc[sex_unemp["SEX"] == "Male", "Percentage"]
    female_pct = sex_unemp.loc[sex_unemp["SEX"] == "Female", "Percentage"]

    male_pct = float(male_pct.values[0]) if not male_pct.empty else 0
    female_pct = float(female_pct.values[0]) if not female_pct.empty else 0

    if male_pct > female_pct:
        higher_group = "males"
        diff = round(male_pct - female_pct, 1)
    else:
        higher_group = "females"
        diff = round(female_pct - male_pct, 1)

    st.markdown(f"""
    **Interpretation:**
    Within the selected sector (**{sector_option}**), unemployment is higher among 
    **{higher_group}**, with a difference of approximately **{diff}%**.
    """)

# COLUMN 2 — Age Bar Chart

   with col2:

    st.subheader("Age Distribution of Unemployed Youth (%)")


    age_filtered = unemp_df[
        (unemp_df["AGE"] >= age_slider[0]) &
        (unemp_df["AGE"] <= age_slider[1])
    ]

    age_unemp = age_filtered["AGE"].value_counts().sort_index()
    age_unemp_pct = age_unemp / age_unemp.sum() * 100

    age_df = age_unemp_pct.reset_index()
    age_df.columns = ["AGE", "Percentage"]

    fig_age = px.bar(
        age_df,
        x="AGE",
        y="Percentage",
        text=age_df["Percentage"].round(1),
        color_discrete_sequence=["#0B3C5D"],
        title="Age Distribution of Unemployed Youth"
    )

    fig_age.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )

    fig_age.update_layout(
        yaxis_title="Percentage (%)",
        xaxis_title="Age",
        title_x=0.5
    )

    st.plotly_chart(fig_age, use_container_width=True)

    # Dynamic Interpretation
    highest_age = age_df.loc[age_df["Percentage"].idxmax(), "AGE"]
    highest_pct = age_df["Percentage"].max()

    st.markdown(f"""
    **Interpretation:**
    The highest unemployment percentage occurs at age **{int(highest_age)}**, 
    accounting for approximately **{highest_pct:.1f}%** within the selected range.
    """)


# HYPOTHESIS TESTING – INDIVIDUAL CONTRIBUTION

if menu == "Hypothesis Testing":

    st.markdown("## Hypothesis Testing – Statistical Inference")

    st.markdown("""
    This section examines whether youth unemployment is statistically 
    associated with demographic characteristics using formal hypothesis testing.

    We test whether observed differences are statistically significant 
    or occurred by random chance.
    """)

    st.divider()

# CHI-SQUARE TEST SECTION
    st.subheader("1️⃣ Chi-Square Test of Independence")

    st.markdown("""
    This test checks whether employment status is associated 
    with a selected categorical variable.
    """)

    variable = st.selectbox(
        "Select Variable to Test",
        ["SEX", "SECTOR", "PROVINCE"]
    )

    st.markdown("### Hypotheses")

    st.info(f"""
    **H₀ (Null Hypothesis):** Employment status is NOT associated with {variable}.  
    **H₁ (Alternative Hypothesis):** Employment status IS associated with {variable}.
    """)

    contingency = pd.crosstab(df[variable], df["employment_status"])

    st.markdown("### Contingency Table")
    st.dataframe(contingency)

    chi2, p_value, dof, expected = chi2_contingency(contingency)

    st.markdown("### Test Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Chi-Square Statistic", round(chi2, 3))
        st.metric("Degrees of Freedom", dof)

    with col2:
        st.metric("p-value", round(p_value, 5))

    st.markdown("### Decision Rule")
    st.write("Significance Level (α) = 0.05")

    if p_value < 0.05:
        st.success(f"""
        Since p-value < 0.05, we reject H₀.  
        There is a statistically significant association between employment status and {variable}.
        """)
    else:
        st.warning(f"""
        Since p-value ≥ 0.05, we fail to reject H₀.  
        There is no statistically significant association between employment status and {variable}.
        """)

    st.divider()

# INDEPENDENT T-TEST SECTION
    st.subheader("2️⃣ Independent Sample t-Test (Age Difference)")

    st.markdown("""
    This test checks whether the average age differs significantly 
    between employed and unemployed youth.
    """)

    st.markdown("### Hypotheses")

    st.info("""
    **H₀ (Null Hypothesis):** The mean age of employed and unemployed youth is equal.  
    **H₁ (Alternative Hypothesis):** The mean age of employed and unemployed youth is different.
    """)

    employed = df[df["employment_status"] == "Employed"]["AGE"]
    unemployed = df[df["employment_status"] == "Unemployed"]["AGE"]

    t_stat, p_val = ttest_ind(employed, unemployed, equal_var=False)

    st.markdown("### Sample Means")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Mean Age (Employed)", round(employed.mean(), 2))

    with col4:
        st.metric("Mean Age (Unemployed)", round(unemployed.mean(), 2))

    st.markdown("### Test Results")

    col5, col6 = st.columns(2)

    with col5:
        st.metric("t-Statistic", round(t_stat, 3))

    with col6:
        st.metric("p-value", round(p_val, 5))

    st.markdown("### Decision Rule")
    st.write("Significance Level (α) = 0.05")

    if p_val < 0.05:
        st.success("""
        Since p-value < 0.05, we reject H₀.  
        There is a statistically significant difference in mean age 
        between employed and unemployed youth.
        """)
    else:
        st.warning("""
        Since p-value ≥ 0.05, we fail to reject H₀.  
        There is no statistically significant difference in mean age 
        between employed and unemployed youth.
        """)

    st.divider()

    st.markdown("""
    ### Overall Interpretation

    These statistical tests help confirm whether the patterns observed 
    in the exploratory analysis are statistically significant. 

    Significant results suggest that demographic characteristics such as 
    sex, sector, province, and age play an important role in youth unemployment.
    """)



####Model1: Logistic Regression
if menu == "Logistic Regression":

    df_model = df.copy()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_model["employment_status"] = le.fit_transform(df_model["employment_status"])

    X = pd.get_dummies(
         df_model[["AGE", "SEX", "SECTOR", "EDU_group", "PROVINCE"]],
         drop_first=True
         )

    y = df_model["employment_status"]

    X_train, X_test, y_train, y_test = train_test_split(
      X, y,
      test_size=0.3,
      random_state=42,
      stratify=y   # better for classification balance
    )

    st.header("Logistic Regression Model")

    st.info("""
    Logistic Regression is used to estimate the probability that a youth is unemployed.

    Model Prediction:
    
    • 0 → Employed  
    • 1 → Unemployed  

    The model helps identify how demographic and regional factors influence youth unemployment.
    """)


# Train Model

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)


# Class Distribution

    st.subheader("Class Distribution in Dataset")

    class_dist = df_model["employment_status"].value_counts().reset_index()
    class_dist.columns = ["Employment Status", "Count"]

    class_dist["Employment Status"] = class_dist["Employment Status"].map({
        0: "Employed",
        1: "Unemployed"
    })

    total = class_dist["Count"].sum()
    unemployed = class_dist.loc[
        class_dist["Employment Status"] == "Unemployed", "Count"
    ].values[0]

    unemployed_pct = unemployed / total * 100

    col1, col2, col3 = st.columns([2,3,2])
    with col2:
        st.table(class_dist)

    st.success(f"""
    Out of {total} youth, {unemployed} ({unemployed_pct:.1f}%) are unemployed.
    """)

    st.markdown("---")


# Key Model Performance Metrics

    st.subheader("Model Performance Summary")

    accuracy = report["accuracy"]
    recall_unemp = report["1"]["recall"]
    precision_unemp = report["1"]["precision"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Overall Accuracy", f"{accuracy:.2%}")
    col2.metric("Recall (Unemployed)", f"{recall_unemp:.2%}")
    col3.metric("Precision (Unemployed)", f"{precision_unemp:.2%}")

    st.markdown("""
    **Interpretation:**

    • Accuracy shows overall correctness of predictions.  
    • Recall (Unemployed) shows how well the model identifies unemployed youth.  
    • Precision shows how reliable unemployment predictions are.  
    """)

    st.markdown("---")


# Interactive Prediction Section

    st.subheader("🔎 Try Your Own Prediction")

    age_input = st.slider("Select Age", 15, 29, 22)
    sex_input = st.selectbox("Select Sex", df["SEX"].unique())
    sector_input = st.selectbox("Select Sector", df["SECTOR"].unique())
    edu_input = st.selectbox("Select Education Group", df["EDU_group"].unique())
    province_input = st.selectbox("Select Province", df["PROVINCE"].unique())

    # Create user input dataframe
    user_df = pd.DataFrame({
        "AGE": [age_input],
        "SEX": [sex_input],
        "SECTOR": [sector_input],
        "EDU_group": [edu_input],
        "PROVINCE": [province_input]
    })

    # Convert to dummy variables same as training
    user_df = pd.get_dummies(user_df)

    # Align columns with training data
    user_df = user_df.reindex(columns=X.columns, fill_value=0)

    # Predict probability
    probability = model.predict_proba(user_df)[0][1]

    st.success(f"""
    Predicted Probability of Being Unemployed: {probability:.2%}
    """)

    if probability > 0.5:
        st.warning("The model predicts that this youth is more likely to be Unemployed.")
    else:
        st.info("The model predicts that this youth is more likely to be Employed.")

    st.markdown("""
### Model Performance Interpretation

The model achieved an accuracy of approximately 63.4%. 
This moderate predictive performance suggests that youth unemployment 
is influenced by complex structural and socio-economic factors beyond 
the variables included in this model (age, sex, education, sector, and province).

Therefore, while the model provides useful insights into key predictors, 
it does not fully capture all determinants of youth unemployment.
""")





# ROC Curve & AUC (Individual Contribution)

    st.markdown("---")
    st.subheader("📈 Model Evaluation – ROC Curve & AUC ")

    st.markdown("""
This section adds a small new analysis to evaluate the logistic regression model.
We use the ROC curve and compute the AUC to measure how well the model distinguishes
between employed and unemployed youth.
""")

# Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC values
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    st.metric("AUC Score", f"{auc_score:.3f}")

# Plot ROC Curve
    fig_roc, ax = plt.subplots(figsize=(4, 2))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    st.pyplot(fig_roc)

    st.markdown("### Interpretation")
    st.write("""
The ROC curve shows how well the logistic regression model can distinguish 
between unemployed and employed youth.

- The **AUC (Area Under the Curve)** score ranges from 0 to 1:
  - 0.5 → The model is guessing randomly.
  - Closer to 1 → The model has better discrimination ability.

- Here, the AUC value indicates the model performs better than random guessing,
  meaning it has a moderate ability to separate unemployed from employed youth.

On the ROC graph:
- **True Positive Rate (TPR)** represents the percentage of unemployed youth 
  correctly predicted as unemployed.
- **False Positive Rate (FPR)** represents the percentage of employed youth 
  incorrectly predicted as unemployed.

In simple terms:
- A higher TPR is desirable because it means the model successfully identifies 
  more unemployed youth.
- A lower FPR is desirable because it reduces incorrect classification 
  of employed youth as unemployed.

Although the model is not perfect, the ROC analysis shows that it captures 
meaningful patterns in the data and provides additional evaluation beyond accuracy.
""")


# CONCLUSION PAGE
if menu == "Conclusion":

    st.markdown("## Conclusion & Key Insights")

    st.markdown("""
    This web application explored youth unemployment in Sri Lanka 
    using exploratory data analysis, hypothesis testing, and logistic regression modelling.
    """)

    st.divider()

# Key Findings Section
    st.subheader("🔎 Key Statistical Findings")

    st.markdown("""
    • Chi-square tests showed that employment status is significantly associated 
      with key demographic characteristics such as sex, sector, and province.

    • The independent sample t-test indicated whether there is a significant 
      difference in the average age between employed and unemployed youth.

    • These findings confirm that youth unemployment is not randomly distributed 
      across the population.
    """)

    st.divider()

# Logistic Regression Summary

    st.subheader("📊 Logistic Regression Model Insights")

    st.markdown("""
    The logistic regression model was used to estimate the probability of being unemployed 
    based on age, sex, education level, sector, and province.

    The model achieved moderate predictive accuracy. While it provides useful insights 
    into important predictors, it does not fully capture all structural and economic 
    factors influencing youth unemployment.
    """)

    st.info("""
    This suggests that youth unemployment is influenced by broader socio-economic 
    conditions beyond the variables included in this dataset.
    """)

    st.divider()

# Policy Implications
    st.subheader("🏛 Policy Implications")

    st.markdown("""
    • Targeted employment programs may be necessary for specific provinces or sectors.

    • Education-to-employment transition policies should be strengthened.

    • Gender-sensitive labour policies may help reduce disparities in youth employment.

    These results provide evidence-based support for policy planning and labour market interventions.
    """)

    st.divider()

# Limitations
    st.subheader("⚠ Limitations")

    st.markdown("""
    • The dataset includes only selected demographic variables.

    • Other important factors such as income level, work experience, 
      household background, and macroeconomic conditions were not included.

    • The model accuracy indicates that youth unemployment is a complex issue 
      requiring broader structural analysis.
    """)

    st.divider()

# Final Statement
    st.success("""
    Overall, this application demonstrates how statistical methods and 
    predictive modelling can be used to understand youth unemployment patterns 
    and support data-driven decision-making in Sri Lanka.
    """)