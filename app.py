import streamlit as st
import pickle
import re
import os
from streamlit_option_menu import option_menu
import pandas as pd

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="IT Helpdesk Classifier",
    page_icon="💻",
    layout="centered"
)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

try:
    model, tfidf = load_model()
except FileNotFoundError:
    st.error("❌ Model files not found. Make sure `model/model.pkl` and `model/tfidf.pkl` exist.")
    st.stop()

# ----------------------------
# CLEAN TEXT
# Matches notebook Cell 12 exactly:
# lowercase → remove non-letters → strip extra spaces
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------
# PREDICT
# Notebook used: tfidf.transform() then model.predict() + predict_proba()
# TfidfVectorizer was created with max_features=10000, stop_words='english'
# LogisticRegression with max_iter=1000, random_state=42
# ----------------------------
def predict(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = round(max(model.predict_proba(vectorized)[0]) * 100, 2)
    return prediction, confidence

# ----------------------------
# CATEGORY COLORS
# 8 categories from notebook Cell 10 output
# ----------------------------
category_colors = {
    "Hardware":               "#1f77b4",
    "HR Support":             "#ff7f0e",
    "Access":                 "#2ca02c",
    "Miscellaneous":          "#9467bd",
    "Storage":                "#8c564b",
    "Purchase":               "#e377c2",
    "Internal Project":       "#17becf",
    "Administrative rights":  "#d62728"
}

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("IT Helpdesk Classifier")
st.sidebar.markdown("""
**AI-powered IT Support Ticket Classifier**

This app uses a machine learning model trained on
47,837 real IT support tickets to automatically
classify them into the correct department category.

**Tech Stack:** Python, Scikit-learn, TF-IDF,
Logistic Regression, Streamlit
""")

st.sidebar.markdown("### Capabilities")
st.sidebar.markdown("""
- Single ticket classification
- Bulk CSV ticket analysis
- Confidence score per prediction
- Downloadable results
""")

st.sidebar.markdown("### Categories")
for cat, color in category_colors.items():
    st.sidebar.markdown(
        f'<span style="color:{color}; font-weight:600;">● </span>{cat}',
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")
st.sidebar.markdown("""
[GitHub](https://github.com/Fiazbhk) |
[LinkedIn](https://www.linkedin.com/in/fiazbhk/) |
[LeetCode](https://leetcode.com/u/muhammadfiazbhk/)
""")

# ----------------------------
# TITLE
# ----------------------------
st.title("IT Helpdesk Classifier")

# ----------------------------
# NAVIGATION
# ----------------------------
selected_tab = option_menu(
    menu_title=None,
    options=["Single Ticket", "CSV Upload", "About the Model"],
    icons=["ticket-detailed", "file-earmark-spreadsheet", "info-circle"],
    orientation="horizontal",
    default_index=0
)

# ----------------------------
# TAB 1 — SINGLE TICKET
# ----------------------------
if selected_tab == "Single Ticket":
    st.subheader("Classify a Single Ticket")
    st.write("Enter an IT support ticket below and the model will classify it into the correct category.")

    ticket = st.text_area(
        "Enter ticket description here:",
        height=150,
        placeholder="Example: My laptop screen is broken and I cannot turn it on"
    )

    if st.button("Classify"):
        if ticket.strip() == "":
            st.warning("Please enter a ticket description before classifying.")
        else:
            category, confidence = predict(ticket)
            color = category_colors.get(category, "#333333")

            st.divider()

            st.markdown(
                f'<div style="background-color:{color}20; border-left:5px solid {color}; '
                f'padding:16px; border-radius:8px;">'
                f'<p style="margin:0; font-size:14px; color:{color}; font-weight:600;">PREDICTED CATEGORY</p>'
                f'<p style="margin:4px 0 0 0; font-size:28px; font-weight:700; color:{color};">{category}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.metric(label="Confidence Score", value=f"{confidence}%")

            st.divider()
            st.markdown("**Cleaned text (what the model sees):**")
            st.code(clean_text(ticket))

# ----------------------------
# TAB 2 — CSV UPLOAD
# ----------------------------
if selected_tab == "CSV Upload":
    st.subheader("Classify Multiple Tickets via CSV")
    st.write("Upload a CSV file with a column named **Document** containing the ticket descriptions.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "Document" not in df.columns:
            st.error("The uploaded CSV does not have a column named 'Document'. Please check your file.")
        else:
            st.success(f"{len(df)} tickets loaded successfully.")
            st.dataframe(df.head())

            if st.button("Run Classification"):
                with st.spinner("Classifying tickets..."):
                    df["Predicted Category"] = df["Document"].apply(lambda x: predict(str(x))[0])
                    df["Confidence (%)"]      = df["Document"].apply(lambda x: predict(str(x))[1])

                st.success("Classification complete.")
                st.divider()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Tickets",    len(df))
                col2.metric("Categories Found", df["Predicted Category"].nunique())
                col3.metric("Avg Confidence",   f"{round(df['Confidence (%)'].mean(), 2)}%")

                st.divider()
                st.markdown("### Category Distribution")
                category_counts = df["Predicted Category"].value_counts().reset_index()
                category_counts.columns = ["Category", "Count"]
                st.dataframe(category_counts, use_container_width=True)

                st.divider()
                st.markdown("### Full Results")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="classified_tickets.csv",
                    mime="text/csv"
                )

# ----------------------------
# TAB 3 — ABOUT THE MODEL
# ----------------------------
if selected_tab == "About the Model":
    st.subheader("Model Information")

    st.markdown("""
**Algorithm:** Logistic Regression (`max_iter=1000`, `random_state=42`)

**Text Features:** TF-IDF (`max_features=10000`, `stop_words='english'`)

**Dataset:** IT Service Ticket Classification Dataset — Kaggle

**File:** `all_tickets_processed_improved_v3.csv`

**Total Records:** 47,837 tickets

**Training Samples:** 38,269 (80%)

**Testing Samples:** 9,568 (20%)

**Train/Test Split:** `random_state=42`
""")

    st.divider()

    st.markdown("### Model Performance")

    performance_data = {
        "Category": [
            "Access", "Administrative rights", "HR Support", "Hardware",
            "Internal Project", "Miscellaneous", "Purchase", "Storage",
            "Overall Accuracy"
        ],
        "Precision": [0.92, 0.87, 0.85, 0.79, 0.91, 0.80, 0.97, 0.93, ""],
        "Recall":    [0.87, 0.68, 0.83, 0.88, 0.80, 0.82, 0.88, 0.84, ""],
        "F1 Score":  [0.89, 0.76, 0.84, 0.83, 0.86, 0.81, 0.92, 0.88, "84.53%"]
    }

    st.dataframe(pd.DataFrame(performance_data), use_container_width=True)

    st.divider()

    st.markdown("""
### Ticket Count per Category
| Category              | Tickets |
|-----------------------|---------|
| Hardware              | 13,617  |
| HR Support            | 10,915  |
| Access                |  7,125  |
| Miscellaneous         |  7,060  |
| Storage               |  2,777  |
| Purchase              |  2,464  |
| Internal Project      |  2,119  |
| Administrative rights |  1,760  |
""")

    st.divider()

    st.markdown("""
### Notes
- Model was trained on English IT support tickets only
- TF-IDF removes common English stop words (e.g. *the*, *is*, *and*)
- Best suited for formal helpdesk ticket language
- Confidence score reflects model probability, not absolute certainty
- Administrative rights has the lowest recall because it has the fewest training samples
""")