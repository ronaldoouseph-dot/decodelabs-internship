import streamlit as st
from app.ui import (
    inject_custom_css,
    init_session_state,
    render_dashboard_page,
    render_dataset_page,
    render_preprocessing_page,
    render_split_page,
    render_knn_page,
    render_evaluation_page,
    render_prediction_page,
    render_how_it_works
)

# Set Page Config
st.set_page_config(
    page_title="AI Iris Classification Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom styles
inject_custom_css()

# Initialize session state (load dataset, fit scaler, train default KNN)
init_session_state()

# Sidebar title and logo
st.sidebar.markdown("## 🌸 Iris AI Classifier")
st.sidebar.write("Educational Supervised Learning Pipeline")

# Navigation Menu
page = st.sidebar.radio(
    "Go to Step:",
    [
        "1. Dashboard",
        "2. Dataset Details",
        "3. Data Preprocessing",
        "4. Train / Test Split",
        "5. KNN Classifier",
        "6. Model Evaluation",
        "7. Interactive Prediction",
        "8. ML Academy (How It Works)"
    ]
)

# Routing
if page == "1. Dashboard":
    render_dashboard_page()
elif page == "2. Dataset Details":
    render_dataset_page()
elif page == "3. Data Preprocessing":
    render_preprocessing_page()
elif page == "4. Train / Test Split":
    render_split_page()
elif page == "5. KNN Classifier":
    render_knn_page()
elif page == "6. Model Evaluation":
    render_evaluation_page()
elif page == "7. Interactive Prediction":
    render_prediction_page()
elif page == "8. ML Academy (How It Works)":
    render_how_it_works()
