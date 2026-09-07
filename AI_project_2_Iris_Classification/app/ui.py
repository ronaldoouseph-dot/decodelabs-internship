import streamlit as st
import pandas as pd
import numpy as np
import os
from app.data import load_raw_data, get_train_test_split_data
from app.preprocessing import IrisScaler
from app.models import train_knn, analyze_k_values
from app.evaluation import compute_all_metrics, generate_confusion_matrix_df, generate_report_df
from app.visualization import (
    plot_interactive_scatter,
    plot_scaling_comparison,
    plot_train_test_split_dist,
    plot_k_error_rates,
    plot_confusion_matrix_heatmap,
    plot_knn_space,
    get_nearest_neighbors_info,
    CLASS_COLORS
)

# Helper to load and inject CSS
def inject_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Initialize Session State variables to store the pipeline components
def init_session_state():
    if 'data_loaded' not in st.session_state:
        # Load raw data
        iris_bunch, df_raw = load_raw_data()
        st.session_state.iris_bunch = iris_bunch
        st.session_state.df_raw = df_raw
        
        # Split data 80% train, 20% test
        X_train, X_test, y_train, y_test, feature_names, class_names = get_train_test_split_data()
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.feature_names = feature_names
        st.session_state.class_names = class_names
        
        # Fit scaler on X_train ONLY to prevent data leakage
        scaler = IrisScaler()
        X_train_scaled = scaler.fit_transform_train(X_train)
        X_test_scaled = scaler.transform_test(X_test)
        st.session_state.scaler = scaler
        st.session_state.X_train_scaled = X_train_scaled
        st.session_state.X_test_scaled = X_test_scaled
        
        # Set default K
        st.session_state.k = 5
        
        # Train default KNN model
        st.session_state.model = train_knn(X_train_scaled, y_train, k=st.session_state.k)
        
        # Precompute tuning analysis
        k_tuning, accuracies_tuning, error_rates_tuning = analyze_k_values(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        st.session_state.k_tuning = k_tuning
        st.session_state.accuracies_tuning = accuracies_tuning
        st.session_state.error_rates_tuning = error_rates_tuning
        
        st.session_state.data_loaded = True

# Helper to update model when K is changed
def update_knn_model(new_k):
    st.session_state.k = new_k
    st.session_state.model = train_knn(
        st.session_state.X_train_scaled,
        st.session_state.y_train,
        k=new_k
    )

def render_pipeline():
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step">
            <div class="icon">📥</div>
            <div class="title">1. Input Data</div>
            <div class="subtitle">Iris Dataset (150 Samples)</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="icon">⚖️</div>
            <div class="title">2. Scaling</div>
            <div class="subtitle">StandardScaler (Train only fit)</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="icon">✂️</div>
            <div class="title">3. Train/Test Split</div>
            <div class="subtitle">80% Train / 20% Test</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="icon">🧠</div>
            <div class="title">4. KNN Model</div>
            <div class="subtitle">K-Nearest Neighbors</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="icon">🎯</div>
            <div class="title">5. Predictions</div>
            <div class="subtitle">Unseen Test / Custom Inputs</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="icon">📊</div>
            <div class="title">6. Evaluation</div>
            <div class="subtitle">Confusion Matrix & Metrics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_dashboard_page():
    st.title("🛡️ Project 2: Data Classification Using AI")
    st.subheader("Interactive Supervised Machine Learning Dashboard - Iris Dataset")
    
    render_pipeline()
    
    # Calculate current predictions and metrics
    predictions = st.session_state.model.predict(st.session_state.X_test_scaled)
    metrics = compute_all_metrics(st.session_state.y_test, predictions)
    
    st.markdown("### 📊 Active Model Performance Metrics (Test Set)")
    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card accuracy">
            <div class="metric-title">Accuracy Score</div>
            <div class="metric-value">{metrics['accuracy']:.2%}</div>
        </div>
        <div class="metric-card precision">
            <div class="metric-title">Weighted Precision</div>
            <div class="metric-value">{metrics['precision']:.2%}</div>
        </div>
        <div class="metric-card recall">
            <div class="metric-title">Weighted Recall</div>
            <div class="metric-value">{metrics['recall']:.2%}</div>
        </div>
        <div class="metric-card f1">
            <div class="metric-title">Weighted F1-Score</div>
            <div class="metric-value">{metrics['f1_score']:.2%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="custom-card">
            <h3>📈 Key Pipeline Statistics</h3>
            <ul>
                <li><b>Total Samples:</b> 150 flowers</li>
                <li><b>Classes:</b> 3 species (Setosa, Versicolor, Virginica)</li>
                <li><b>Features:</b> 4 input dimensions (Sepal/Petal lengths & widths)</li>
                <li><b>Split Strategy:</b> 80% Training (120), 20% Testing (30)</li>
                <li><b>Model Setting:</b> K = {st.session_state.k} (Majority Voting)</li>
                <li><b>Random State:</b> 42 (Ensures exact reproducibility)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3>🎓 Educational Goal</h3>
            <p>This application demonstrates a complete <b>supervised machine learning workflow</b>.
            The system fits a <i>K-Nearest Neighbors</i> (KNN) model on standardized training data 
            and predicts the labels of unseen flower samples based on their physical features.</p>
            <p>Use the navigation menu on the left to explore each step of the pipeline in depth!</p>
        </div>
        """, unsafe_allow_html=True)

def render_dataset_page():
    st.title("📂 Dataset Exploration")
    st.write("Understand the distribution and structure of the classic Iris dataset programmatically loaded from scikit-learn.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="custom-card">
            <h3>📊 Dataset Summary</h3>
            <ul>
                <li><b>Total Samples:</b> 150</li>
                <li><b>Setosa (Class 0):</b> 50 samples</li>
                <li><b>Versicolor (Class 1):</b> 50 samples</li>
                <li><b>Virginica (Class 2):</b> 50 samples</li>
                <li><b>Features:</b> 4 physical dimensions</li>
            </ul>
            <p>The classes are perfectly balanced (50 samples per class), making accuracy a relatively reliable metric here, though we still analyze Precision, Recall, and F1 scores.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive plot controls
        x_feature = st.selectbox("X-Axis Feature", st.session_state.feature_names, index=2) # Petal Length
        y_feature = st.selectbox("Y-Axis Feature", st.session_state.feature_names, index=3) # Petal Width
        
    with col2:
        fig = plot_interactive_scatter(st.session_state.df_raw, x_feature, y_feature)
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("### 📋 Browse Raw Data Samples")
    st.write("Search and filter the raw table data below.")
    st.dataframe(st.session_state.df_raw, use_container_width=True)

def render_preprocessing_page():
    st.title("⚖️ Data Preprocessing & Scaling")
    
    st.markdown("""
    <div class="custom-card">
        <h3>🔍 Why Feature Scaling is Mandatory for KNN</h3>
        <p>K-Nearest Neighbors is a <b>distance-based</b> algorithm. It computes Euclidean distances between data points to identify neighbors.</p>
        <p>If one feature has values ranging from 0 to 100 (e.g., height) and another ranges from 0 to 1 (e.g., width), the distance calculations will be completely dominated by the feature with the larger scale. Scaling resolves this by shifting all features to have a <b>Mean of 0</b> and a <b>Variance of 1</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("> [!IMPORTANT]\n> **Strict Data Leakage Protection:** We fit the `StandardScaler` on the training split only (`X_train`), and use those parameters to transform both training (`X_train`) and testing (`X_test`) sets. Fitting on the test set would leak information about the test set distribution into the training pipeline!")

    feature_to_plot = st.selectbox("Choose feature to view before/after standardization", st.session_state.feature_names, index=2)
    
    # Generate scaled DataFrame for comparison display
    X_train_scaled_df = pd.DataFrame(st.session_state.X_train_scaled, columns=st.session_state.feature_names)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Raw Values (First 5 Rows)")
        st.dataframe(st.session_state.df_raw[st.session_state.feature_names].head(), use_container_width=True)
    with col2:
        st.markdown("#### Standard Scaled Values (First 5 Rows)")
        st.dataframe(X_train_scaled_df.head(), use_container_width=True)
        
    fig = plot_scaling_comparison(st.session_state.df_raw, X_train_scaled_df, feature_to_plot)
    st.plotly_chart(fig, use_container_width=True)

def render_split_page():
    st.title("✂️ Train / Test Split")
    
    st.markdown("""
    <div class="custom-card">
        <h3>📂 80/20 Stratified Train-Test Split</h3>
        <p>To validate a machine learning model, we must test it on data it has never seen during training. This simulates real-world performance.</p>
        <ul>
            <li><b>Total Samples:</b> 150 (100%)</li>
            <li><b>Training Split (80%):</b> 120 samples (Used to fit the model)</li>
            <li><b>Testing Split (20%):</b> 30 samples (Kept hidden, used for evaluation)</li>
        </ul>
        <p><b>Stratification</b> guarantees that each split contains the exact same class proportions (1:1:1 split of Setosa, Versicolor, and Virginica) as the original dataset. This prevents class imbalance issues during evaluation.</p>
        <p><b>Shuffling</b> is done prior to splitting to eliminate any ordering biases that might occur if the dataset was sorted (e.g. Setosa samples grouped at the top).</p>
    </div>
    """, unsafe_allow_html=True)
    
    fig = plot_train_test_split_dist(st.session_state.y_train, st.session_state.y_test, st.session_state.class_names)
    st.plotly_chart(fig, use_container_width=True)

def render_knn_page():
    st.title("🧠 K-Nearest Neighbors Classifier & Tuning")
    
    st.markdown("""
    <div class="custom-card">
        <h3>💡 How K-Nearest Neighbors (KNN) Works</h3>
        <p>KNN is a simple, intuitive classification algorithm:</p>
        <ol>
            <li>Given an unseen data point, compute its distance to all training points.</li>
            <li>Select the <b>K nearest training points</b> (the neighbors).</li>
            <li>Count the class labels of these K neighbors.</li>
            <li>Use <b>majority voting</b> to assign the label with the most votes to the new point.</li>
        </ol>
        <p><b>K = 5</b> means the model looks at the 5 nearest training examples and uses majority voting to classify.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive K selection
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 🎛️ Tune Hyperparameter K")
        new_k = st.slider("Select K (number of neighbors)", min_value=1, max_value=20, value=st.session_state.k, step=1)
        if new_k != st.session_state.k:
            update_knn_model(new_k)
            st.rerun()
            
        st.markdown(f"""
        <div style="background-color:rgba(108, 92, 231, 0.1); border-left:4px solid #6C5CE7; padding:12px; border-radius:4px;">
            <b>Active Settings:</b><br>
            K = {st.session_state.k}<br>
            Distance Metric: Euclidean<br>
            Weighting: Uniform (all neighbors vote equally)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        #### Understanding K Size:
        - **Small K (e.g., K=1):** Highly sensitive to noise and outliers (High Variance/Overfitting).
        - **Large K (e.g., K=20):** Smooths the decision boundary but may miss local patterns (High Bias/Underfitting).
        - **Optimal K:** The "elbow" point where error rate stabilizes on unseen test data.
        """)

    with col2:
        # Plot K-value analysis plot
        fig_tuning = plot_k_error_rates(st.session_state.k_tuning, st.session_state.error_rates_tuning, st.session_state.k)
        st.plotly_chart(fig_tuning, use_container_width=True)
        
    st.markdown("### 🔍 Interactive Neighborhood Visualizer")
    st.write("Input a query sample coordinates, choose 2 dimensions to plot, and watch the model perform neighbor lookups in real time.")
    
    vcol1, vcol2 = st.columns([1, 2])
    with vcol1:
        sepal_len = st.slider("Query Sepal Length (cm)", 4.0, 8.0, 5.8, step=0.1)
        sepal_wid = st.slider("Query Sepal Width (cm)", 2.0, 4.5, 3.0, step=0.1)
        petal_len = st.slider("Query Petal Length (cm)", 1.0, 7.0, 4.35, step=0.1)
        petal_wid = st.slider("Query Petal Width (cm)", 0.1, 2.5, 1.3, step=0.1)
        
        # Projection features
        proj_x = st.selectbox("Plot X Dimension", st.session_state.feature_names, index=2, key="proj_x")
        proj_y = st.selectbox("Plot Y Dimension", st.session_state.feature_names, index=3, key="proj_y")
        
        feat_x_idx = st.session_state.feature_names.index(proj_x)
        feat_y_idx = st.session_state.feature_names.index(proj_y)
        
    with vcol2:
        query_vector = np.array([sepal_len, sepal_wid, petal_len, petal_wid])
        
        fig_space, neighbor_details = plot_knn_space(
            st.session_state.X_train,
            st.session_state.y_train,
            query_vector,
            st.session_state.k,
            (feat_x_idx, feat_y_idx),
            st.session_state.feature_names,
            st.session_state.class_names,
            st.session_state.scaler
        )
        st.plotly_chart(fig_space, use_container_width=True)
        
        # Display neighbor voting breakdown
        st.markdown("#### 🗳️ K-Nearest Neighbors Voting Breakdown")
        vote_cols = st.columns(3)
        for i, cname in enumerate(st.session_state.class_names):
            cap_name = cname.capitalize()
            v_count = neighbor_details['vote_counts'].get(cap_name, 0)
            percentage = (v_count / st.session_state.k) * 100 if st.session_state.k > 0 else 0
            with vote_cols[i]:
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.6); border-radius: 8px; border-left: 4px solid {CLASS_COLORS[i]}; padding: 10px; text-align: center;">
                    <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase;">{cap_name}</div>
                    <div style="font-size: 22px; font-weight: 700; color: #FFF;">{v_count} / {st.session_state.k}</div>
                    <div style="font-size: 12px; color: #38BDF8;">{percentage:.0f}% of votes</div>
                </div>
                """, unsafe_allow_html=True)

def render_evaluation_page():
    st.title("🎯 Model Evaluation & Validation")
    st.write("Evaluate model predictions using robust classification metrics on the unseen 20% test set.")
    
    # Active K banner with quick slider
    eval_col1, eval_col2 = st.columns([2, 1])
    with eval_col1:
        st.markdown(f"""
        <div style="background: rgba(108, 92, 231, 0.15); border-left: 4px solid #6C5CE7; padding: 14px; border-radius: 8px; margin-bottom: 16px;">
            <b>Active Model Configuration:</b> Evaluated with <b>K = {st.session_state.k}</b> Neighbors | <b>Test Split:</b> 30 Samples (Stratified)
        </div>
        """, unsafe_allow_html=True)
    with eval_col2:
        eval_k = st.slider("Quickly Tune K for Evaluation:", min_value=1, max_value=20, value=st.session_state.k, step=1, key="eval_k_slider")
        if eval_k != st.session_state.k:
            update_knn_model(eval_k)
            st.rerun()
            
    # Calculate predictions
    predictions = st.session_state.model.predict(st.session_state.X_test_scaled)
    metrics = compute_all_metrics(st.session_state.y_test, predictions)
    
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("### 📐 Metrics Performance Breakdown")
        st.markdown(f"""
        <ul>
            <li><b>Accuracy ({metrics['accuracy']:.2%}):</b> Proportion of correct predictions over total predictions. <i>Warning: Can be highly misleading with class imbalance!</i></li>
            <li><b>Precision ({metrics['precision']:.2%}):</b> Out of all samples predicted as a specific class, how many actually belonged to it? (Minimizes False Positives).</li>
            <li><b>Recall ({metrics['recall']:.2%}):</b> Out of all actual samples of a class, how many did the model capture? (Minimizes False Negatives).</li>
            <li><b>F1-Score ({metrics['f1_score']:.2%}):</b> Harmonic mean of Precision and Recall.</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        #### Harmonic Mean Formula for F1-Score:
        """)
        st.markdown('<div class="formula-box">F1 = 2 * (Precision * Recall) / (Precision + Recall)</div>', unsafe_allow_html=True)
        
    with col2:
        cm, cm_df = generate_confusion_matrix_df(st.session_state.y_test, predictions, st.session_state.class_names)
        fig_cm = plot_confusion_matrix_heatmap(cm, st.session_state.class_names)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    st.markdown("### 🧮 Multi-Class Confusion Matrix Interpretation")
    st.markdown("""
    <div class="custom-card">
        <h4>One-vs-Rest (OvR) Perspective</h4>
        <p>In multi-class settings, we interpret the confusion matrix by focusing on one class at a time:</p>
        <ul>
            <li><b>True Positives (TP):</b> Actual Setosa predicted as Setosa (top-left diagonal cell).</li>
            <li><b>False Positives (FP):</b> Other species (Versicolor, Virginica) incorrectly predicted as Setosa. Sum of the Setosa column minus the diagonal.</li>
            <li><b>False Negatives (FN):</b> Actual Setosa incorrectly predicted as something else. Sum of the Setosa row minus the diagonal.</li>
            <li><b>True Negatives (TN):</b> All other correct or incorrect predictions that do not involve Setosa (all cells outside the Setosa row and column).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Scikit-Learn Classification Report")
    report_df = generate_report_df(st.session_state.y_test, predictions, st.session_state.class_names)
    st.dataframe(report_df.style.format(precision=4), use_container_width=True)

def render_prediction_page():
    st.title("🔮 Interactive Prediction Portal")
    st.write("Input physical measurements of a flower to predict its species using the fully trained KNN pipeline.")
    
    # Initialize default inputs in session state if missing
    if 'pred_sl' not in st.session_state:
        st.session_state.pred_sl = 5.1
        st.session_state.pred_sw = 3.5
        st.session_state.pred_pl = 1.4
        st.session_state.pred_pw = 0.2
        
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📝 Input Flower Dimensions")
        
        # Sample Presets
        st.markdown("<div style='font-size: 12px; color: #94A3B8; font-weight: 600; margin-bottom: 6px;'>SAMPLE FLOWER PRESETS:</div>", unsafe_allow_html=True)
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            if st.button("🌺 Setosa", use_container_width=True):
                st.session_state.pred_sl = 5.1
                st.session_state.pred_sw = 3.5
                st.session_state.pred_pl = 1.4
                st.session_state.pred_pw = 0.2
                st.session_state.trigger_pred = True
                st.rerun()
        with pcol2:
            if st.button("🌸 Versicolor", use_container_width=True):
                st.session_state.pred_sl = 6.0
                st.session_state.pred_sw = 2.7
                st.session_state.pred_pl = 5.1
                st.session_state.pred_pw = 1.6
                st.session_state.trigger_pred = True
                st.rerun()
        with pcol3:
            if st.button("🌷 Virginica", use_container_width=True):
                st.session_state.pred_sl = 6.9
                st.session_state.pred_sw = 3.1
                st.session_state.pred_pl = 5.4
                st.session_state.pred_pw = 2.1
                st.session_state.trigger_pred = True
                st.rerun()
                
        # User input fields with numeric validation
        sepal_length = st.number_input(
            "Sepal Length (cm)", min_value=3.0, max_value=10.0,
            value=float(st.session_state.pred_sl), step=0.1,
            help="Expected range: 4.0 - 8.0 cm"
        )
        sepal_width = st.number_input(
            "Sepal Width (cm)", min_value=1.5, max_value=6.0,
            value=float(st.session_state.pred_sw), step=0.1,
            help="Expected range: 2.0 - 4.5 cm"
        )
        petal_length = st.number_input(
            "Petal Length (cm)", min_value=0.5, max_value=8.0,
            value=float(st.session_state.pred_pl), step=0.1,
            help="Expected range: 1.0 - 7.0 cm"
        )
        petal_width = st.number_input(
            "Petal Width (cm)", min_value=0.05, max_value=3.5,
            value=float(st.session_state.pred_pw), step=0.1,
            help="Expected range: 0.1 - 2.5 cm"
        )
        
        # Keep session state updated with manual inputs
        st.session_state.pred_sl = sepal_length
        st.session_state.pred_sw = sepal_width
        st.session_state.pred_pl = petal_length
        st.session_state.pred_pw = petal_width
        
        predict_clicked = st.button("🔮 PREDICT SPECIES", use_container_width=True)
        
    with col2:
        st.markdown("### 🎯 Classification Result")
        
        should_run = predict_clicked or st.session_state.get('trigger_pred', False)
        if should_run:
            st.session_state.trigger_pred = False
            raw_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            scaled_input = st.session_state.scaler.transform_input(raw_input)
            pred_class_id = st.session_state.model.predict(scaled_input)[0]
            pred_probs = st.session_state.model.predict_proba(scaled_input)[0]
            
            st.session_state.last_prediction = {
                'pred_class_id': int(pred_class_id),
                'pred_probs': pred_probs,
                'raw_input': [sepal_length, sepal_width, petal_length, petal_width]
            }
            
        if 'last_prediction' in st.session_state:
            pred_info = st.session_state.last_prediction
            pred_class_id = pred_info['pred_class_id']
            pred_probs = pred_info['pred_probs']
            
            predicted_species = st.session_state.class_names[pred_class_id].upper()
            confidence = pred_probs[pred_class_id]
            theme_color = CLASS_COLORS[pred_class_id]
            
            st.markdown(f"""
            <div style="background-color:rgba(30, 41, 59, 0.7); border-radius:12px; border:2px solid {theme_color}; padding:24px; text-align:center;">
                <span style="font-size:14px; text-transform:uppercase; color:#94A3B8; letter-spacing:0.05em; font-weight:600;">Predicted Species</span>
                <h2 style="color:{theme_color}; margin: 8px 0; font-size:36px; font-weight:800;">{predicted_species}</h2>
                <div style="font-size:18px; font-weight:600; color:#FFFFFF; margin-bottom:16px;">
                    Prediction Confidence: {confidence:.2%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Probability Distribution:")
            for idx, cname in enumerate(st.session_state.class_names):
                prob = pred_probs[idx]
                st.write(f"**{cname.capitalize()}:** {prob:.2%}")
                st.progress(float(prob))
        else:
            st.info("Adjust the sliders/inputs or pick a preset on the left, then click Predict to classify the flower.")

    # Nearest Neighbors Diagnostic Table
    if 'last_prediction' in st.session_state:
        st.markdown(f"### 🔍 K-Nearest Neighbors Diagnostic (K = {st.session_state.k})")
        st.write("Below are the exact training data samples closest to your query point in standardized Euclidean space:")
        nn_df = get_nearest_neighbors_info(
            st.session_state.X_train,
            st.session_state.y_train,
            st.session_state.last_prediction['raw_input'],
            st.session_state.k,
            st.session_state.feature_names,
            st.session_state.class_names,
            st.session_state.scaler
        )
        st.dataframe(nn_df, use_container_width=True)

def render_how_it_works():
    st.title("🎓 Machine Learning Academy - How It Works")
    st.write("An educational guide breaking down fundamental concepts in Artificial Intelligence, Machine Learning, and Evaluation.")
    
    with st.expander("🤖 A. What is AI?"):
        st.write("""
        **Artificial Intelligence (AI)** is a broad field of computer science dedicated to building software systems capable of performing tasks that typically require human intelligence. Examples include visual perception, speech recognition, decision-making, and language translation.
        """)
        
    with st.expander("📈 B. What is Machine Learning?"):
        st.write("""
        **Machine Learning (ML)** is a subset of AI. Instead of writing explicit, rule-based instructions (e.g. `if sepal_length > 5.0 and petal_width < 0.3 then Setosa`), we write algorithms that learn patterns directly from data. As we feed the model more examples, it refines its parameters to improve performance.
        """)
        
    with st.expander("🏷️ C. What is Supervised Learning?"):
        st.write("""
        **Supervised Learning** is the most common paradigm in ML. The algorithm learns from a labeled dataset—meaning we supply both the input features (e.g., measurements) and the correct output labels (e.g., species species name). The goal is to learn a mapping function from inputs to outputs so we can predict labels on future, unseen data.
        """)
        
    with st.expander("🎯 D. What is Classification?"):
        st.write("""
        **Classification** is a supervised learning task where the output is a discrete category (a class label). Examples include classifying emails as spam or not spam, identifying diseases from medical images, or classifying Iris flowers into Setosa, Versicolor, or Virginica.
        """)
        
    with st.expander("🌸 E. What is the Iris Dataset?"):
        st.write("""
        Introduced by British statistician Ronald Fisher in 1936, the **Iris Dataset** is a classic benchmark. It consists of 150 samples of three Iris species (Setosa, Versicolor, Virginica), with 50 samples per class. For each sample, four physical features are measured in centimeters:
        1. Sepal Length
        2. Sepal Width
        3. Petal Length
        4. Petal Width
        """)
        
    with st.expander("⚖️ F. What is Feature Scaling?"):
        st.write("""
        **Feature Scaling** (or standardization) is a data preprocessing step. It transforms numerical features so they share a common scale. The standard scaler uses the formula:
        ```
        z = (x - mean) / std_deviation
        ```
        This centers features around 0 (mean = 0) with a standard deviation of 1.
        """)
        
    with st.expander("⚡ G. Why is Scaling Important for KNN?"):
        st.write("""
        K-Nearest Neighbors computes the **Euclidean distance** between features:
        ```
        d = sqrt( (x1-x2)^2 + (y1-y2)^2 + ... )
        ```
        If one feature spans a range of 100 and another spans a range of 1, the distance is entirely dominated by the first feature. Scaling puts all features on equal footing so that the distance calculation is balanced.
        """)
        
    with st.expander("✂️ H. What is Train-Test Split?"):
        st.write("""
        To evaluate a model's true performance, we must test it on data it hasn't seen during training. A **Train-Test Split** splits the original dataset (e.g. 80% for training and 20% for testing). Shuffling before splitting prevents order bias.
        """)
        
    with st.expander("🧠 I. What is K-Nearest Neighbors (KNN)?"):
        st.write("""
        **KNN** is a non-parametric, distance-based supervised classifier. To classify an unknown point, it calculates the Euclidean distance between that point and all training examples, picks the K closest points, and takes a majority vote of their classes.
        """)
        
    with st.expander("🎛️ J. What does K mean?"):
        st.write("""
        **K** is a hyperparameter indicating the number of neighbors checked.
        - A small K (e.g., K=1) is highly sensitive to noise and can overfit.
        - A large K (e.g., K=20) is robust to noise but can underfit local patterns.
        """)
        
    with st.expander("📊 K. What is a Confusion Matrix?"):
        st.write("""
        A **Confusion Matrix** is a table showing the actual vs predicted classes. It provides a visual breakdown of correct classifications (diagonal cells) and specific misclassification categories.
        """)
        
    with st.expander("🔍 L. What are TP, TN, FP, and FN?"):
        st.write("""
        - **True Positive (TP):** Model correctly predicted positive class.
        - **True Negative (TN):** Model correctly predicted negative class.
        - **False Positive (FP):** Model predicted positive, but it was actually negative (Type I Error).
        - **False Negative (FN):** Model predicted negative, but it was actually positive (Type II Error).
        """)
        
    with st.expander("🎯 M. What are Precision and Recall?"):
        st.write("""
        - **Precision:** Of the samples predicted as positive, how many were actually positive?
          `Precision = TP / (TP + FP)`
        - **Recall (Sensitivity):** Of all actual positive samples, how many did the model correctly identify?
          `Recall = TP / (TP + FN)`
        """)
        
    with st.expander("🧬 N. What is the F1-Score?"):
        st.write("""
        **F1-Score** is the harmonic mean of Precision and Recall. It balances both metrics, especially when there is class imbalance:
        ```
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        ```
        """)
        
    with st.expander("⚠️ O. Why is Accuracy not always enough?"):
        st.write("""
        In imbalanced datasets (e.g., a disease classification where 99% are healthy), a model that predicts "healthy" for everything gets 99% accuracy but fails completely at detecting the disease. Hence, Precision, Recall, and F1-Score are vital.
        """)
