# AI Iris Data Classification System

A comprehensive, educational, supervised machine-learning dashboard designed to classification using the classic Iris dataset, feature standardization, and the K-Nearest Neighbors (KNN) algorithm.

---

## 🎯 Project Objective
The goal of this project is to demonstrate a complete machine learning pipeline: from data ingestion and understanding, feature standardization, stratified train-test splitting, KNN model training and hyperparameter tuning, through to robust metrics evaluation and interactive live predictions.

---

## 🎨 Key Features
1. **Interactive Dashboard:** High-level summary metrics, key statistics, and a visual flowchart of the machine learning pipeline.
2. **Dataset Details Explorer:** Searchable raw data table and dynamic 2D scatter plots of Iris physical measurements.
3. **Interactive Preprocessing Visuals:** Before-and-after standardization distributions demonstrating standard scaling.
4. **Train / Test Split Auditor:** Clear visualization showing split ratio and stratification class counts.
5. **Interactive KNN Sandbox:** Slide to select different $K$ values, look at error rates dynamically, and project neighborhood connections in a 2D plot.
6. **Robust Evaluation Suite:** Interactive confusion matrix heatmap and a detailed classification report mapping Precision, Recall, and F1 scores.
7. **Predictive Portal:** Input custom sepal/petal dimensions and perform live predictions with confidence score distribution bars.
8. **ML Academy:** A structured glossary explaining ML, scaling, metrics, and algorithms for beginners.

---

## 🛠️ Technology Stack
- **Core Language:** Python 3.14+
- **Machine Learning Library:** `scikit-learn`
- **Data Engineering:** `pandas`, `numpy`
- **Visualization:** `plotly` (interactive), `matplotlib` (internal/fallback)
- **Web App Interface:** `streamlit`

---

## 📊 Dataset & Pipeline Details
- **Name:** Fisher's Iris Dataset
- **Samples:** 150 (perfectly balanced 50 per class)
- **Classes:** 3 Species
  1. *Iris Setosa*
  2. *Iris Versicolor*
  3. *Iris Virginica*
- **Features:** 4 Physical measurements
  1. Sepal Length (cm)
  2. Sepal Width (cm)
  3. Petal Length (cm)
  4. Petal Width (cm)

### 1. Data Preprocessing (Scaling)
Feature standardization is mandatory for distance-based models. We apply:
$$z = \frac{x - \mu}{\sigma}$$
*Mean ($\mu$) centered at 0, standard deviation ($\sigma$) of 1.*
**Data Leakage Prevention Rule:** The `StandardScaler` is fitted *only* on the 80% training data, and then used to transform both the train and test splits.

### 2. Train / Test Split
- **Training Set:** 80% (120 samples)
- **Testing Set:** 20% (30 samples)
- **Stratification:** Maintains equal representation of Setosa, Versicolor, and Virginica (10 samples of each in test, 40 of each in train).
- **Shuffle:** Randomized before split to avoid bias.
- **Reproducibility:** Seeded with `random_state=42`.

### 3. K-Nearest Neighbors (KNN)
To classify a new point:
1. Calculates the **Euclidean Distance** to all training set points.
2. Selects the **K** nearest points.
3. Checks class labels of neighbors and performs a **majority vote**.
4. Default is **K=5**.

---

## 📈 Evaluation Metrics Explained
- **Accuracy:** General correct predictions proportion. *Note: Can be misleading with imbalanced classes.*
- **Precision:** $\frac{TP}{TP + FP}$. Minimizes False Positives.
- **Recall:** $\frac{TP}{TP + FN}$. Minimizes False Negatives.
- **F1 Score:** Harmonic mean of Precision and Recall:
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 📁 Project Architecture
```
project/
├── app/
│   ├── __init__.py
│   ├── data.py          # Data loaders and split functions
│   ├── preprocessing.py # Scaling logic with StandardScaler
│   ├── models.py        # KNN fitters and K-analysis loop
│   ├── evaluation.py    # Metric computations and DF generators
│   ├── visualization.py # Custom Plotly chart engines
│   ├── ui.py            # Page layouts and Streamlit renderers
│   └── styles.css       # Custom CSS styling variables
├── app.py               # Main navigation entry point
├── test_pipeline.py     # Comprehensive ML pipeline & visualization tests (10 steps)
├── test_app.py          # End-to-end Streamlit AppTest integration suite
├── requirements.txt     # List of library dependencies
└── README.md            # Technical documentation
```

---

## 🧪 How to Run Automated Tests

Run the complete pipeline verification (10 automated checkpoints) and the Streamlit AppTest integration suite:

```bash
# 1. Pipeline & scaling verification
wsl python3 test_pipeline.py

# 2. End-to-end Streamlit UI & interaction testing
wsl python3 test_app.py
```

---

## 🚀 How to Run the Application

If running under Windows Subsystem for Linux (WSL) or a standard Linux/macOS terminal:

1. Clone or navigate to the project directory:
   ```bash
   cd "c:\Users\acer\OneDrive\Desktop\AI chatbot\AI_project_2_Iris_Classification"
   ```

2. Run the Streamlit web dashboard:
   ```bash
   wsl /home/acer/.local/bin/streamlit run app.py
   ```

3. Streamlit will start a web server and display network URLs. Navigate to the local URL (usually `http://localhost:8501`) in your browser to view the application.

---

## 🔮 Example Prediction Input/Output
- **Inputs:**
  - Sepal Length: `5.1 cm`
  - Sepal Width: `3.5 cm`
  - Petal Length: `1.4 cm`
  - Petal Width: `0.2 cm`
- **Output:**
  - Predicted Species: **SETOSA**
  - Confidence: **100.00%** (all 5 nearest neighbors belong to the Setosa class)

---

## 🎓 Learning Outcomes
- Implementing a complete clean machine learning lifecycle.
- Preventing data leakage during scaling and preprocessing.
- Tuning hyperparameters ($K$ value) and identifying overfitting vs underfitting.
- Visualizing multi-class models using heatmaps and 2D projections.
