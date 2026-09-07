import numpy as np
import pandas as pd
from app.data import load_raw_data, get_train_test_split_data
from app.preprocessing import IrisScaler
from app.models import train_knn, analyze_k_values
from app.evaluation import compute_all_metrics, generate_confusion_matrix_df, generate_report_df

def test_pipeline():
    print("--- STARTING PIPELINE VERIFICATION ---")
    
    # 1. Dataset Verification
    print("1. Loading raw dataset...")
    iris, df = load_raw_data()
    assert df.shape == (150, 6), f"Expected shape (150, 6) including Species and Target columns, got {df.shape}"
    assert len(iris.target_names) == 3, "Expected 3 classes"
    assert len(iris.feature_names) == 4, "Expected 4 features"
    print("✓ Dataset loads correctly with 150 samples, 4 features, and 3 classes.")
    
    # 2. Split Verification
    print("2. Verifying train/test splits...")
    X_train, X_test, y_train, y_test, feature_names, class_names = get_train_test_split_data(test_size=0.20, random_state=42)
    assert X_train.shape == (120, 4), f"Expected 120 train samples, got {X_train.shape[0]}"
    assert X_test.shape == (30, 4), f"Expected 30 test samples, got {X_test.shape[0]}"
    assert len(y_train) == 120 and len(y_test) == 30
    
    # Verify stratification (10 samples of each class in test set, 40 in train)
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    assert np.all(train_counts == 40), f"Expected stratified train counts [40, 40, 40], got {train_counts}"
    assert np.all(test_counts == 10), f"Expected stratified test counts [10, 10, 10], got {test_counts}"
    print("✓ Split is exactly 80/20 (120 train, 30 test) and correctly stratified.")
    
    # 3. Scaling & Data Leakage Verification
    print("3. Testing StandardScaler and checking for data leakage...")
    scaler = IrisScaler()
    X_train_scaled = scaler.fit_transform_train(X_train)
    
    # Check that mean is close to 0 and variance is close to 1 on training set
    means = np.mean(X_train_scaled, axis=0)
    stds = np.std(X_train_scaled, axis=0)
    assert np.allclose(means, 0, atol=1e-7), f"Expected scaled means close to 0, got {means}"
    assert np.allclose(stds, 1, atol=1e-7), f"Expected scaled stds close to 1, got {stds}"
    
    # Transform test set
    X_test_scaled = scaler.transform_test(X_test)
    # Check that scaler parameters didn't change (the mean/scale must be the same as X_train mean/scale)
    original_mean = np.copy(scaler.scaler.mean_)
    original_scale = np.copy(scaler.scaler.scale_)
    
    # Triggering transform_test again should not affect mean/scale
    _ = scaler.transform_test(X_test)
    assert np.all(scaler.scaler.mean_ == original_mean), "Scaler mean changed during test transform! Data leakage detected."
    assert np.all(scaler.scaler.scale_ == original_scale), "Scaler scale changed during test transform! Data leakage detected."
    print("✓ StandardScaler works correctly. Fitting is restricted to train set, preventing data leakage.")
    
    # 4. Model Training & Prediction
    print("4. Training KNN with default K=5...")
    model = train_knn(X_train_scaled, y_train, k=5)
    test_predictions = model.predict(X_test_scaled)
    assert test_predictions.shape == (30,), f"Expected 30 predictions, got {test_predictions.shape}"
    print("✓ KNN Model trains successfully and makes predictions.")
    
    # 5. Metrics & Evaluation Verification
    print("5. Evaluating metrics...")
    metrics = compute_all_metrics(y_test, test_predictions)
    print(f"Computed Metrics: {metrics}")
    assert 'accuracy' in metrics and 'precision' in metrics and 'recall' in metrics and 'f1_score' in metrics
    assert 0.0 <= metrics['accuracy'] <= 1.0
    
    # Confusion matrix
    cm, cm_df = generate_confusion_matrix_df(y_test, test_predictions, class_names)
    assert cm.shape == (3, 3), f"Expected 3x3 confusion matrix, got {cm.shape}"
    print("✓ Metrics (Accuracy, Precision, Recall, F1) and Confusion Matrix calculated successfully.")
    
    # 6. Interactive Prediction Pipeline
    print("6. Verifying interactive prediction flow...")
    test_query = np.array([[5.1, 3.5, 1.4, 0.2]]) # Setosa
    scaled_query = scaler.transform_input(test_query)
    pred_class = model.predict(scaled_query)[0]
    pred_probs = model.predict_proba(scaled_query)[0]
    
    predicted_label = class_names[pred_class]
    print(f"Query {test_query} -> Predicted Species: {predicted_label.upper()} with probabilities {pred_probs}")
    assert predicted_label == 'setosa', f"Expected Setosa prediction for input {test_query}, got {predicted_label}"
    assert np.allclose(np.sum(pred_probs), 1.0), "Prediction probabilities must sum to 1.0"
    # 7. Model Tuning (K values 1 to 20)
    print("7. Verifying K value analysis loop...")
    k_vals, accs, errs = analyze_k_values(X_train_scaled, y_train, X_test_scaled, y_test)
    assert len(k_vals) == 20
    assert len(accs) == 20
    assert len(errs) == 20
    print("✓ K-value evaluation from 1 to 20 works successfully.")
    
    # 8. Scaler Robustness (1D vs 2D array & List handling)
    print("8. Verifying IrisScaler robustness for 1D, 2D, and list inputs...")
    t_list = [5.1, 3.5, 1.4, 0.2]
    t_1d = np.array([5.1, 3.5, 1.4, 0.2])
    t_2d = np.array([[5.1, 3.5, 1.4, 0.2]])
    
    out_list = scaler.transform_input(t_list)
    out_1d = scaler.transform_input(t_1d)
    out_2d = scaler.transform_input(t_2d)
    assert np.allclose(out_list, out_1d) and np.allclose(out_1d, out_2d)
    assert out_list.shape == (1, 4)
    
    # Assert dimension check works
    try:
        scaler.transform_input([5.1, 3.5, 1.4])
        assert False, "Expected ValueError on 3-feature input"
    except ValueError:
        pass
    print("✓ Scaler correctly handles 1D arrays, lists, 2D arrays, and validates feature dimensions.")
    
    # 9. Plotly Visualizations & Diagnostic Verification
    print("9. Verifying Plotly visualization engines & nearest-neighbor extraction...")
    from app.visualization import (
        plot_interactive_scatter,
        plot_scaling_comparison,
        plot_train_test_split_dist,
        plot_k_error_rates,
        plot_confusion_matrix_heatmap,
        plot_knn_space,
        get_nearest_neighbors_info
    )
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    fig_scatter = plot_interactive_scatter(df, feature_names[0], feature_names[1])
    assert fig_scatter is not None
    
    fig_scale = plot_scaling_comparison(df, X_train_scaled_df, feature_names[0])
    assert fig_scale is not None
    
    fig_dist = plot_train_test_split_dist(y_train, y_test, class_names)
    assert fig_dist is not None
    
    fig_k = plot_k_error_rates(k_vals, errs, selected_k=5)
    assert fig_k is not None
    
    fig_cm = plot_confusion_matrix_heatmap(cm, class_names)
    assert fig_cm is not None
    
    fig_space, neighbor_details = plot_knn_space(
        X_train, y_train, np.array([5.1, 3.5, 1.4, 0.2]), 5, (2, 3), feature_names, class_names, scaler
    )
    assert fig_space is not None
    assert 'vote_counts' in neighbor_details and sum(neighbor_details['vote_counts'].values()) == 5
    
    nn_table = get_nearest_neighbors_info(X_train, y_train, [5.1, 3.5, 1.4, 0.2], 5, feature_names, class_names, scaler)
    assert len(nn_table) == 5
    assert 'Scaled Distance' in nn_table.columns
    print("✓ All 6 Plotly visualization engines and neighbor extraction diagnostics run cleanly.")
    
    # 10. Classification Report Verification
    print("10. Verifying classification report...")
    rep_df = generate_report_df(y_test, test_predictions, class_names)
    assert 'precision' in rep_df.columns and 'recall' in rep_df.columns and 'f1-score' in rep_df.columns
    print("✓ Classification report DataFrame generated successfully.")
    
    print("\n✓✓✓ ALL PIPELINE TESTS (STEPS 1-10) PASSED SUCCESSFULLY! ✓✓✓")

if __name__ == '__main__':
    test_pipeline()
