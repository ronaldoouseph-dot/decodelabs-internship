from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd

def compute_all_metrics(y_test, y_pred):
    """
    Computes accuracy, precision, recall, and F1 score.
    Uses weighted average to handle multi-class targets.
    
    Returns:
        metrics: dict of metric names and values.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def generate_confusion_matrix_df(y_test, y_pred, class_names):
    """
    Generates a confusion matrix DataFrame with labeled rows and columns.
    
    Returns:
        cm: raw numpy confusion matrix
        cm_df: labeled pandas DataFrame
    """
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c.capitalize()}" for c in class_names],
        columns=[f"Predicted {c.capitalize()}" for c in class_names]
    )
    return cm, cm_df

def generate_report_df(y_test, y_pred, class_names):
    """
    Generates a detailed classification report as a pandas DataFrame.
    """
    report_dict = classification_report(y_test, y_pred, target_names=[c.capitalize() for c in class_names], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    return report_df
