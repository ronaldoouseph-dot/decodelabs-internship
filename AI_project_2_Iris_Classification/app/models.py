from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_knn(X_train_scaled, y_train, k=5):
    """
    Instantiates and fits a K-Nearest Neighbors classifier.
    
    Args:
        X_train_scaled: Scaled training feature array
        y_train: Training labels
        k: Number of neighbors (default 5)
    Returns:
        model: Trained KNeighborsClassifier model
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    return model

def analyze_k_values(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Trains KNN models for K values from 1 to 20 and computes metrics.
    
    Returns:
        k_values: list of range 1-20
        accuracies: list of model accuracies
        error_rates: list of model error rates (1 - accuracy)
    """
    k_values = list(range(1, 21))
    accuracies = []
    error_rates = []
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, predictions)
        accuracies.append(acc)
        error_rates.append(1.0 - acc)
        
    return k_values, accuracies, error_rates
