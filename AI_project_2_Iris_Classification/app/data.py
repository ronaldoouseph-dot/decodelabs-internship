import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_raw_data():
    """
    Loads the classic Iris dataset programmatically.
    Returns:
        iris: sklearn Bunch object containing metadata.
        df: pandas DataFrame containing features and target names.
    """
    iris = load_iris()
    # Create DataFrame from iris data
    df = pd.DataFrame(data=iris.data, columns=[
        'Sepal Length (cm)',
        'Sepal Width (cm)',
        'Petal Length (cm)',
        'Petal Width (cm)'
    ])
    df['Species'] = [iris.target_names[t] for t in iris.target]
    df['Target'] = iris.target
    return iris, df

def get_train_test_split_data(test_size=0.20, random_state=42):
    """
    Splits the dataset into training and testing splits.
    Applies shuffling and stratification.
    
    Returns:
        X_train, X_test, y_train, y_test: arrays of features/targets
        feature_names: list of feature names
        class_names: list of target class names
    """
    iris, _ = load_raw_data()
    X = iris.data
    y = iris.target
    
    # Split into 80% train, 20% test, with stratification and shuffle (default in train_test_split)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    feature_names = [
        'Sepal Length (cm)',
        'Sepal Width (cm)',
        'Petal Length (cm)',
        'Petal Width (cm)'
    ]
    class_names = list(iris.target_names) # ['setosa', 'versicolor', 'virginica']
    
    return X_train, X_test, y_train, y_test, feature_names, class_names
