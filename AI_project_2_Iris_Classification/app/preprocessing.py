# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.preprocessing import StandardScaler

class IrisScaler:
    """
    Manages scaling lifecycle to ensure no data leakage.
    Features are scaled using StandardScaler: mean = 0, variance = 1.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fit = False

    def fit_transform_train(self, X_train):
        """
        Fits the scaler on the training data and transforms it.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self._is_fit = True
        return X_train_scaled

    def transform_test(self, X_test):
        """
        Transforms testing data using the scaler fitted on training data.
        Crucial: Never fit on test data.
        """
        if not self._is_fit:
            raise ValueError("StandardScaler must be fitted on training data first!")
        return self.scaler.transform(X_test)

    def transform_input(self, single_input):
        """
        Transforms a custom 1D or 2D input array/list for prediction.
        Reshapes 1D input (4,) to 2D (1, 4) and checks feature count.
        """
        if not self._is_fit:
            raise ValueError("StandardScaler must be fitted on training data first!")
        arr = np.asarray(single_input, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(f"Input must have 4 features (Sepal/Petal lengths & widths), got shape {arr.shape}")
        return self.scaler.transform(arr)
