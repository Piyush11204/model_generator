import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_predict(model_path: str, preprocessor_path: str, test_data: pd.DataFrame) -> np.ndarray:
    """
    Load a trained model and preprocessors, preprocess test data, and make predictions.

    Args:
        model_path (str): Path to the saved model (.pkl).
        preprocessor_path (str): Path to the saved preprocessors (.pkl).
        test_data (pd.DataFrame): Test data with the same features as training data.

    Returns:
        np.ndarray: Model predictions.
    """
    try:
        # Load model and preprocessors
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Loading preprocessors from {preprocessor_path}")
        preprocessors = joblib.load(preprocessor_path)
        scaler = preprocessors['scaler']
        label_encoders = preprocessors['label_encoders']

        # Validate test data
        expected_features = list(label_encoders.keys()) + [col for col in test_data.columns if col not in label_encoders]
        if not all(col in test_data.columns for col in expected_features):
            raise ValueError(f"Test data must contain features: {expected_features}")

        # Preprocess test data
        X_test = test_data.copy()
        
        # Handle missing values
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns
        categorical_cols = X_test.select_dtypes(include=['object', 'category']).columns
        X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].median())
        X_test[categorical_cols] = X_test[categorical_cols].fillna('missing')

        # Encode categorical variables
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    X_test[col] = label_encoders[col].transform(X_test[col])
                except ValueError as e:
                    logger.error(f"Error encoding column {col}: {e}")
                    raise ValueError(f"Column {col} contains unseen categories. Ensure test data matches training data categories.")
            else:
                logger.warning(f"No encoder found for {col}. Applying new encoding.")
                X_test[col] = LabelEncoder().fit_transform(X_test[col])

        # Scale features
        X_test_scaled = scaler.transform(X_test)

        # Make predictions
        logger.info("Making predictions")
        predictions = model.predict(X_test_scaled)
        return predictions

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        # Load test data (replace with your CSV or manual input)
        test_data = pd.DataFrame({
            'City': ['Shimla', 'Jaipur'],
            'Tourist_Numbers': [50000, 75000],
            'Growth_Rate': [0.05, 0.03],
            'Season': ['Summer', 'Winter']
        })

        # Paths to saved model and preprocessors
        model_path = "model.pkl"
        preprocessor_path = "preprocessors.pkl"

        # Make predictions
        predictions = load_and_predict(model_path, preprocessor_path, test_data)
        logger.info(f"Predictions: {predictions}")

        # Display results
        test_data['Predicted_Rating'] = predictions
        print(test_data)

    except Exception as e:
        print(f"Failed to make predictions: {e}")