import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer, KNNImputer
import joblib
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLModelTrainer:
    """
    Enhanced Machine Learning Model Trainer with dynamic model selection and optimization.
    """
    
    def __init__(self, base_dir: str = ".", random_state: int = 42):
        """
        Initialize the ML Model Trainer.
        
        Args:
            base_dir: Base directory for saving models and preprocessors
            random_state: Random seed for reproducibility
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "Models"
        self.preprocessors_dir = self.base_dir / "Preprocessors"
        self.random_state = random_state
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.preprocessors_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self._setup_logging()
        
        # Available models
        self.available_models = {
            'random_forest': RandomForestClassifier(random_state=random_state, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=random_state),
            'extra_trees': ExtraTreesClassifier(random_state=random_state, n_jobs=-1),
            'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'svm': SVC(random_state=random_state),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(),
            'decision_tree': DecisionTreeClassifier(random_state=random_state)
        }
        
        # Available scalers
        self.available_scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        # Available imputers
        self.available_imputers = {
            'simple_mean': SimpleImputer(strategy='mean'),
            'simple_median': SimpleImputer(strategy='median'),
            'simple_mode': SimpleImputer(strategy='most_frequent'),
            'knn': KNNImputer(n_neighbors=5)
        }
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"ml_training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_and_validate_data(self, csv_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load and validate CSV data.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Tuple of (DataFrame, summary_dict)
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        self.logger.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if df.empty:
            raise ValueError("CSV file is empty")
        if len(df.columns) < 2:
            raise ValueError("CSV must have at least 2 columns (features + target)")
            
        # Generate data summary
        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'features': df.columns[:-1].tolist(),
            'target_column': df.columns[-1],
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'target_classes': df.iloc[:, -1].value_counts().to_dict() if df.iloc[:, -1].dtype == 'object' else None
        }
        
        self.logger.info(f"Data loaded successfully: {summary['rows']} rows, {summary['columns']} columns")
        return df, summary
        
    def preprocess_data(self, 
                       df: pd.DataFrame, 
                       imputation_strategy: str = 'simple_median',
                       scaling_method: str = 'standard') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Preprocess the data with configurable strategies.
        
        Args:
            df: Input DataFrame
            imputation_strategy: Strategy for handling missing values
            scaling_method: Method for scaling features
            
        Returns:
            Tuple of (X_processed, y, preprocessors_dict)
        """
        self.logger.info("Starting data preprocessing...")
        
        # Split features and target
        X = df.iloc[:, :-1].copy()
        y = df.iloc[:, -1].copy()
        
        preprocessors = {}
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        self.logger.info(f"Found {len(categorical_cols)} categorical and {len(numeric_cols)} numeric columns")
        
        # Encode categorical variables
        label_encoders = {}
        if len(categorical_cols) > 0:
            self.logger.info("Encoding categorical variables...")
            for col in categorical_cols:
                le = LabelEncoder()
                # Handle missing values in categorical columns
                X[col] = X[col].fillna('missing')
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
                
        preprocessors['label_encoders'] = label_encoders
        
        # Handle missing values in numeric columns
        if len(numeric_cols) > 0 and X[numeric_cols].isnull().any().any():
            self.logger.info(f"Handling missing values using {imputation_strategy}...")
            if imputation_strategy in self.available_imputers:
                imputer = self.available_imputers[imputation_strategy]
                X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
                preprocessors['imputer'] = imputer
            else:
                self.logger.warning(f"Unknown imputation strategy: {imputation_strategy}, using median")
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Scale features
        self.logger.info(f"Scaling features using {scaling_method}...")
        if scaling_method in self.available_scalers:
            scaler = self.available_scalers[scaling_method]
            X_scaled = scaler.fit_transform(X)
            preprocessors['scaler'] = scaler
        else:
            self.logger.warning(f"Unknown scaling method: {scaling_method}, using standard scaling")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            preprocessors['scaler'] = scaler
            
        return X_scaled, y, preprocessors
        
    def train_single_model(self, 
                          X_train: np.ndarray, 
                          X_test: np.ndarray, 
                          y_train: np.ndarray, 
                          y_test: np.ndarray,
                          model_name: str,
                          hyperparameter_tuning: bool = False) -> Dict:
        """
        Train a single model and evaluate its performance.
        
        Args:
            X_train, X_test, y_train, y_test: Train/test splits
            model_name: Name of the model to train
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with model performance metrics
        """
        if model_name not in self.available_models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.available_models.keys())}")
            
        self.logger.info(f"Training {model_name}...")
        model = self.available_models[model_name]
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            self.logger.info(f"Performing hyperparameter tuning for {model_name}...")
            model = self._tune_hyperparameters(model, X_train, y_train, model_name)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'model': model
        }
        
        self.logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return results
        
    def _tune_hyperparameters(self, model, X_train, y_train, model_name):
        """Perform basic hyperparameter tuning."""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=3, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        
        return model
        
    def train_multiple_models(self, 
                             X_scaled: np.ndarray, 
                             y: np.ndarray,
                             models_to_train: Optional[list] = None,
                             test_size: float = 0.2,
                             hyperparameter_tuning: bool = False) -> Dict:
        """
        Train multiple models and compare their performance.
        
        Args:
            X_scaled: Preprocessed features
            y: Target variable
            models_to_train: List of model names to train (None for all)
            test_size: Proportion of data for testing
            hyperparameter_tuning: Whether to tune hyperparameters
            
        Returns:
            Dictionary with all model results
        """
        if models_to_train is None:
            models_to_train = list(self.available_models.keys())
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state
        )
        
        results = {}
        
        for model_name in models_to_train:
            try:
                result = self.train_single_model(
                    X_train, X_test, y_train, y_test, 
                    model_name, hyperparameter_tuning
                )
                results[model_name] = result
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
                
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        results['best_model'] = best_model_name
        results['best_accuracy'] = results[best_model_name]['accuracy']
        
        self.logger.info(f"Best model: {best_model_name} with accuracy: {results['best_accuracy']:.4f}")
        return results
        
    def save_models_and_preprocessors(self, 
                                    results: Dict, 
                                    preprocessors: Dict,
                                    experiment_name: str = None) -> Dict[str, str]:
        """
        Save trained models and preprocessors.
        
        Args:
            results: Results from model training
            preprocessors: Preprocessing objects
            experiment_name: Name for this experiment
            
        Returns:
            Dictionary with file paths
        """
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        saved_files = {}
        
        # Save best model
        best_model_name = results['best_model']
        best_model = results[best_model_name]['model']
        
        model_filename = f"{experiment_name}_{best_model_name}_best.pkl"
        model_path = self.models_dir / model_filename
        joblib.dump(best_model, model_path)
        saved_files['best_model'] = str(model_path)
        
        # Save all models
        all_models_path = self.models_dir / f"{experiment_name}_all_models.pkl"
        all_models = {name: result['model'] for name, result in results.items() 
                     if name not in ['best_model', 'best_accuracy']}
        joblib.dump(all_models, all_models_path)
        saved_files['all_models'] = str(all_models_path)
        
        # Save preprocessors
        preprocessor_path = self.preprocessors_dir / f"{experiment_name}_preprocessors.pkl"
        joblib.dump(preprocessors, preprocessor_path)
        saved_files['preprocessors'] = str(preprocessor_path)
        
        # Save results summary
        results_path = self.models_dir / f"{experiment_name}_results.pkl"
        joblib.dump(results, results_path)
        saved_files['results'] = str(results_path)
        
        self.logger.info(f"Saved files: {saved_files}")
        return saved_files
        
    def run_complete_pipeline(self, 
                             csv_path: str,
                             models_to_train: Optional[list] = None,
                             imputation_strategy: str = 'simple_median',
                             scaling_method: str = 'standard',
                             test_size: float = 0.2,
                             hyperparameter_tuning: bool = False,
                             experiment_name: str = None) -> Dict:
        """
        Run the complete ML pipeline from data loading to model saving.
        
        Args:
            csv_path: Path to input CSV file
            models_to_train: List of models to train
            imputation_strategy: Strategy for missing value imputation
            scaling_method: Method for feature scaling
            test_size: Test set proportion
            hyperparameter_tuning: Whether to tune hyperparameters
            experiment_name: Name for this experiment
            
        Returns:
            Complete results dictionary
        """
        try:
            # Load and validate data
            df, data_summary = self.load_and_validate_data(csv_path)
            
            # Preprocess data
            X_scaled, y, preprocessors = self.preprocess_data(
                df, imputation_strategy, scaling_method
            )
            
            # Train models
            model_results = self.train_multiple_models(
                X_scaled, y, models_to_train, test_size, hyperparameter_tuning
            )
            
            # Save everything
            saved_files = self.save_models_and_preprocessors(
                model_results, preprocessors, experiment_name
            )
            
            # Compile final results
            final_results = {
                'data_summary': data_summary,
                'model_results': model_results,
                'saved_files': saved_files,
                'preprocessing_config': {
                    'imputation_strategy': imputation_strategy,
                    'scaling_method': scaling_method,
                    'test_size': test_size
                }
            }
            
            self.logger.info("Pipeline completed successfully!")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = MLModelTrainer()
    
    # Define models to train (or None for all)
    models_to_train = ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm']
    
    try:
        # Run complete pipeline
        results = trainer.run_complete_pipeline(
            csv_path="data.csv",
            models_to_train=models_to_train,
            imputation_strategy='simple_median',
            scaling_method='robust',
            test_size=0.2,
            hyperparameter_tuning=True,
            experiment_name="my_experiment"
        )
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Best Model: {results['model_results']['best_model']}")
        print(f"Best Accuracy: {results['model_results']['best_accuracy']:.4f}")
        print(f"Models saved in: {trainer.models_dir}")
        print(f"Preprocessors saved in: {trainer.preprocessors_dir}")
        
    except Exception as e:
        print(f"Training failed: {e}")