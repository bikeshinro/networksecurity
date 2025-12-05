import os
import sys

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig



from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


os.environ["MLFLOW_ENABLE_LOGGED_MODELS"] = "false"
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/bikeshinro/networksecurity/mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="bikeshinro"
os.environ["MLFLOW_TRACKING_PASSWORD"]="b27a913b272e42564d8315cb120af19684ccd3e8"

import dagshub
dagshub.init(repo_owner='bikeshinro', repo_name='networksecurity', mlflow=True)

import mlflow
from urllib.parse import urlparse



class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        self.config = model_trainer_config
        self.artifact = data_transformation_artifact

    def _log_to_mlflow(self, model, metrics: dict):
        mlflow.set_registry_uri("https://dagshub.com/bikeshinro/networksecurity/mlflow")

        with mlflow.start_run():
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log model (no registry)
            mlflow.sklearn.log_model(model, name="model")

    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
            },
            "Random Forest": {
                "n_estimators": [8, 16, 32, 128, 256],
            },
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Logistic Regression": {},
            "AdaBoost": {
                "learning_rate": [0.1, 0.01, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
        }

        # Evaluate all models
        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        # Select best model
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        # Compute metrics
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)

        train_metrics = get_classification_score(y_train, train_pred)
        test_metrics = get_classification_score(y_test, test_pred)

        # Log to MLflow
        self._log_to_mlflow(
            best_model,
            {
                "train_f1": train_metrics.f1_score,
                "train_precision": train_metrics.precision_score,
                "train_recall": train_metrics.recall_score,
                "test_f1": test_metrics.f1_score,
                "test_precision": test_metrics.precision_score,
                "test_recall": test_metrics.recall_score,
            },
        )

        # Save final model pipeline
        preprocessor = load_object(self.artifact.transformed_object_file_path)
        final_model = NetworkModel(preprocessor=preprocessor, model=best_model)

        os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
        save_object(self.config.trained_model_file_path, final_model)

        # Also save raw model for convenience
        save_object("final_model/model.pkl", best_model)

        return ModelTrainerArtifact(
            trained_model_file_path=self.config.trained_model_file_path,
            train_metric_artifact=train_metrics,
            test_metric_artifact=test_metrics,
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        train_arr = load_numpy_array_data(self.artifact.transformed_train_file_path)
        test_arr = load_numpy_array_data(self.artifact.transformed_test_file_path)

        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        return self.train_model(X_train, y_train, X_test, y_test)