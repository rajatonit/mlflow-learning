import os
import warnings
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import mlflow
import mlflow.sklearn
from mlflow.models import make_metric, MetricThreshold

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--penality", type=str, required=False, default="l2")
parser.add_argument("--C", type=float, required=False, default=1.0)
args = parser.parse_args()

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='weighted')
    recall = recall_score(actual, pred, average='weighted')
    f1 = f1_score(actual, pred, average='weighted')
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    # Read the dataset
    data = pd.read_csv('iris.csv')
    # Split the data into training and test sets.
    train, test = train_test_split(data)
    # Data Preprocessing
    le = LabelEncoder()
    train['variety'] = le.fit_transform(train['variety'])
    test['variety'] = le.fit_transform(test['variety'])

    # Storing the training and testing dataset
    if not os.path.isdir('data'):
        os.mkdir('data')
    train.to_csv("data/train.csv")
    test.to_csv("data/test.csv")

    # Split
    train_x = train.drop(["variety"], axis=1)
    test_x = test.drop(["variety"], axis=1)
    train_y = train[["variety"]]
    test_y = test[["variety"]]

    # Hyperparameters
    penality = args.penality
    C = args.C

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    experiment = mlflow.set_experiment(
        experiment_name="Classifier"
    )

    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("Creation timestamp: {}".format(experiment.creation_time))

    with mlflow.start_run( experiment_id=experiment.experiment_id):

        # Model: Dummy Classifier
        lr = DummyClassifier(strategy="uniform")
        lr.fit(train_x, train_y)

        predicted_classes = lr.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"Dummy Classifier model")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        mlflow.log_param("strategy", "uniform")
        mlflow.sklearn.log_model(lr, "baseline_model")
        baseline_model_uri = mlflow.get_artifact_uri('baseline_model')

        # Model: Logistic Regression
        lr = LogisticRegression(penalty=penality, C=C)
        lr.fit(train_x, train_y)

        predicted_classes = lr.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"Logistic Regression model (penality={penality}, C={C}):")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "logistic_reg_penality": penality,
            "logistic_reg_C": C
        }
        mlflow.log_params(params)
        # Logging model
        mlflow.sklearn.log_model(lr, "candidate_model")
        candidate_model_uri = mlflow.get_artifact_uri('candidate_model')

        # Logging artifacts the data
        mlflow.log_artifacts("data/")

        # Thresholds
        thresholds = {
            "accuracy_score": MetricThreshold(
                threshold=0.8,  # accuracy should be >=0.8
                min_absolute_change=0.05,  # accuracy should be at least 0.05 greater than baseline model accuracy
                min_relative_change=0.05,  # accuracy should be at least 5 percent greater than baseline model accuracy
                greater_is_better=True,
            ),
        }

        def custom_accuracy_score(eval_df, _builtin_metrics):
            return accuracy_score(eval_df['prediction'], eval_df['target'])

        def custom_precision_score(eval_df, _builtin_metrics):
            return precision_score(eval_df['prediction'], eval_df['target'], average='weighted')

        def custom_recall_score(eval_df, _builtin_metrics):
            return recall_score(eval_df['prediction'], eval_df['target'], average='weighted')

        def custom_f1_score(eval_df, _builtin_metrics):
            return f1_score(eval_df['prediction'], eval_df['target'], average='weighted')

        result = mlflow.evaluate(
            candidate_model_uri,
            test,
            targets="variety",
            model_type="classifier",
            custom_metrics=[
                make_metric(
                    eval_fn=custom_accuracy_score,
                    greater_is_better=True,
                    name='Accuracy Metric'
                ),
                make_metric(
                    eval_fn=custom_precision_score,
                    greater_is_better=True,
                    name='Precision Metric'
                ),
                make_metric(
                    eval_fn=custom_recall_score,
                    greater_is_better=True,
                    name='Recall Metric'
                ),
                make_metric(
                    eval_fn=custom_f1_score,
                    greater_is_better=True,
                    name='F1 Metric'
                ),
            ],
            validation_thresholds=thresholds,
            baseline_model=baseline_model_uri
        )
        run = mlflow.last_active_run()

        print("Active run_id: {}".format(run.info.run_id))