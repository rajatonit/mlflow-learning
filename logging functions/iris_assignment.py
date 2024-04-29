import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import mlflow


#evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # np.random.seed()

    # Assuming 'df' is your DataFrame and 'label_column' is the column containing labels
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    data = pd.read_csv("iris.csv")


    # Fit and transform the label column
    data['variety'] = label_encoder.fit_transform(data['variety'])

    train,test = train_test_split(data)
    train.to_csv("./data/train.csv")
    test.to_csv("./data/test.csv")

    train_x = train.drop(["variety"], axis=1)
    test_x = test.drop(["variety"], axis=1)
    train_y = train[["variety"]]
    test_y = test[["variety"]]

    # mlflow.set_tracking_uri(uri="./mytracks")
    mlflow.set_tracking_uri(uri="")

    print("The set tracking url is ", mlflow.get_tracking_uri())
    exp = mlflow.set_experiment(experiment_name="iris_assignment")

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))


    mlflow.start_run()

    tags = {
        "project": "iris_test",
        "release.candidate": "RC1",
        "release.version": "1.0"
    }

    mlflow.set_tags(tags)



    lr = LinearRegression()
    lr.fit(train_x,train_y)
    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(lr, "iris_test_model")

    run = mlflow.active_run()

    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))
    mlflow.end_run()

    run = mlflow.last_active_run()

    print("Last run id is {}".format(run.info.run_id))
    print("Last run name is {}".format(run.info.run_name))



