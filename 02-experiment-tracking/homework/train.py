import os
import pickle
import click
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")  
mlflow.set_experiment("random-forest-train")  

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    mlflow.sklearn.autolog()  
    
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    
    with mlflow.start_run():        
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(
            rf, 
            "model", 
            input_example=X_train[:5],  
            signature=mlflow.models.signature.infer_signature(X_train, y_train)  
        )

if __name__ == '__main__':
    run_train()
