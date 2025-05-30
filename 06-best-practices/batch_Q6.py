import os
import pandas as pd
from datetime import datetime
import batch

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}


def dt(hour, minute=0, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def save_data(df: pd.DataFrame, file_path: str):
    df.to_parquet(
        file_path,
        engine="pyarrow",
        compression=None,
        index=False,
        storage_options=options
    )


def test_integration():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),       
        (1, 1, dt(1, 2), dt(1, 10)),             
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),    
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),        
    ]
    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]
    df_input = pd.DataFrame(data, columns=columns)

    input_file = batch.get_input_path(2023, 1)
    save_data(df_input, input_file)

    os.system("python batch.py 2023 1")

    output_file = batch.get_output_path(2023, 1)
    df_result = pd.read_parquet(output_file, storage_options=options)
    total_duration = df_result["predicted_duration"].sum()

    print(f"Predicted duration total: {round(total_duration, 2)}")


if __name__ == "__main__":
    test_integration()
