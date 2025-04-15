import os
import pandas as pd
import s3fs
from datetime import datetime
from batch import get_input_path, get_output_path, read_data

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def save_data(df: pd.DataFrame, path: str):
    df.to_parquet(
        path,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

def test_integration():
    # 1. Create fake input data
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

    # 2. Save input to S3
    input_file = get_input_path(2023, 1)
    save_data(df_input, input_file)

    # 3. Run batch.py script (this will read from input and write to output)
    exit_code = os.system("python batch.py 2023 1")
    assert exit_code == 0

    # 4. Read output from S3 and verify
    output_file = get_output_path(2023, 1)
    df_result = pd.read_parquet(output_file, storage_options=options)

    total_duration = df_result["predicted_duration"].sum().round(2)
    print(f"\nPredicted duration total: {total_duration}")
    assert total_duration == 36.28  # choose the closest option
