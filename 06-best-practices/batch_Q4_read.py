import os
import pandas as pd

def read_data(path):
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")
    if s3_endpoint_url:
        options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }
        return pd.read_parquet(path, storage_options=options)
    else:
        return pd.read_parquet(path)

df = pd.read_parquet(
    "s3://nyc-duration/out/2023-03.parquet",
    storage_options={"client_kwargs": {"endpoint_url": "http://localhost:4566"}}
)

print(df.head())
