#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd
import sklearn
import os

# Add argument parsing functionality
parser = argparse.ArgumentParser(description='Run predictions for a given year and month.')
parser.add_argument('--year', type=int, required=True, help='Year for prediction')
parser.add_argument('--month', type=int, required=True, help='Month for prediction')

# Parse the arguments
args = parser.parse_args()

year = args.year
month = args.month

# Input and output file paths
input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Load the pre-trained model
with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

# Categorical features
categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    # Calculate trip duration
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    # Filter the duration between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Handle missing values in categorical columns
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# Read the data
df = read_data(input_file)
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

# Prepare the data for prediction
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)

# Output the standard deviation of the predicted durations (for Q1)
print(f"Standard Deviation of Predicted Durations: {y_pred.std()}")

# Prepare the result DataFrame
df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

# Save the result to a parquet file
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

# Print the mean predicted duration
mean_predicted_duration = y_pred.mean()
print(f"Mean Predicted Duration: {mean_predicted_duration:.2f} minutes")
