#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

year = 2021
month = 4
taxi_type = 'fhv'

#df = read_data('fhv_tripdata_2021-03.parquet')
df = read_data(f'{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)

print('y_pred mean is:',y_pred.mean())

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df['predictions'] = y_pred

df_result = df[['ride_id', 'predictions']].copy()


output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'


output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

