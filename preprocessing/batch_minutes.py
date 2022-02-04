from time import time
import pandas as pd
import os
import osmnx as ox
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial
from datetime import timedelta
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

def coords_to_nodes_test(G, table, trip_type):
    longs, lats = table[trip_type + '_longitude'].values, table[trip_type + '_latitude'].values
    nodes = ox.distance.nearest_nodes(G, longs, lats)
    return nodes

def batch_minutes(filename, G, save_folder):

    table = pd.read_csv(filename)
    table.drop(labels=table.columns.difference(['tpep_pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']), axis=1, inplace=True)
    table = table.assign(pickup=pd.Series(np.zeros(len(table))).values, 
                         dropoff=pd.Series(np.zeros(len(table))).values,
                         day=pd.Series(np.zeros(len(table))).values,
                         val=pd.Series(np.ones(len(table))).values)
    table['tpep_pickup_datetime'] = table['tpep_pickup_datetime'].map(lambda x: x[ : -3])
    table['tpep_pickup_datetime'] = pd.to_datetime(table['tpep_pickup_datetime'], format='%Y%m%d %H:%M')
    start_date_march = '2016-03-23  00:00'
    end_date_march = '2016-04-01  00:00'
    start_date_april = '2016-04-01  00:00'
    end_date_april = '2016-04-09  23:59'
    table = table[(table['tpep_pickup_datetime'] > start_date_march) & (table['tpep_pickup_datetime'] <= end_date_march) | (table['tpep_pickup_datetime'] > start_date_april) & (table['tpep_pickup_datetime'] <= end_date_april)]
    table.sort_values('tpep_pickup_datetime', inplace=True)
    table.reset_index(inplace=True)
    table['pickup'] = coords_to_nodes_test(G, table, 'pickup')
    table['dropoff'] = coords_to_nodes_test(G, table, 'dropoff')
    table['day'] = pd.to_datetime(table['tpep_pickup_datetime']).dt.day
    table['tpep_pickup_datetime'] = table['tpep_pickup_datetime'].dt.strftime('%Y_%m_%d %H:%M')
    unique_days = table['day'].unique()
    for day in tqdm(unique_days):
        with open(f"{save_folder}/{day}.txt", "w") as f:
            f.write("1440\n")
            unique_minutes = table[table['day'] == day]['tpep_pickup_datetime'].unique()
            for minute_info in unique_minutes:
                hour, minute = minute_info[-5 : ].split(':')
                minute_nr = int(hour) * 60 + int(minute)
                f.write(f"Flows:{minute_nr}-{minute_nr}\n")
                trip_info = table[table['tpep_pickup_datetime'] == minute_info][['pickup', 'dropoff', 'val']]
                trip_info.to_csv(f, header=None, index=None, sep=',')

def rename_files_for_experiment(save_folder):
    os.chdir(save_folder)
     
    for count, f in enumerate(os.listdir()):
        f_name, f_ext = os.path.splitext(f)
        f_name = 'test_flow_5000_' + str(count + 1)
     
        new_name = f'{f_name}{f_ext}'
        os.rename(f, new_name)

def generate_request_batches_per_minute():
    parallel_processing = False
    graphname = 'Manhattan_Island'
    dirname = 'yellow_taxi_data_manhattan'
    save_folder = '../data/ny_preprocessed/files_60sec'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    G = ox.load_graphml(graphname + '.graphml')
    files = [dirname + '/' + item for item in os.listdir(dirname) if item.endswith('.csv')]
    
    if parallel_processing:
        with Pool() as pool:
            batch_minutes_with_args = partial(batch_minutes, G=G, save_folder=save_folder)
            with tqdm(total=len(files)) as pbar:
                for i, _ in enumerate(pool.imap_unordered(batch_minutes_with_args, files)):
                    pbar.update()
    else:
        for file in files:
            batch_minutes(file, G, save_folder)

    rename_files_for_experiment(save_folder)

if __name__ == "__main__":
    generate_request_batches_per_minute()
