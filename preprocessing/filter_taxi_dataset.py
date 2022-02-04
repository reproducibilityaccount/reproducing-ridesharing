import osmnx as ox
import networkx as nx
import pandas as pd
import os
import numpy as np
from tqdm.auto import tqdm
import matplotlib.path as mplPath

def get_bounding_box_and_polypath(graph_name):
    G = ox.load_graphml(graph_name)
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_node_ID')

    nodes = ox.graph_to_gdfs(G, edges=False)
    edges = ox.graph_to_gdfs(G, nodes=False)

    coords = np.zeros((len(nodes), 2))

    for idx, (x, y) in tqdm(enumerate(zip(nodes.x, nodes.y)), total=len(nodes)):
        coords[idx][0], coords[idx][1] = x, y

    return mplPath.Path(coords), nodes.geometry.total_bounds

def filter_taxi_data(from_dirname, to_dirname, filename, poly_path, x_min, y_min, x_max, y_max):
    data = pd.read_csv(from_dirname + '/' + filename + '.csv')

    print(filename + ' before filtering: ', len(data))

    data = data[(data['pickup_longitude'] >= x_min) & (data['pickup_longitude'] <= x_max) &
                (data['dropoff_longitude'] >= x_min) & (data['dropoff_longitude'] <= x_max) &
                (data['pickup_latitude'] >= y_min) & (data['pickup_latitude'] <= y_max) &
                (data['dropoff_latitude'] >= y_min) & (data['dropoff_latitude'] <= y_max)].reset_index()

    indices_to_drop = []

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        pickup = row['pickup_longitude'], row['pickup_latitude']
        dropoff = row['dropoff_longitude'], row['dropoff_latitude']
        if not (poly_path.contains_point(pickup) and poly_path.contains_point(dropoff)):
            indices_to_drop.append(idx)
        
    data.drop(data.index[indices_to_drop], inplace=True)

    print(filename + ' after filtering: ', len(data))

    data.to_csv(to_dirname + '/' + filename + '_manhattan.csv')

def filter_taxi_dataset():
    graphname = 'Manhattan_Island'
    dirname = 'yellow_taxi_data'
    poly_path, (x_min, y_min, x_max, y_max) = get_bounding_box_and_polypath(graphname + '.graphml')
    for file in os.listdir(dirname):
        if file.endswith('.csv'):
            filter_taxi_data(dirname, dirname + '_' + graphname.split('_')[0].lower(), file[ : -4], poly_path, x_min, y_min, x_max, y_max)

if __name__ == "__main__":
    filter_taxi_dataset()