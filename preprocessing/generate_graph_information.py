import osmnx as ox
import networkx as nx
import pandas as pd
from os import path
import numpy as np
from tqdm.auto import tqdm
from shapely.geometry import Point
import matplotlib.path as mplPath
from random import randint

def generate_graph(cityname):
    G = ox.graph_from_place(cityname, network_type = 'drive')
    # add the edge speeds and calculate the travel time between nodes
    G = ox.speed.add_edge_speeds(G, precision=1)
    G = ox.speed.add_edge_travel_times(G, precision=1)
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_node_ID')
    return G


# code for this function accessed from: https://towardsdatascience.com/finding-time-dependent-travel-times-between-every-pair-of-locations-in-manhattan-c3c48b0db7ba
def get_UM_travel_info(G):
    # downloaded from Uber Movement
    speed_df = pd.read_csv('nyc_avg_speeds_2019-06.csv')
    speed_df = speed_df[['osm_way_id', 'hour', 'speed']]
    speed_dict = dict([((t.osm_way_id, t.hour), t.speed) for t in speed_df.itertuples()])

    for edge in G.edges:
        edge_obj = G[edge[0]][edge[1]][edge[2]]
        wayid = edge_obj['osmid']
        try:
            speed = speed_dict[wayid] * 1.60934 # Convert from mph to kph
            distance = edge_obj['length'] / 1000 # Convert from m to km
            travel_time = distance / speed * 60 # Convert from hours to minutes
        except:
            travel_time = edge_obj['travel_time'] / 60 # Convert from seconds to minutes
        G[edge[0]][edge[1]][edge[2]]['um_travel_time'] = travel_time
    return G

def get_zone_latlong_info(nodes):
    x, y = nodes.x, nodes.y
    df_latlong = pd.DataFrame([x, y]).T.reset_index()
    return df_latlong.drop(labels='osmid', axis=1)

def get_zone_path_info(G, num_nodes):
    zone_path = np.zeros((num_nodes, num_nodes))
    paths_info = nx.all_pairs_shortest_path(G)

    for key, value in tqdm(paths_info, total=num_nodes):
        for src, dest_list in value.items():
            if len(dest_list) > 1:
                zone_path[key][src] = dest_list[1]
            else:
                zone_path[key][src] = dest_list[0]

    return pd.DataFrame(zone_path)

def get_zone_traveltime_info(G, num_nodes):
    shortest_time = np.zeros((num_nodes, num_nodes))
    path_generator = nx.shortest_path_length(G)
    for origin_data in tqdm(path_generator, total=num_nodes):
        origin = origin_data[0]
        dist_dict = origin_data[1]
        for destination in dist_dict:
            shortest_time[origin, destination] = dist_dict[destination]

    return pd.DataFrame(shortest_time)

def get_ignorezones(G, nodes):
    # gets the largest connected component
    # this ensures that there is a way to get from every point A to every point B in the network
    G = ox.utils_graph.get_largest_component(G, strongly=True)
    nodes_sc = ox.graph_to_gdfs(G, edges=False)
    return pd.DataFrame(np.setdiff1d(nodes.index.values, nodes_sc.index.values))

def get_initial_locs(num_agents, num_locations):
    f = open('../data/ny_preprocessed/taxi_3000_final.txt', 'w')

    for _ in range(num_agents):
        initial_state = randint(0, num_locations - 1)
        f.write(str(initial_state) + '\n')

def generate_graph_information():
    cityname = 'Manhattan Island'
    filename = cityname.replace(' ', '_')
    use_UM_speed_info = False
    num_agents = 200

    if path.exists(filename + '.graphml'):
        G = ox.load_graphml(filename + '.graphml')
    else:
        G = generate_graph(cityname)
        ox.save_graphml(G, filename.repalce(' ', '_') + '.graphml')

    if use_UM_speed_info:
        G = get_UM_travel_info(G)

    num_nodes = len(G.nodes)
    nodes = ox.graph_to_gdfs(G, edges=False)
    edges = ox.graph_to_gdfs(G, nodes=False)
    print(f'Number of nodes: {nodes.shape[0]}')
    print(f'Number of edges: {edges.shape[0]}')

    df_latlong = get_zone_latlong_info(nodes)
    df_latlong.to_csv('../data/ny_preprocessed/zone_latlong.csv', header=False)

    df_zone_path = get_zone_path_info(G, num_nodes)
    df_zone_path.to_csv('../data/ny_preprocessed/zone_path.csv', index=False, header=False)

    df_travel_times = get_zone_traveltime_info(G, num_nodes)
    df_travel_times.to_csv('../data/ny_preprocessed/zone_traveltime.csv', index=False, header=False)

    df_ignorezones = get_ignorezones(G, nodes)
    df_ignorezones.to_csv('../data/ny_preprocessed/ignorezonelist.txt', index=False, header=False)

    get_initial_locs(num_agents, num_nodes)

if __name__ == "__main__":
    generate_graph_information()