import os
from generate_graph_information import generate_graph_information
from filter_taxi_dataset import filter_taxi_dataset
from batch_minutes import generate_request_batches_per_minute
from sklearn.cluster import KMeans
import pickle

def write_kmeans(dirname="../data/ny_preprocessed"):

    zone_lat_long = open(dirname + "/zone_latlong.csv").read().split("\n")
    d = {}
    coords = []
    for i in zone_lat_long:
        if i!='':
            a,b,c = i.split(",")
            d[a] = (float(b),float(c))
            coords.append((float(b),float(c)))

    regions = KMeans(n_clusters=10).fit(coords)
    labels = regions.labels_
    centers = regions.cluster_centers_

    pickle.dump(labels,open(dirname + "/new_labels.pkl","wb"))

if __name__ == "__main__":

	data_folder = '../data/ny_preprocessed'

	if not os.path.exists(data_folder):
		os.makedirs(data_folder)

	#generate_graph_information()
	#filter_taxi_dataset()
	#generate_request_batches_per_minute()
	write_kmeans(data_folder)
