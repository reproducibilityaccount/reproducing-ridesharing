import pickle
import matplotlib.path as mplPath
import pandas as pd
from get_polygon_coords_from_csv import get_polygons_and_other_fields

def generate_neighbourhood_idx_mapping(neigh_other, save_filename):
    with open(save_filename, "w") as f:
        for neigh_idx, neighbourhood in enumerate(neigh_other):
            f.write(f"{neigh_idx},{neigh_other[neigh_idx]['BoroName']},{neigh_other[neigh_idx]['BoroCode']},{neigh_other[neigh_idx]['CountyFIPS']},{neigh_other[neigh_idx]['NTACode']},{neigh_other[neigh_idx]['NTAName']}\n")

def map_locations_to_neighbourhoods(filename,filename_to_save):
    neighbourhood_data, other_fields_for_all_lines = get_polygons_and_other_fields("nynta.csv")
    generate_neighbourhood_idx_mapping(other_fields_for_all_lines, "neigh_to_idx_mapping.txt")
    paths_per_neigh = []
    for neigh_idx, neighbourhood in enumerate(neighbourhood_data):
        paths_for_curr_neigh = []
        for first_level in neighbourhood:
            for second_level in first_level:
                paths_for_curr_neigh.append(mplPath.Path(second_level))
        paths_per_neigh.append(paths_for_curr_neigh)
    with open(filename, 'r') as f:
        neighbourhood_mapping = []    
        lines = f.readlines()
        for line_idx, line in enumerate(lines):
            _, latitude, longitude = line.split(",")
            latitude = float(latitude)
            longitude = float(longitude)
            found = False
            for neigh_idx, neigh_paths in enumerate(paths_per_neigh):
                for curr_path in neigh_paths:
                        if curr_path.contains_point((latitude,longitude)):
                            neighbourhood_mapping.append(neigh_idx)
                            found = True
            if not found:
                raise ValueError(f"point {line_idx} was not found in any neighbourhood")
        print(neighbourhood_mapping)
        items = {}
        for nr in neighbourhood_mapping:
            if nr not in items:
                items[nr] = 0
            else:
                items[nr]+=1
        print(items)
        pickle.dump(neighbourhood_mapping, open(filename_to_save,"wb"))



if __name__ == "__main__":

    dirname = '../data/ny_preprocessed'
    map_locations_to_neighbourhoods(dirname + '/' + 'zone_latlong.csv', dirname + '/' + 'small_neighbourhood_labels.pkl')