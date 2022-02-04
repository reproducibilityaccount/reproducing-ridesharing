import json
import os
import pickle
import numpy as np


# Rider Utility #2
def requests_completed(data):
    return np.sum(data['epoch_requests_completed'])/np.sum(data['epoch_requests_seen'])


# Rider fairness
def rider_min(data,loc_region, region_labels, clusters=10):
    success = get_region_percentages(data, loc_region, region_labels)

    return np.min(success)


def get_region_percentages(data, loc_region, region_labels, return_details=False):
    loc_requests = {}
    loc_acceptances = {}
    for i in set(region_labels):
        loc_requests[i] = 0
        loc_acceptances[i] = 0

    for i in data['epoch_locations_all']:
        for j in i:
            loc_requests[loc_region[j]]+=1

    for i in data['epoch_locations_accepted']:
        for j in i:
            loc_acceptances[loc_region[j]]+=1

    success = []
    for i in loc_requests:
        success.append(loc_acceptances[i]/loc_requests[i])
    if return_details:
        return success, loc_acceptances, loc_requests
    else:
        return success


def payment_by_driver(data,n=-1):
    if n == -1:
        n = len(data['epoch_each_agent_profit'])

    num_drivers = data['settings']['num_agents']
    driver_pays = {}
    for i in range(num_drivers):
        driver_pays[i] = 0
    for i in data['epoch_each_agent_profit'][:n]:
        for j,k in i:
            driver_pays[j]+=k

    return driver_pays


def load_regions(data_dir=None):
    """Loading the KMeans regions"""
    if data_dir is None:
        data_dir = "../../data/ny/"
    zone_lat_long = open(data_dir+"zone_latlong.csv").read().split("\n")
    d = {}
    for i in zone_lat_long:
        if i != '':
            a, b, c = i.split(",")
            d[a] = (float(b), float(c))

    coords = [d[i] for i in d]
    region_labels = pickle.loads(open(data_dir+"new_labels.pkl", "rb").read())

    loc_region = {}

    for i in range(len(region_labels)):
        loc_region[i] = region_labels[i]

    return loc_region, region_labels, coords


def get_data(filename):
    with open(filename, "r") as f: # add other json files for different objective function models
        data = json.load(f)
    obj_func = data['settings']['fairness_obj']
    return data, obj_func


def get_model_paths(models_dir=None, sub_dirs=None):
    if models_dir is None:
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        models_dir = os.path.join(cur_dir, '../../models')

    if sub_dirs is None:
        sub_dirs = [os.path.join(models_dir, sub_dir) for sub_dir in os.listdir(models_dir)]
    else:
        sub_dirs = [os.path.join(models_dir, sub_dir) for sub_dir in sub_dirs]
    sub_dirs = [sub_dir for sub_dir in sub_dirs if os.path.isdir(sub_dir)]
    model_paths = [os.path.join(sub_dir, model_name) for sub_dir in sub_dirs for model_name in os.listdir(sub_dir)]
    model_paths = [model_path for model_path in model_paths if os.path.isdir(model_path)]

    return model_paths


def load_test_results(model_paths):
    test_results = {}
    for model_path in model_paths:
        try:
            data, obj_func = get_data(filename=os.path.join(model_path, 'test_data.json'))
        except FileNotFoundError:
            continue

        test_results[obj_func] = test_results.get(obj_func, []) + [data]

    return test_results


# implements formula 12 in paper
def determine_redistribution_values(driver_payments, shapley_values, r_vals, num_agents):
    q_vals = np.zeros((len(r_vals), num_agents))
    v_vals = np.array(shapley_values)
    pi_vals = driver_payments

    redist_consts = np.sum(np.einsum('i,j->ij', (1 - r_vals), pi_vals), axis=1)
    numerators = np.maximum(0, v_vals - np.einsum('i,j->ij', r_vals, pi_vals))
    norm_consts = np.sum(numerators, axis=1)
    q_vals = np.einsum('i,j->ij', r_vals, pi_vals) + np.einsum('ij,i->ij', numerators, redist_consts / norm_consts)

    return q_vals


def get_driver_payments(data, num_agents):
    driver_payments = dict.fromkeys(range(num_agents), 0)

    for minute_data in data:
        for agent_idx, payment in minute_data:
            driver_payments[agent_idx] += payment
    driver_payments = np.array(list(driver_payments.values()))

    return driver_payments


def determine_gain_and_std_values(data, r_vals):
    shapley_values = data['truncated_shapley']
    num_agents = data['settings']['num_agents']
    driver_payments = get_driver_payments(data['epoch_each_agent_profit'], num_agents)
    driver_payments = np.sort(driver_payments)

    q_vals = determine_redistribution_values(driver_payments, shapley_values, r_vals, num_agents)
    gain = np.zeros((len(r_vals), num_agents))
    
    for idx in range(num_agents):
        q_vals_before = q_vals[:, idx]
        driver_payments[idx] *= 2
        q_vals_after = determine_redistribution_values(driver_payments, shapley_values, r_vals, num_agents)[:, idx]
        driver_payments[idx] /= 2
        gain[ : , idx] = q_vals_after / q_vals_before
    
    gain = np.mean(gain, axis=1) - 1.0

    std_vals = np.std(q_vals / shapley_values, axis=1)
    std_vals = std_vals / np.max(std_vals)

    return gain, std_vals