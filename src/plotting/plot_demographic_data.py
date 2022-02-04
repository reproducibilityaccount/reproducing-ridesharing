import os
import sys

cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(cur_dir)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.plotting.recreate_figures import create_fig_1, get_lambda_str, add_details_to_plot, FILE_EXTENSIONS
from src.plotting.util import load_test_results, get_model_paths, load_regions, get_region_percentages


def load_data(data_dir=None, model_dir=None, sub_dirs=None):
    model_paths = get_model_paths(model_dir, sub_dirs=sub_dirs)
    test_results = load_test_results(model_paths)

    demographic_data_path = os.path.join(data_dir, 'nyc_demographics_by_nta_cleaned.csv')
    id_mapping_path = os.path.join(data_dir, 'used_neighbourhood_to_idx_mapping.txt')

    demographic_data = pd.read_csv(demographic_data_path)
    demographic_data = demographic_data.iloc[1:, [1, 4, 8, 9, 10, 11, 12]]
    mapping_data = pd.read_csv(id_mapping_path, header=None)

    return test_results, demographic_data, mapping_data


def plot_demographic_data(test_results, demographic_data, mapping_data, loc_region, region_labels, threshold=None):
    # define ethnic groups according to dataset
    ethnicities = ['Hispanic/Latino', 'White', 'African American', 'Asian', 'Other']

    # get mapping id -> nta codes
    nta_codes = mapping_data.sort_values(0)[5].values

    # get demographic data for selected nta codes
    selected_demographic_data = pd.concat(
        [demographic_data[demographic_data['NTA Code'] == nta_code] for nta_code in nta_codes])
    # convert str to int and float
    selected_demographic_data.iloc[:, 1] = [int(val.replace(',', '').replace('.', '')) for val in
                                            selected_demographic_data.iloc[:, 1].values]
    for i in range(2, 6):
        selected_demographic_data.iloc[:, i] = [float(val) / 100 for val in
                                                selected_demographic_data.iloc[:, i].values]
    selected_demographic_data.iloc[:, -1] = [float(val[:-1]) / 100 for val in
                                             selected_demographic_data.iloc[:, -1].values]

    # get number of residents per neighbourhood and ethnicity
    ethn_population = selected_demographic_data['Total population'].values.reshape(-1,1)
    ethn_population = ethn_population * selected_demographic_data.iloc[:, 2:].values

    for obj_func, model_results in test_results.items():
        for model_result in model_results:
            # plot only one configuration for driver and rider fairness
            if obj_func == "driver_fairness" and model_result['settings']['lambda'] != 0.5:
                continue
            elif obj_func == "rider_fairness" and model_result['settings']['lambda'] != 1e10:
                continue

            # calculate percentage of serviced requests per neighbourhood
            reg_percentages, loc_acceptances, loc_requests = get_region_percentages(model_result, loc_region,
                                                                                    region_labels, return_details=True)
            masked_reg_percentages = np.array(reg_percentages)

            if threshold is not None:
                # ignore neighbourhoods with less than threshold requests
                mask = np.array(list(loc_requests.values())) < threshold
            else:
                mask = np.zeros(len(masked_reg_percentages)).astype(bool)
            masked_reg_percentages[mask] = 0

            # calculate
            served_citizens = np.array(masked_reg_percentages).reshape(-1, 1) * ethn_population
            served_ratio = np.sum(served_citizens, axis=0) / np.sum(ethn_population[~mask], axis=0)

            # plot differences between service ratios and means per group
            lambda_text = get_lambda_str(obj_func, model_result["settings"]["lambda"])
            if lambda_text != "":
                lambda_text = ', ' + lambda_text
            plt.plot(ethnicities, served_ratio - np.mean(served_ratio), label=f'{obj_func}{lambda_text}')

    ylabel = 'Servicing Rate Difference'
    xlabel = 'Ethnicity'
    fontsizes = {'title': 24, 'xlabel': 16, 'ylabel': 16, 'xticks': 12, 'yticks': 12}
    add_details_to_plot(f"Average Success Rates", xlabel, ylabel, fontsizes)

    plt.legend()


def analyse_demographics(data_dir=None, model_dir=None, plot_dir=None):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    if data_dir is None:
        data_dir = os.path.join(cur_dir, '../../data/ny_new_neighbourhoods/')

    if model_dir is None:
        model_dir = os.path.join(cur_dir, '../../models/')

    if plot_dir is None:
        root_dir = os.path.dirname(os.path.dirname(cur_dir))
        plot_dir = os.path.join(root_dir, 'plots')
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    sub_dirs = ['demographic']
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    create_fig_1(models_dir=model_dir, data_dir=data_dir, use_limits=True, plot_name='demographic_fig_1',
                 plot_dir=plot_dir, num_agents=200, sub_dirs=sub_dirs, create_fig=False, spec_title='Success Rates')
    test_results, demographic_data, mapping_data = load_data(data_dir, model_dir, sub_dirs=sub_dirs)
    loc_region, region_labels, _ = load_regions(data_dir)
    plt.subplot(1, 2, 2)
    plot_demographic_data(test_results, demographic_data, mapping_data, loc_region, region_labels)

    plt.tight_layout()
    for file_ext in FILE_EXTENSIONS:
        plt.savefig(os.path.join(plot_dir, f'demographic.{file_ext}'))
    plt.show()
    plt.close()


if __name__ == '__main__':
    analyse_demographics()
