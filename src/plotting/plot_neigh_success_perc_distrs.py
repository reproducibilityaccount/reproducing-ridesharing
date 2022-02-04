import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(cur_dir)))

from src.plotting.recreate_figures import add_details_to_plot, FILE_EXTENSIONS
from src.plotting.util import load_test_results, get_model_paths, load_regions, get_region_percentages


def load_data(model_dir=None, sub_dirs=None):
    model_paths = get_model_paths(model_dir, sub_dirs=sub_dirs)
    test_results = load_test_results(model_paths)

    return test_results


def violin_plots_per_obj_func(sub_dirs, plot_dir = None, data_dir = None, demographic_data_dir = None, model_dir = None, compare_seeds = False, plot_all = False, plot_locations = False, plot_all_lambdas = False):
    
    if compare_seeds == False:
        print("Not taking into account the same runs with different seeds")
    else:
        print("Taking into account the same runs with different seeds")
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    if data_dir is None:
        data_dir = os.path.join(cur_dir, '../../data/ny/')
    if demographic_data_dir is None:
        demographic_data_dir = os.path.join(cur_dir, '../../data/ny_new_neighbourhoods/')
    if model_dir is None:
        model_dir = os.path.join(cur_dir, '../../models/')

    if plot_dir is None:
        root_dir = os.path.dirname(os.path.dirname(cur_dir))
        plot_dir = os.path.join(root_dir, 'plots')

    loc_region, region_labels, _ = load_regions(data_dir)
    demographic_loc_region, demographic_region_labels, _ = load_regions(demographic_data_dir)
    for sub_dir in sub_dirs:
        test_results = load_data(model_dir, sub_dirs=[sub_dir])
        reg_percs_all = []
        nr_models = 0
        for obj_func, model_results in test_results.items():
            for model_result in model_results:
                settings_obj = model_result['settings']
                lambda_val = settings_obj['lambda']
                if compare_seeds == False and 'seed' in settings_obj:
                    continue
                if not plot_all_lambdas:
                    if settings_obj['fairness_obj'] == 'rider_fairness' and lambda_val != 10**10:
                        continue
                    if settings_obj['fairness_obj'] == 'driver_fairness' and lambda_val != 0.5:
                        continue
                if sub_dir == 'demographic':
                    reg_percentages, loc_acceptances, loc_requests = get_region_percentages(model_result, demographic_loc_region, demographic_region_labels, return_details=True)
                else:
                    reg_percentages, loc_acceptances, loc_requests = get_region_percentages(model_result, loc_region, region_labels, return_details=True)
                print("nr neighbourhoods:",len(reg_percentages))
                nr_models += 1
                for reg_perc, loc_request in zip(reg_percentages,loc_requests.items()):
                    neigh_idx, total_requests = loc_request
                    if total_requests < 1000:
                        curr_population = "between 0 and 1000"
                    elif total_requests < 3000:
                        curr_population = " between 1000 and 3000"
                    elif total_requests < 10000:
                        curr_population = "between 3000 and 10000"
                    elif total_requests < 50000:
                        curr_population = "between 10000 and 50000"
                    else:
                        curr_population = "greater than 50000"

                    if plot_all:
                        reg_percs_all.append([reg_perc,obj_func+',$\lambda$='+str(round(float(model_result['settings']['lambda']),4)), curr_population])
                    else:
                        reg_percs_all.append([reg_perc,obj_func, curr_population])
        print("nr models: ",nr_models)
        ref_df = pd.DataFrame(reg_percs_all,columns = ['success rate','objective function', 'nr of requests'])
        ref_df = ref_df.sort_values(by=['objective function'])
        x_axis = "success rate" if plot_all else "objective function"
        y_axis = "objective function" if plot_all else "success rate"
        plt.figure(figsize=(6,4))
        violin = sns.violinplot(x=x_axis,y=y_axis, data=ref_df, color="0.8")
        if plot_locations:
            plt.setp(violin.collections, alpha = .2)
            sns.stripplot(x=x_axis,y=y_axis, hue = 'nr of requests', data=ref_df, jitter=True)
        else:
            sns.stripplot(x=x_axis,y=y_axis, data=ref_df, jitter=True)
        xlabel = 'Objective function'
        ylabel = 'Success rate'
        fontsizes = {'title': 24, 'xlabel': 16, 'ylabel': 16, 'xticks': 12, 'yticks': 12}
        add_details_to_plot(f"Neighbourhood success rates", xlabel, ylabel, fontsizes, tighten_layout=True)

        for file_ext in FILE_EXTENSIONS:
            plt.savefig(os.path.join(plot_dir, f'violin_{sub_dir}.{file_ext}'))
        plt.show()


if __name__ == '__main__':
    # pass compare_seeds = True if seeded runs are wanted as well
    # pass plot_all = True if all hyperparameters for each objective function
    # are wanted to be plotted as violin plots as well instead of just gathered as part of the corresponding objective function
    violin_plots_per_obj_func(['demographic'])