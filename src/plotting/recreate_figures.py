import os
import sys

cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(cur_dir)))


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from src.plotting.util import get_model_paths, load_test_results, load_regions, requests_completed, rider_min, \
    payment_by_driver, determine_gain_and_std_values

OBJ_FUNCS = ['income', 'rider_fairness', 'driver_fairness', 'requests']

FILE_EXTENSIONS = ['pdf', 'png']


def add_details_to_plot(title, xlabel, ylabel, fontsizes, xticks=None, xticks_labels=None, yticks=None,
                        yticks_labels=None, remove_x=False, remove_y=False, tighten_layout=True):
    plt.title(title, fontsize=fontsizes['title'])
    plt.xlabel(xlabel, fontsize=fontsizes['xlabel'])
    plt.ylabel(ylabel, fontsize=fontsizes['ylabel'])
    if xticks is None:
        plt.xticks(fontsize=fontsizes['xticks'])
    else:
        plt.xticks(xticks, xticks_labels, fontsize=fontsizes['xticks'])
    if yticks is None:
        plt.yticks(fontsize=fontsizes['yticks'])
    else:
        plt.yticks(yticks, yticks_labels, fontsize=fontsizes['yticks'])
    ax = plt.gca()
    ax.get_xaxis().set_visible(not remove_x)
    ax.get_yaxis().set_visible(not remove_y)
    if tighten_layout:
        plt.tight_layout()


def get_lambda_str(obj_func, lambda_):
    if obj_func in [OBJ_FUNCS[0], OBJ_FUNCS[-1]]:
        return ''
    if lambda_ >= 10:
        lambda_text = f'$\lambda$={lambda_: .0e}'
    else:
        lambda_text = f'$\lambda$={round(lambda_, 2)}'
    return lambda_text


def get_color_label_mark(obj_func):
    if obj_func == 'driver_fairness':
        label = 'Driver Fairness'
        color = 'g'
        mark = '+'
    elif obj_func == 'income':
        label = 'Income'
        color = 'y'
        mark = 'X'
    elif obj_func == 'requests':
        label = 'Requests'
        color = 'b'
        mark = 's'
    elif obj_func == 'rider_fairness':
        label = 'Rider Fairness'
        color = 'k'
        mark = 'v'
    else:
        label = obj_func
        mark = 'x'
        color = 'r'
    return color, label, mark


def plot_num_min_request(full_data, loc_region, region_labels, num_agents, verbose=True,
                         annotate=False):
    for obj_func, model_results in full_data.items():
        num_list = []
        min_list = []
        is_def_seed_list = []
        lambdas = []

        for model_result in model_results:
            if model_result['settings']['num_agents'] != num_agents:
                continue
            is_def_seed = 'seed' not in model_result['settings']
            if not is_def_seed:
                if verbose:
                    print(f'Ignoring {obj_func} model with non-default seed {model_result["settings"]["seed"]}')
                continue
            num_list.append(requests_completed(model_result))
            min_list.append(rider_min(model_result, loc_region, region_labels))
            is_def_seed_list.append(is_def_seed)
            lambdas.append(model_result['settings']['lambda'])

        if len(num_list) > 0:
            color, label, mark = get_color_label_mark(obj_func)

            plt.scatter(num_list, min_list, label=label, marker=mark, color=color, s=70)
            if annotate:
                if obj_func in OBJ_FUNCS[1:3]:
                    for x, y, lambda_ in zip(num_list, min_list, lambdas):
                        plt.annotate(get_lambda_str(obj_func, lambda_), [x, y])


def plot_num_min_requests_distributions(full_data, loc_region, region_labels, num_agents=200):
    # https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    N = 3000
    render_factor = 0.1
    threshold = 1000
    differentiating_setting = 'lambda'

    points = {}
    x_min, x_max = (0.024, 0.22)
    y_min, y_max = (0.019, 0.15)
    # get x and y value for different seeded runs
    for obj_func, model_results in full_data.items():
        points[obj_func] = {}
        for model_result in model_results:
            if model_result['settings']['num_agents'] != num_agents:
                continue
            re_completed = requests_completed(model_result)
            min_rider = rider_min(model_result, loc_region, region_labels)
            x_min = min(x_min, re_completed)
            x_max = max(x_max, re_completed)
            y_min = min(y_min, min_rider)
            y_max = max(y_max, min_rider)
            model_points = points[obj_func].get(model_result['settings'][differentiating_setting], [])
            model_points.append([re_completed, min_rider])
            points[obj_func][model_result['settings'][differentiating_setting]] = model_points

        remove_keys = []
        for diff_setting, coords in points[obj_func].items():
            # remove configurations with only one run
            if len(coords) == 1:
                remove_keys.append(diff_setting)
            else:
                # calculate mean and covariance
                mean = np.mean(coords, axis=0)
                cov = np.cov(coords, rowvar=False)
                points[obj_func][diff_setting] = [mean, cov]
        for remove_key in remove_keys:
            points[obj_func].pop(remove_key)

    X = np.linspace(x_min * (1 - render_factor), x_max * (1 + render_factor), N)
    Y = np.linspace(y_min * (1 - render_factor), y_max * (1 + render_factor), N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2 * np.pi) ** n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

        return np.exp(-fac / 2) / N

    handles = []
    for obj_func, el in points.items():
        color, label, _ = get_color_label_mark(obj_func)

        for key, distribution in el.items():
            # Mean vector and covariance matrix
            mu, Sigma = distribution

            # The distribution on the variables X, Y packed into pos.
            Z = multivariate_gaussian(pos, mu, Sigma)
            mask = Z < threshold
            Z[mask] = -np.inf
            x_values = np.argmax(~mask, axis=1)[np.any(~mask, axis=1)]
            y_values = np.argmax(~mask, axis=0)[np.any(~mask, axis=0)]
            x_min = min(x_min, X[0, min(x_values)])
            x_max = max(x_max, X[0, max(x_values)])
            y_min = min(y_min, Y[min(y_values), 0])
            y_max = max(y_max, Y[max(y_values), 0])

            # get colormap
            ncolors = 256
            color_array = plt.get_cmap()(range(ncolors))
            # change alpha values
            color_array[:, :3] = [colors.to_rgba(color)[:-1] for _ in range(ncolors)]
            color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
            # create a colormap object
            map_object = LinearSegmentedColormap.from_list(name='alpha', colors=color_array)

            plt.contourf(X, Y, Z, cmap=map_object)

        _, label, _ = get_color_label_mark(obj_func)
        red_patch = mpatches.Patch(color=color, label=label)
        handles.append(red_patch)
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.legend(handles=handles, loc='lower right')


def create_fig_1(models_dir=None, data_dir=None, use_limits=False, compare_seeds=False, verbose=True, plot_dir='',
                 plot_name='fig1', num_agents=None, sub_dirs=['50_agents', '200_agents'], create_fig=True, spec_title=None):
    model_paths = get_model_paths(models_dir, sub_dirs=sub_dirs)
    test_results = load_test_results(model_paths)

    loc_region, region_labels, _ = load_regions(data_dir)
    xlabel = "Overall Success Rate"
    ylabel = "Min. Request Success Rate"
    fontsizes = {'title': 24, 'xlabel': 16, 'ylabel': 16, 'xticks': 12, 'yticks': 12}
    if compare_seeds:
        if create_fig:
            plt.figure(figsize=(6, 4))
        plot_num_min_requests_distributions(test_results, loc_region, region_labels)
        title = f"Distribution of seeded runs"
        if spec_title:
            title = spec_title
        add_details_to_plot(title, xlabel, ylabel, fontsizes)
        if create_fig:
            for file_ext in FILE_EXTENSIONS:
                plt.savefig(os.path.join(plot_dir, f'{plot_name}_seeds.{file_ext}'))
            plt.show()
            plt.close()
    else:
        if num_agents is None:
            plotted_agents = [50, 200]
        elif isinstance(num_agents, list):
            plotted_agents = num_agents
        else:
            plotted_agents = [num_agents]
        if create_fig:
            plt.figure(figsize=(6*len(plotted_agents), 4))
        for idx, num_agents in enumerate(plotted_agents):
            if create_fig:
                plt.subplot(1, len(plotted_agents), idx + 1)
            plot_num_min_request(test_results, loc_region, region_labels, num_agents, verbose=verbose)
            title = f"{num_agents} Drivers"
            if spec_title is not None:
                title = spec_title
            add_details_to_plot(title, xlabel, ylabel, fontsizes)
            plt.legend(loc='upper left')
            if use_limits:
                if num_agents == 50:
                    plt.xlim(0.018, 0.052)
                    plt.ylim(0.008, 0.047)
                elif num_agents == 200:
                    plt.xlim(0.024)  # , 0.22)
                    plt.ylim(0.019, 0.15)
        if create_fig:
            for file_ext in FILE_EXTENSIONS:
                plt.savefig(os.path.join(plot_dir, f'{plot_name}.{file_ext}'))
            plt.show()
            plt.close()


def plot_income_distro(incomes, names, use_limits=False):
    title = "Income by Objective Function"
    fontsizes = {'title': 20, 'xlabel': 18, 'ylabel': 18, 'xticks': 14, 'yticks': 18}
    plt.boxplot(incomes, showfliers=False, vert=False)
    add_details_to_plot(title, 'Payment ($)', 'Policy', fontsizes,
                        yticks=list(range(1, len(names) + 1)), yticks_labels=names)
    if use_limits:
        plt.xlim(900, 4900)


def create_fig_2(models_dir=None, num_agents=200, use_limits=False, compare_seeds=False, verbose=True,
                 plot_dir='', plot_name='fig2', sub_dirs=['200_agents']):
    model_paths = get_model_paths(models_dir, sub_dirs=sub_dirs)
    test_results = load_test_results(model_paths)
    incomes_per_obj_func = {}
    for obj_func in OBJ_FUNCS:

        if obj_func not in test_results:
            continue

        for test_run in test_results[obj_func]:
            if test_run['settings']['num_agents'] != num_agents:
                continue
            if not compare_seeds and 'seed' in test_run['settings']:
                if verbose:
                    print(f'Ignoring {obj_func} model with non-default seed {test_run["settings"]["seed"]}')
                continue

            agent_profits = test_run['epoch_each_agent_profit']

            driver_profits = {}
            for timestep in agent_profits:
                for agent_idx, agent_profit in timestep:
                    driver_profits[agent_idx] = driver_profits.get(agent_idx, 0) + agent_profit

            if obj_func == 'driver_fairness':
                if test_run["settings"]["lambda"] != 4 / 6:
                    if verbose:
                        print(f'driver fairness with lambda {round(test_run["settings"]["lambda"], 2)} ignored')
                    continue
            elif obj_func == 'rider_fairness':
                if test_run["settings"]["lambda"] != 1e9:
                    if verbose:
                        print(f'rider fairness with lambda {round(test_run["settings"]["lambda"], 2)} ignored')
                    continue

            if compare_seeds:
                if obj_func not in incomes_per_obj_func:
                    incomes_per_obj_func[obj_func] = []

                incomes_per_obj_func[obj_func].append(list(driver_profits.values()))
                if verbose and len(list(driver_profits.values())) != num_agents:
                    print(f'Some drivers without profits in {obj_func}')
            else:
                if obj_func in incomes_per_obj_func:
                    if verbose:
                        print(f'obj_func {obj_func} already present')
                else:
                    _, label, _ = get_color_label_mark(obj_func)
                    incomes_per_obj_func[label] = list(driver_profits.values())
                if verbose and len(list(driver_profits.values())) != num_agents:
                    print(f'Some drivers without profits in {obj_func}')

    if compare_seeds:
        for obj_func, results in incomes_per_obj_func.items():
            plt.figure(figsize=(8, 3))
            _, label, _ = get_color_label_mark(obj_func)
            plot_income_distro(results, [label] * len(results), use_limits)
            for file_ext in FILE_EXTENSIONS:
                plt.savefig(os.path.join(plot_dir, f'{plot_name}_{obj_func}.{file_ext}'))
            plt.show()
            plt.close()
    else:
        plt.figure(figsize=(8, 3))
        plot_income_distro(incomes_per_obj_func.values(), incomes_per_obj_func.keys(), use_limits)
        for file_ext in FILE_EXTENSIONS:
            plt.savefig(os.path.join(plot_dir, f'{plot_name}.{file_ext}'))
        plt.show()
        plt.close()


def create_fig_3(models_dir=None, num_agents=200, obj_func='requests', verbose=True, plot_dir='',
                 plot_name='fig3', sub_dirs=['200_agents']):
    model_paths = get_model_paths(models_dir, sub_dirs=sub_dirs)
    test_results = load_test_results(model_paths)
    if obj_func in test_results.keys():
        test_runs = test_results[obj_func]
        test_run = [test_run for test_run in test_runs if test_run['settings']['num_agents'] == num_agents]
    else:
        print(f'No model found for obj_func={obj_func}')
        return

    if len(test_run) == 0:
        if verbose:
            print(f'No model found for num_agents={num_agents}, obj_func={obj_func}')
        return
    elif len(test_run) > 1:
        if verbose:
            print(f'Found {len(test_run)} models, using first one')
    test_run = test_run[0]

    r_vals = np.linspace(0, 1, 101)
    gain, std_vals = determine_gain_and_std_values(test_run, r_vals)
    fontsizes = {'title': 24, 'xlabel': 16, 'ylabel': 16, 'xticks': 14, 'yticks': 14}

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(r_vals, gain)
    add_details_to_plot('Gain vs. r', 'r', 'Gain', fontsizes)
    plt.subplot(1, 2, 2)
    plt.plot(r_vals, std_vals)
    add_details_to_plot('Standard Deviation vs. r', 'r', 'Standard Deviation', fontsizes)
    for file_ext in FILE_EXTENSIONS:
        plt.savefig(os.path.join(plot_dir, f'{plot_name}.{file_ext}'))
    plt.show()
    plt.close()


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    plot_dir = os.path.join(root_dir, 'plots')
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    create_fig_1(use_limits=True, plot_dir=plot_dir)
    create_fig_1(use_limits=True, compare_seeds=True, plot_dir=plot_dir)
    create_fig_2(use_limits=True, plot_dir=plot_dir)
    create_fig_2(use_limits=True, compare_seeds=True, plot_dir=plot_dir)
    create_fig_3(plot_dir=plot_dir)


if __name__ == '__main__':
    main()
