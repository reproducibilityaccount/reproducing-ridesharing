import numpy as np
import torch

from src import Util
from src.utils.Settings import Settings
from src.value_functions.PathBasedNN import PathBasedNN
from src.value_functions.GreedyValueFunction import GreedyValueFunction


def driver_0_score(envt, action, agent, driver_num):
    if driver_num == 0:
        score = sum([request.value for request in action.requests])
    else:
        score = 0

    return score


def closest_driver_score(envt, action, agent, driver_num):
    score = sum([request.value for request in action.requests])
    position = agent.position.next_location
    all_positions = [request.pickup for request in action.requests]
    all_positions += [request.dropoff for request in action.requests]

    if len(all_positions) == 0:
        return 0

    max_distance = max([envt.get_travel_time(position, j) for j in all_positions])

    if max_distance != 0:
        score /= max_distance
    else:
        score = 10000000

    return score


def furthest_driver_score(envt, action, agent, driver_num):
    score = sum([request.value for request in action.requests])
    position = agent.position.next_location
    all_positions = [request.pickup for request in action.requests]
    all_positions += [request.dropoff for request in action.requests]

    max_distance = max([envt.get_travel_time(position, j) for j in all_positions], default=0)
    score *= max_distance
    return score


def two_sided_score(envt, action, agent, driver_num):
    position = agent.position.next_location
    settings = Settings.get_settings()
    lamb = settings.get_value("lambda")

    time_driven = 0
    for request in action.requests:
        time_driven += envt.get_travel_time(request.pickup, request.dropoff)

    times_to_request = sum([envt.get_travel_time(position, request.pickup) for request in action.requests])

    previous_driver_utility = envt.driver_utilities[driver_num]
    if time_driven == 0 or times_to_request > 300 * len(action.requests):
        score = 0
    else:
        raise NotImplementedError('Specify max_driver_utility')
        driver_inequality = abs(max_driver_utility - (previous_driver_utility + time_driven - times_to_request))
        passenger_inequality = times_to_request
        score = 1000 / (lamb * driver_inequality(1 - lamb) * passenger_inequality)

    return score


def lambda_entropy_score(envt, action, agent, driver_num):
    profit = Util.change_profit(envt, action)
    entropy = Util.change_entropy(envt, action, driver_num)
    settings = Settings.get_settings()
    lamb = settings.get_value("lambda")

    if np.isfinite(entropy):
        score = profit - lamb * entropy
    else:
        score = profit

    return score


def lambda_variance_score(envt, action, agent, driver_num):
    profit = Util.change_profit(envt, action)
    variance = Util.change_variance(envt, action, driver_num)
    settings = Settings.get_settings()
    lamb = settings.get_value("lambda")

    return profit - lamb * variance


def lambda_entropy_rider_score(envt, action, agent, driver_num):
    profit = Util.change_profit(envt, action)
    entropy = Util.change_entropy_rider(envt, action, driver_num)
    settings = Settings.get_settings()
    lamb = settings.get_value("lambda")

    if np.isfinite(entropy):
        score = profit - lamb * entropy
    else:
        score = profit

    return score


def lambda_variance_rider_score(envt, action, agent, driver_num):
    profit = Util.change_profit(envt, action)
    variance = Util.change_variance_rider(envt, action, driver_num)
    settings = Settings.get_settings()
    lamb = settings.get_value("lambda")

    return profit - lamb * variance


def immediate_reward_score(envt, action, agent, driver_num):
    immediate_reward = sum([request.value for request in action.requests])
    delay_coefficient = 0
    settings = Settings.get_settings()

    if settings.has_value("delay_coefficient"):
        delay_coefficient = settings.get_value("delay_coefficient")

    remaining_delay_bonus = delay_coefficient * action.new_path.total_delay
    score = immediate_reward + remaining_delay_bonus
    return score


def num_to_value_function(envt, num, model_dir=None, embedding_dir=None, check_for_gpu=False):

    if model_dir is None:
        model_dir = '../models/'
    if embedding_dir is None:
        embedding_dir = model_dir
    if check_for_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    else:
        device = torch.device("cpu")
    settings = Settings.get_settings()
    model_loc = ""
    if settings.has_value("model_loc"):
        model_loc = settings.get_value("model_loc")
    if num == 1:
        value_function = PathBasedNN(device, envt, model_dir=model_dir, embedding_dir=embedding_dir, load_model_loc=model_loc)
    elif num == 2:
        value_function = GreedyValueFunction(envt, immediate_reward_score)
    elif num == 3:
        value_function = GreedyValueFunction(envt, driver_0_score)
    elif num == 4:
        value_function = GreedyValueFunction(envt, closest_driver_score)
    elif num == 5:
        value_function = GreedyValueFunction(envt, furthest_driver_score)
    elif num == 6:
        value_function = GreedyValueFunction(envt, two_sided_score)
    elif num == 7:
        value_function = GreedyValueFunction(envt, lambda_entropy_score)
    elif num == 8:
        value_function = PathBasedNN(device, envt, model_dir=model_dir,  embedding_dir=embedding_dir, load_model_loc=model_loc)
    elif num == 9:
        value_function = GreedyValueFunction(envt, lambda_variance_score)
    elif num == 10:
        value_function = PathBasedNN(device, envt, model_dir=model_dir,  embedding_dir=embedding_dir, load_model_loc=model_loc)
    elif num == 11:
        value_function = GreedyValueFunction(envt, lambda_entropy_rider_score)
    elif num == 12:
        value_function = PathBasedNN(device, envt, model_dir=model_dir,  embedding_dir=embedding_dir, load_model_loc=model_loc)
    elif num == 13:
        value_function = GreedyValueFunction(envt, lambda_variance_rider_score)
    elif num == 14:
        value_function = PathBasedNN(device, envt, model_dir=model_dir,  embedding_dir=embedding_dir, load_model_loc=model_loc)
    elif num == 15:
        value_function = PathBasedNN(device, envt, model_dir=model_dir,  embedding_dir=embedding_dir, load_model_loc=model_loc)
    else:
        raise ValueError(f'Num {num} not known')

    return value_function
