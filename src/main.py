import random
import sys
import os

import torch

cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(cur_dir))

from copy import deepcopy
import json
import time
import datetime
from tqdm.auto import tqdm
from typing import List
import numpy as np

from src.utils.Settings import Settings
from Environment import NYEnvironment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Oracle import Oracle
from value_functions import ValueFunctionFactory
from Experience import Experience
from Request import Request

import sys


# Get statistics by simulating the next epoch 
def get_statistics_next_epoch(agent, envt):
    settings = Settings.get_settings()
    def invalid_path_trace(issue: str) -> bool:
        if settings.has_value("print_verbose"):
            print(issue)
            print('Agent {}:'.format(agent.id))
            print('Requests -> {}'.format(agent.path.requests))
            print('Request Order -> {}'.format(agent.path.request_order))
            print()
        return False

    ret_dictionary = {'total_delivery_delay': 0, 'requests_served': 0}
    start_time = envt.current_time
    current_time = envt.current_time + agent.position.time_to_next_location
    current_location = agent.position.next_location
    current_capacity = agent.path.current_capacity

    for node_idx, node in enumerate(agent.path.request_order):
        next_location, deadline = agent.path.get_info(node)

        # Delay related checks
        travel_time = envt.get_travel_time(current_location, next_location)
        if current_time + travel_time > deadline:
            return invalid_path_trace('Does not meet deadline at node {}'.format(node_idx))

        current_time += travel_time
        current_location = next_location

        if current_time - start_time > envt.EPOCH_LENGTH:
            break

        # Updating available delay
        if node.expected_visit_time != current_time:
            invalid_path_trace("(Ignored) Visit time incorrect at node {}".format(node_idx))
            node.expected_visit_time = current_time

        if node.is_dropoff:
            ret_dictionary['total_delivery_delay'] += deadline - node.expected_visit_time
            ret_dictionary['requests_served'] += 1

        # Capacity related checks
        if current_capacity > envt.MAX_CAPACITY:
            return invalid_path_trace('Exceeds MAX_CAPACITY at node {}'.format(node_idx))

        if node.is_dropoff:
            next_capacity = current_capacity - 1
        else:
            next_capacity = current_capacity + 1
        if node.current_capacity != next_capacity:
            invalid_path_trace("(Ignored) Capacity incorrect at node {}".format(node_idx))
            node.current_capacity = next_capacity
        current_capacity = node.current_capacity

    return ret_dictionary


def get_profit_distribution(envt, scored_final_actions):
    profits = []
    agent_profits = []
    agent_indices = []
    for agent, (action, _) in enumerate(scored_final_actions):
        # Calculate the profit 
        for request in action.requests:
            dropoff = request.dropoff
            pickup = request.pickup
            travel_time = envt.get_travel_time(pickup, dropoff)
            action_profit = envt.profit_function(travel_time)

            if action_profit != 0:
                profits.append(action_profit)
                agent_profits.append(action_profit)
                agent_indices.append(agent)

    return profits, agent_profits, agent_indices


def run_epoch(envt,
              oracle,
              central_agent,
              value_function,
              DAY,
              is_training,
              run_name='train',
              agents_predefined=None,
              training_frequency: int = 1,
              use_tqdm=False):
    settings = Settings.get_settings()

    epoch_start = time.time()

    # INITIALISATIONS
    Experience.envt = envt
    # Initialising agents
    if agents_predefined is not None:
        agents = deepcopy(agents_predefined)
    else:
        initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training)
        agents = [LearningAgent(agent_idx, initial_state) for agent_idx, initial_state in enumerate(initial_states)]

    # ITERATING OVER TIMESTEPS
    print("DAY: {}".format(DAY))
    down_sample = 1
    if settings.has_value("down_sample") and settings.get_value("down_sample"):
        down_sample = settings.get_value("down_sample")
    request_generator = envt.get_request_batch(DAY, downsample=down_sample)
    total_value_generated = 0
    num_total_requests = 0

    ret_dictionary = {
        'epoch_requests_completed': [],
        'epoch_requests_accepted': [],
        'epoch_dropoff_delay': [],
        'epoch_requests_seen': [],
        'epoch_requests_accepted_profit': [],
        'epoch_each_agent_profit': [],
        'epoch_locations_all': [],
        'epoch_locations_accepted': []
    }
    if use_tqdm:
        iterator = tqdm(request_generator, total=1440)
    else:
        iterator = request_generator

    for current_requests in iterator:
        start = time.time()
        # Get new requests

        if settings.has_value("print_verbose") and settings.get_value("print_verbose"):
            print(f"Current time: {envt.current_time: >8} or {str(datetime.timedelta(seconds=envt.current_time)): >8} on DAY {DAY: <2}", end='\t')
            print(f"New requests: {len(current_requests)}", end='\t')

        ret_dictionary['epoch_locations_all'].append([i.pickup for i in current_requests])

        for i in current_requests:
            envt.requests_region[envt.labels[i.pickup]] += 1

        # Get feasible actions
        feasible_actions_all_agents = oracle.get_feasible_actions(agents, current_requests)

        # Score feasible actions
        experience = Experience(deepcopy(agents), feasible_actions_all_agents, envt.current_time, len(current_requests))
        scored_actions_all_agents = value_function.get_value([experience], is_training=is_training)

        # Choose actions for each agent
        scored_final_actions = central_agent.choose_actions(scored_actions_all_agents, is_training=is_training,
                                                            epoch_num=envt.num_days_trained)

        # Assign final actions to agents
        for agent_idx, (action, _) in enumerate(scored_final_actions):
            agents[agent_idx].path = deepcopy(action.new_path)

            position = experience.agents[agent_idx].position.next_location
            time_driven = 0
            for request in action.requests:
                time_driven += envt.get_travel_time(request.pickup,request.dropoff)

            time_to_request = sum([envt.get_travel_time(position,request.pickup) for request in action.requests])

            envt.driver_utilities[agent_idx] += max((time_driven-time_to_request), 0)

        # Calculate reward for selected actions
        rewards = []
        locations_served = []
        for action, _ in scored_final_actions:
            reward = len(action.requests)
            locations_served += [request.pickup for request in action.requests]
            for request in action.requests:
                envt.success_region[envt.labels[request.pickup]]+=1
            rewards.append(reward)
            total_value_generated += reward

        if settings.has_value("print_verbose") and settings.get_value("print_verbose"):
            print(f"Reward: {sum(rewards)}", end='\t')

        profits, agent_profits, agent_indices = get_profit_distribution(envt, scored_final_actions)
        envt.driver_profits[agent_indices] += agent_profits

        # Update
        if is_training:
            # Update replay buffer
            value_function.remember(experience)

            # Update value function every TRAINING_FREQUENCY timesteps
            if (int(envt.current_time) / int(envt.EPOCH_LENGTH)) % training_frequency == training_frequency - 1:
                value_function.update(central_agent)

        # Sanity check
        for agent in agents:
            assert envt.has_valid_path(agent)

        # Writing statistics to logs
        value_function.add_to_logs('rewards', sum(rewards), envt.current_time,
                                   f'{run_name}_day_{envt.num_days_trained}')
        avg_capacity = sum([agent.path.current_capacity for agent in agents]) / envt.NUM_AGENTS
        value_function.add_to_logs('avg_capacity', avg_capacity, envt.current_time,
                                   f'{run_name}_day_{envt.num_days_trained}')

        epoch_dictionary = get_statistics_next_epoch(agents[0], envt)
        for agent in agents[1 : ]:
            agent_dictionary = get_statistics_next_epoch(agent, envt)
            for key in agent_dictionary:
                    epoch_dictionary[key] += agent_dictionary[key]

        # Simulate the passing of time
        envt.simulate_motion(agents, current_requests)
        num_total_requests += len(current_requests)

        ret_dictionary['epoch_requests_completed'].append(epoch_dictionary['requests_served'])
        ret_dictionary['epoch_dropoff_delay'].append(epoch_dictionary['total_delivery_delay'])
        ret_dictionary['epoch_requests_accepted'].append(sum(rewards))
        ret_dictionary['epoch_requests_seen'].append(len(current_requests))
        ret_dictionary['epoch_requests_accepted_profit'].append(sum(profits))
        ret_dictionary['epoch_each_agent_profit'].append(list(zip(agent_indices, agent_profits)))
        ret_dictionary['epoch_locations_accepted'].append(locations_served)

        if settings.has_value("print_verbose") and settings.get_value("print_verbose"):
            print(f"Requests served {np.sum(ret_dictionary['epoch_requests_completed'])}", end='\t')
            print(f"accepted {sum(rewards)}", end='\t')

            print(f'{round(time.time() - start, 4): >5}')

    ret_dictionary['total_requests_accepted'] = total_value_generated
    ret_dictionary['total_requests'] = num_total_requests

    # Printing statistics for current epoch
    print('Number of requests accepted: {}'.format(total_value_generated))
    print('Number of requests seen: {}'.format(num_total_requests))

    print(f'Epoch day {DAY} took {round(time.time() - epoch_start, 1): >5}')
    return ret_dictionary


def seed_notebook(seed):
    # set environment variable for newer cuda versions
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # When running on the CuDNN backend two further options must be set for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_test_configuration(data_dir=None, model_dir=None, embedding_dir=None, use_tqdm=False):
    total_start = time.time()

    settings = Settings.get_settings()

    # Seed
    seed = 874
    if settings.has_value('seed'):
        seed = settings.get_value('seed')
    print(f'Setting seed to {seed}')
    seed_notebook(seed)

    PICKUP_DELAY = 300
    CAPACITY = 4
    DECISION_INTERVAL = 60
    START_HOUR: int = 0
    END_HOUR: int = 24
    NUM_EPOCHS: int = 1
    VALID_DAYS: List[int] = [2]
    VALID_FREQ: int = 1000
    SAVE_FREQ: int = VALID_FREQ
    Request.MAX_PICKUP_DELAY = PICKUP_DELAY
    Request.MAX_DROPOFF_DELAY = 2 * PICKUP_DELAY

    # Load in different settings
    training_days = settings.get_value("training_days")
    testing_days = settings.get_value("testing_days")
    num_agents = settings.get_value("num_agents")
    write_to_file = settings.get_value("write_file")
    value_num = settings.get_value("value_num")

    if settings.has_value("show_progress"):
        use_tqdm = settings.get_value("show_progress")

    if settings.has_value("pickup_delay"):
        PICKUP_DELAY = settings.get_value("pickup_delay")

    TRAINING_DAYS: List[int] = list(range(3, 3 + training_days))
    TEST_DAYS: List[int] = list(range(11, 11 + testing_days))

    # Initialising components
    if data_dir is None:
        envt = NYEnvironment(num_agents, START_EPOCH=START_HOUR * 3600, STOP_EPOCH=END_HOUR * 3600,
                             MAX_CAPACITY=CAPACITY, EPOCH_LENGTH=DECISION_INTERVAL)
    else:
        envt = NYEnvironment(num_agents, START_EPOCH=START_HOUR * 3600, STOP_EPOCH=END_HOUR * 3600,
                             MAX_CAPACITY=CAPACITY, EPOCH_LENGTH=DECISION_INTERVAL, DATA_DIR=data_dir)
    oracle = Oracle(envt)
    central_agent = CentralAgent(envt)
    central_agent.mode = "train"
    value_function = ValueFunctionFactory.num_to_value_function(envt, value_num, model_dir=model_dir, embedding_dir=embedding_dir)

    print("Input settings {}".format(settings.settings_dict))

    max_test_score = 0
    for epoch_id in range(NUM_EPOCHS):
        for day in TRAINING_DAYS:
            print('(TRAINING)', end=' ')
            train_data = run_epoch(envt, oracle, central_agent, value_function, day, is_training=True, run_name='train', use_tqdm=use_tqdm)
            total_requests_served = train_data['total_requests_accepted']
            value_function.add_to_logs('requests_served_per_day', total_requests_served, envt.num_days_trained, 'train')

            # Check validation score every VALID_FREQ days
            if (envt.num_days_trained % VALID_FREQ == VALID_FREQ - 1):
                val_requests_served = []
                for day in VALID_DAYS:
                    print('(VALIDATION)', end=' ')
                    val_data = run_epoch(envt, oracle, central_agent, value_function, day, is_training=False,
                                         run_name='val', use_tqdm=use_tqdm)
                    total_requests_served = val_data['total_requests_accepted']
                    val_requests_served.append(total_requests_served)
                val_avg_requests_served = sum(val_requests_served) / len(val_requests_served)
                value_function.add_to_logs('requests_served_per_day', val_avg_requests_served, envt.num_days_trained,
                                           'val')

                if hasattr(value_function, 'save_model'):
                    if (val_avg_requests_served > max_test_score or (envt.num_days_trained % SAVE_FREQ) == (
                            SAVE_FREQ - 1)):
                        value_function.save_model(
                            model_name=f'{type(value_function).__name__}_{envt.NUM_AGENTS}agent_{CAPACITY}capacity_{PICKUP_DELAY}delay_{DECISION_INTERVAL}interval_{envt.num_days_trained}_{val_avg_requests_served}.model')
                        max_test_score = max(val_avg_requests_served, max_test_score)

            envt.num_days_trained += 1
            if hasattr(value_function, 'save_model'):
                value_function.save_model(model_name=f'{num_agents}.model')

    # Reset the driver utilities
    envt.reset()
    central_agent.reset()
    training_time = round(time.time() - total_start, 1)
    print(f'Training took {training_time}s\n')
    test_start = time.time()

    write_dict = {}
    write_dict['settings'] = settings.settings_dict

    for day in TEST_DAYS:
        print('(TEST)', end=' ')
        initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training=False)
        agents = [LearningAgent(agent_idx, initial_state) for agent_idx, initial_state in enumerate(initial_states)]

        test_data = run_epoch(envt, oracle, central_agent, value_function, day, is_training=False, run_name='test',
                              agents_predefined=agents, use_tqdm=use_tqdm)
        total_requests_served = test_data['total_requests_accepted']

        value_function.add_to_logs('test_requests_served', total_requests_served, envt.num_days_trained)

        for key in test_data.keys():
            if key not in write_dict:
                if isinstance(test_data[key], list):
                    write_dict[key] = []
                else:
                    write_dict[key] = 0
            write_dict[key] += test_data[key]

        envt.num_days_trained += 1
    testing_time = round(time.time() - test_start, 1)
    print(f'Testing took {testing_time}s\n')

    total_time = round(time.time() - total_start)
    write_dict['settings']['time'] = total_time

    print(f'Running took {total_time}s')

    # Write our results
    if write_to_file:
        if hasattr(value_function, 'epoch_id'):
            write_dict['epoch_id'] = value_function.epoch_id
        write_dict['truncated_shapley'] = central_agent.truncated_shapley_final
        write_dict['random_shapley'] = central_agent.random_shapley_final
        write_dict['one_permutation_shapley'] = central_agent.one_permutation_shapley_final
        test_file_path = os.path.join(value_function.model_path, 'test_data.json')
        with open(test_file_path, 'w') as f:
            json.dump(write_dict, f, indent=4)

        times_files_path = os.path.join(value_function.model_path, 'times.json')
        times = {
            'training': training_time,
            'test': testing_time,
            'total': total_time
        }
        with open(times_files_path, 'w') as f:
            json.dump(times, f, indent=4)


def main():
    data_dir = None
    if len(sys.argv) > 1:
        Settings.create_settings(model_settings_name=sys.argv[1])
        if len(sys.argv) > 2:
            data_dir = sys.argv[2]
    else:
        Settings.create_settings()

    train_test_configuration(data_dir, embedding_dir='../models/embeds')


if __name__ == '__main__':
    main()
