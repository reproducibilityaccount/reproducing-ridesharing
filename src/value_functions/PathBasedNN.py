import os.path

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Tuple, List, Dict

from src.Environment import Environment
from src.Request import Request
from src.LearningAgent import LearningAgent
from src.value_functions.NeuralNetworkBased import NeuralNetworkBased


class PathBasedNNModel(nn.Module):

    def __init__(self, envt, embedding_dir):
        # DEFINE NETWORK STRUCTURE
        # Check if there are pretrained embeddings
        super().__init__()

        self.embedding_dim = 100
        self.lstm_output_dim = 200

        embed_path = os.path.join(embedding_dir, f'embedding_{self.embedding_dim}.weights')

        if os.path.isfile(embed_path):
            embedding_weight = torch.load(embed_path, map_location=torch.device('cpu'))
            print(f'Using pretrained location embeddings in {embed_path}')
        else:
            print(f'No pretrained embeddings found in {embed_path}')
            embedding_weight = None

        self.location_embed = nn.Embedding(embedding_dim=self.embedding_dim, num_embeddings=envt.NUM_LOCATIONS + 1,
                                           _weight=embedding_weight)
        # Freeze embedding if it was pretrained
        if embedding_weight is not None:
            self.location_embed.requires_grad_(requires_grad=False)

        self.act_fn = nn.ELU()
        # +1 due to concatenated input
        self.bilstm = nn.LSTM(self.embedding_dim + 1, self.lstm_output_dim, batch_first=True, bidirectional=True)
        for weights in self.bilstm.all_weights:
            # initialize input-to-hidden weights
            nn.init.xavier_uniform_(weights[0])
            # initialize hidden-to-hidden weights
            nn.init.orthogonal_(weights[1])

        self.linear_time_embedding = nn.Linear(1, 100)
        nn.init.xavier_uniform_(self.linear_time_embedding.weight)

        self.linear_state_embed_1 = nn.Linear(302, 300)
        nn.init.xavier_uniform_(self.linear_state_embed_1.weight)
        self.linear_state_embed_2 = nn.Linear(300, 300)
        nn.init.xavier_uniform_(self.linear_state_embed_2.weight)

        self.linear_out = nn.Linear(300, 1)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, path_location_input, delay_input, current_time_input, other_agents_input, num_requests_input):

        # get sequence lengths for masking
        lengths = torch.cumsum(path_location_input != 0, dim=1)[:, -1]

        # Get path and current locations' embeddings
        path_location_embed = self.location_embed(path_location_input)

        # Concatenate inputs for path embeddings
        path_input = torch.cat([path_location_embed, delay_input], dim=-1)
        # pack padded sequences
        masked_path_input = nn.utils.rnn.pack_padded_sequence(path_input, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)

        self.bilstm.flatten_parameters()
        # Get entire path's embedding
        path_embed_seq_bi, _ = self.bilstm(masked_path_input)
        # unpack padded sequences
        path_embed_seq_bi_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(path_embed_seq_bi, batch_first=True)
        # only use output of backward lstm part --> use flipped input sequence
        # take final hidden state (first for backward lstm) as embedding
        path_embed = path_embed_seq_bi_unpacked[:, 0, self.lstm_output_dim:]

        # Get current time's embedding
        current_time_embed = self.linear_time_embedding(current_time_input)
        current_time_embed = self.act_fn(current_time_embed)

        # Get Embedding for the entire thing
        state_embed = torch.cat([path_embed, current_time_embed, other_agents_input, num_requests_input], dim=-1)
        state_embed = self.linear_state_embed_1(state_embed)
        state_embed = self.act_fn(state_embed)
        state_embed = self.linear_state_embed_2(state_embed)
        state_embed = self.act_fn(state_embed)

        output = self.linear_out(state_embed)

        return output


class PathBasedNN(NeuralNetworkBased):

    def __init__(self, device, envt: Environment, load_model_loc: str = '', model_dir: str = '../models/', embedding_dir=None):
        super(PathBasedNN, self).__init__(device, envt, load_model_loc, model_dir, embedding_dir=embedding_dir)

    def _init_NN(self):
        return PathBasedNNModel(self.envt, self.embedding_dir)

    def load_model(self, model_path, device, envt):
        # load model
        if not os.path.isfile(model_path):
            return None
        state_dict = torch.load(model_path, device)
        model = PathBasedNNModel(envt, self.embedding_dir)
        model = model.to(device)
        model.load_state_dict(state_dict)
        print(f'Loading model weights ({model_path})')
        return model

    def _format_input(self, agent: LearningAgent, current_time: float, num_requests: float, num_other_agents: float) ->\
            Tuple[np.ndarray, np.ndarray, float, float, float, float]:
        # Normalising Inputs
        current_time_input = (current_time - self.envt.START_EPOCH) / (self.envt.STOP_EPOCH - self.envt.START_EPOCH)
        num_requests_input = num_requests / self.envt.NUM_AGENTS
        num_other_agents_input = num_other_agents / self.envt.NUM_AGENTS
        if np.mean(self.envt.driver_profits) != 0:
            agent_profit_input = (agent.profit - np.mean(self.envt.driver_profits)) / (np.std(self.envt.driver_profits))
        else:
            agent_profit_input = 0.0

        # For some reason, this mode was weird
        if self.load_model_loc == "../models/PathBasedNN_1000agent_4capacity_300delay_60interval_2_245261.h5":
            location_order: np.ndarray = np.zeros(shape=(self.envt.MAX_CAPACITY * 5 + 1,), dtype='int32')
            delay_order: np.ndarray = np.zeros(shape=(self.envt.MAX_CAPACITY * 5 + 1, 1)) - 1

        else:  # Getting path based inputs
            location_order: np.ndarray = np.zeros(shape=(self.envt.MAX_CAPACITY * 2 + 1,), dtype='int32')
            delay_order: np.ndarray = np.zeros(shape=(self.envt.MAX_CAPACITY * 2 + 1, 1)) - 1

        # Adding current location
        location_order[0] = agent.position.next_location + 1
        delay_order[0] = 1

        for idx, node in enumerate(agent.path.request_order):
            if idx >= 2 * self.envt.MAX_CAPACITY:
                break

            location, deadline = agent.path.get_info(node)
            visit_time = node.expected_visit_time

            location_order[idx + 1] = location + 1
            delay_order[idx + 1, 0] = (deadline - visit_time) / Request.MAX_DROPOFF_DELAY  # normalising

        return location_order, delay_order, current_time_input, num_requests_input, num_other_agents_input, agent_profit_input

    def _format_input_batch(self, all_agents_post_actions: List[List[LearningAgent]], current_time: float,
                            num_requests: int) -> Dict[str, Any]:
        input_: Dict[str, List[Any]] = {
            "path_location_input": [],
            "delay_input": [],
            "current_time_input": [],
            "other_agents_input": [],
            "num_requests_input": [],
            "agent_profit_input": []
        }

        # Format all the other inputs
        for agent_post_actions in all_agents_post_actions:
            current_time_input = []
            num_requests_input = []
            path_location_input = []
            delay_input = []
            other_agents_input = []
            agent_profit_input = []

            # Get number of surrounding agents
            current_agent = agent_post_actions[0]  # Assume first action is _null_ action
            num_other_agents = 0
            for other_agents_post_actions in all_agents_post_actions:
                other_agent = other_agents_post_actions[0]
                if (self.envt.get_travel_time(current_agent.position.next_location,
                                              other_agent.position.next_location) < Request.MAX_PICKUP_DELAY or
                        self.envt.get_travel_time(other_agent.position.next_location,
                                                  current_agent.position.next_location) < Request.MAX_PICKUP_DELAY):
                    num_other_agents += 1

            for agent in agent_post_actions:
                # Get formatted output for the state
                location_order, delay_order, current_time_scaled, num_requests_scaled, num_other_agents_scaled, agent_profit = self._format_input(
                    agent, current_time, num_requests, num_other_agents)

                current_time_input.append(num_requests_scaled)
                num_requests_input.append(num_requests)
                path_location_input.append(location_order)
                delay_input.append(delay_order)
                other_agents_input.append(num_other_agents_scaled)
                agent_profit_input.append(agent_profit)

            input_["current_time_input"].append(current_time_input)
            input_["num_requests_input"].append(num_requests_input)
            input_["delay_input"].append(delay_input)
            input_["path_location_input"].append(path_location_input)
            input_["other_agents_input"].append(other_agents_input)

        return input_
