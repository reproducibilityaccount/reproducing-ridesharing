import os
from abc import abstractmethod
from copy import deepcopy
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src import Util
from src.Action import Action
from src.CentralAgent import CentralAgent
from src.Environment import Environment
from src.ReplayBuffer import PrioritizedReplayBuffer
from src.LearningAgent import LearningAgent
from src.Experience import Experience
from src.value_functions.ValueFunction import ValueFunction


class NeuralNetworkBased(ValueFunction):
    """docstring for NeuralNetwork"""

    def __init__(self, device, envt: Environment, load_model_loc: str, model_dir: str = '../models/', embedding_dir=None,
                 gamma: float = 0.9, batch_size_fit: int = 32, batch_size_predict: int = 8192, target_update_tau: float = 0.1):
        super(NeuralNetworkBased, self).__init__(model_dir)

        # Initialise Constants
        self.device = device
        self.envt = envt
        self.gamma = gamma
        self.batch_size_fit = batch_size_fit
        self.BATCH_SIZE_PREDICT = batch_size_predict
        self.target_update_tau = target_update_tau
        self.load_model_loc = load_model_loc

        self.epoch_id = 0

        # Get Replay Buffer
        min_len_replay_buffer = 1e6 / self.envt.NUM_AGENTS
        epochs_in_episode = (self.envt.STOP_EPOCH - self.envt.START_EPOCH) / self.envt.EPOCH_LENGTH
        len_replay_buffer = max((min_len_replay_buffer, epochs_in_episode))
        self.replay_buffer = PrioritizedReplayBuffer(MAX_LEN=int(len_replay_buffer))

        if embedding_dir is None:
            embedding_dir = model_dir
        self.embedding_dir = embedding_dir

        # Get NN Model
        self.model: nn.Module = self.load_model(load_model_loc, device, envt)
        if self.model is None:
            self.model = self._init_NN()
        self.model = self.model.to(self.device)

        # Define Loss and Compile
        self.optimizer = optim.Adam(self.model.parameters(), eps=1e-07)
        self.loss_module = nn.MSELoss(reduction='none')

        # Get target-NN
        self.target_model = deepcopy(self.model)

    # Define soft-update function for target_model_update
    # Essentially weighted average between weights
    def update_target_model(self):
        target_weights = self.target_model.parameters()
        source_weights = self.model.parameters()

        with torch.no_grad():
            for target_weight, source_weight in zip(target_weights, source_weights):
                target_weight.copy_(self.target_update_tau * source_weight + (1. - self.target_update_tau) * target_weight)

    @abstractmethod
    def load_model(self, model_path, device, envt):
        raise NotImplementedError()

    @abstractmethod
    def _init_NN(self):
        raise NotImplementedError()

    @abstractmethod
    def _format_input_batch(self, agents: List[List[LearningAgent]], current_time: float, num_requests: int):
        raise NotImplementedError

    def save_model(self, model_path=None, model_name='model.model'):
        if model_path is None:
            model_path = self.model_path
        torch.save(self.model.state_dict(), os.path.join(model_path, model_name))

    def _get_input_batch_next_state(self, experience: Experience) -> Dict[str, np.ndarray]:
        # Move agents to next states
        all_agents_post_actions = []
        agent_num = 0
        for agent, feasible_actions in zip(experience.agents, experience.feasible_actions_all_agents):
            agents_post_actions = []
            for action in feasible_actions:
                # Moving agent according to feasible action
                agent_next_time = deepcopy(agent)
                assert action.new_path
                agent_next_time.profit = Util.change_profit(self.envt, action) + self.envt.driver_profits[agent_num]
                agent_next_time.path = deepcopy(action.new_path)
                self.envt.simulate_motion([agent_next_time], rebalance=False)

                agents_post_actions.append(agent_next_time)
            all_agents_post_actions.append(agents_post_actions)
            agent_num += 1

        next_time = experience.time + self.envt.EPOCH_LENGTH

        # Return formatted inputs of these agents
        return self._format_input_batch(all_agents_post_actions, next_time, experience.num_requests)

    def _flatten_NN_input(self, NN_input: Dict[str, np.ndarray]) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], List[int]]:
        shape_info: List[int] = []

        for key, value in NN_input.items():
            # Remember the shape information of the inputs
            if not shape_info:
                cumulative_sum = 0
                shape_info.append(cumulative_sum)
                for idx, list_el in enumerate(value):
                    cumulative_sum += len(list_el)
                    shape_info.append(cumulative_sum)

            # Reshape
            NN_input[key] = np.array([element for array in value for element in array])

        NN_input = {key: torch.Tensor(value) for key, value in NN_input.items()}
        NN_input['path_location_input'] = NN_input['path_location_input'].int()

        NN_input['current_time_input'] = NN_input['current_time_input'][..., None]
        NN_input['other_agents_input'] = NN_input['other_agents_input'][..., None]
        NN_input['num_requests_input'] = NN_input['num_requests_input'][..., None]

        # convert dict to input tuple
        NN_input = (
            NN_input['path_location_input'].to(self.device),
            NN_input['delay_input'].to(self.device),
            NN_input['current_time_input'].to(self.device),
            NN_input['other_agents_input'].to(self.device),
            NN_input['num_requests_input'].to(self.device),
        )

        return NN_input, shape_info

    def _reconstruct_NN_output(self, NN_output: np.ndarray, shape_info: List[int]) -> List[List[int]]:
        # Flatten output
        NN_output = NN_output.flatten()

        # Reshape
        assert shape_info
        output_as_list = []
        for idx in range(len(shape_info) - 1):
            start_idx = shape_info[idx]
            end_idx = shape_info[idx + 1]
            list_el = NN_output[start_idx:end_idx].tolist()
            output_as_list.append(list_el)

        return output_as_list

    def _format_experiences(self, experiences: List[Experience], is_current: bool) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], List[int]]:
        action_inputs_all_agents = None
        for experience in experiences:
            # If experience hasn't been formatted, format it
            if not (self.__class__.__name__ in experience.representation):
                experience.representation[self.__class__.__name__] = self._get_input_batch_next_state(experience)

            if is_current:
                batch_input = self._format_input_batch([[agent] for agent in experience.agents], experience.time, experience.num_requests)
            else:
                batch_input = deepcopy(experience.representation[self.__class__.__name__])

            if action_inputs_all_agents is None:
                action_inputs_all_agents = batch_input
            else:
                for key, value in batch_input.items():
                    action_inputs_all_agents[key].extend(value)
        assert action_inputs_all_agents is not None

        return self._flatten_NN_input(action_inputs_all_agents)

    def get_value(self, experiences: List[Experience], network: nn.Module = None, is_training: bool = False) -> List[List[Tuple[Action, float]]]:
        # Format experiences
        action_inputs_all_agents, shape_info = self._format_experiences(experiences, is_current=False)

        # set model mode
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        # Score experiences
        if network is None:
            expected_future_values_all_agents = self.model(*action_inputs_all_agents)
        else:
            expected_future_values_all_agents = network(*action_inputs_all_agents)

        # Format output
        expected_future_values_all_agents = self._reconstruct_NN_output(expected_future_values_all_agents, shape_info)

        # Get Q-values by adding associated rewards
        def get_score(action: Action, value: float, driver_num: int):
            return self.envt.get_reward(action, driver_num) + self.gamma * value

        driver_num = 0
        feasible_actions_all_agents = [feasible_actions for experience in experiences for feasible_actions in experience.feasible_actions_all_agents]

        scored_actions_all_agents: List[List[Tuple[Action, float]]] = []
        for expected_future_values, feasible_actions in zip(expected_future_values_all_agents, feasible_actions_all_agents):
            scored_actions = [(action, get_score(action, value, driver_num)) for action, value in zip(feasible_actions, expected_future_values)]
            scored_actions_all_agents.append(scored_actions)
            driver_num += 1
            driver_num %= self.envt.NUM_AGENTS

        return scored_actions_all_agents

    def remember(self, experience: Experience):
        self.replay_buffer.add(experience)

    def update(self, central_agent: CentralAgent, num_samples: int = 3):
        # Check if replay buffer has enough samples for an update
        # Epochs we need
        num_min_train_samples = int(5e5 / self.envt.NUM_AGENTS)

        if num_min_train_samples > len(self.replay_buffer):
            return

        # SAMPLE FROM REPLAY BUFFER
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            beta = min(1, 0.4 + 0.6 * (self.envt.num_days_trained / 200.0))
            experiences, orig_weights, batch_idxes = self.replay_buffer.sample(num_samples, beta)
        else:
            experiences = self.replay_buffer.sample(num_samples)
            orig_weights = None
            batch_idxes = np.zeros(len(experiences))

        # ITERATIVELY UPDATE POLICY BASED ON SAMPLE
        for experience_idx, (experience, batch_idx) in enumerate(zip(experiences, batch_idxes)):
            # Flatten experiences and associate weight of batch with every flattened experience
            if orig_weights is not None:
                weights = torch.tensor([orig_weights[experience_idx]] * self.envt.NUM_AGENTS, device=self.device, dtype=torch.float32).view(-1, 1)
            else:
                weights = None

            # GET TD-TARGET
            # Score experiences
            # So now we have, for each agent, a list of actions with their score, and we'll run an ILP over this
            scored_actions_all_agents = self.get_value([experience], network=self.target_model)  # type: ignore

            # Run ILP on these experiences to get expected value at next time step
            value_next_state = []
            for idx in range(0, len(scored_actions_all_agents), self.envt.NUM_AGENTS):
                final_actions = central_agent.choose_actions(scored_actions_all_agents[idx:idx + self.envt.NUM_AGENTS], is_training=False)
                value_next_state.extend([score for _, score in final_actions])

            supervised_targets = torch.tensor(value_next_state, dtype=torch.float32, device=self.device).view(-1, 1)

            # So we want to predict, from "action_inputs_all_agents", predict the "supervised_targets"
            # Supervised targets is the values chosen, which is from scores + ILP
            # Action_inputs_all_agents = List of all experiences
            # UPDATE NN BASED ON TD-TARGET
            action_inputs_all_agents, _ = self._format_experiences([experience], is_current=True)
            train_losses = self.fit(action_inputs_all_agents, supervised_targets, sample_weights=weights)

            # Write to logs
            avg_loss = sum(train_losses) / len(train_losses)
            self.add_to_logs('loss', avg_loss, self.epoch_id)

            # Update weights of replay buffer after update
            if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                # Calculate new squared errors
                predicted_values = self.model(*action_inputs_all_agents)
                loss = ((predicted_values - supervised_targets) ** 2 + 1e-6).mean()
                loss = loss.cpu().detach().numpy()
                # Update priorities
                self.replay_buffer.update_priorities([batch_idx], [loss])

            # Soft update target_model based on the learned model
            self.update_target_model()

            self.epoch_id += 1

    def fit(self, inputs, targets, sample_weights):

        if sample_weights is None:
            sample_weights_tensor = torch.ones(targets.shape)
        else:
            sample_weights_tensor = sample_weights
        train_dataset = TensorDataset(*inputs, sample_weights_tensor, targets)

        data_loader = DataLoader(train_dataset, self.batch_size_fit)

        train_losses = []
        self.model.train()
        for batch_data in data_loader:
            batch_x = batch_data[:5]
            batch_weights = batch_data[5]
            batch_labels = batch_data[-1]
            # Training
            self.optimizer.zero_grad()
            preds = self.model(*batch_x)

            losses = self.loss_module(preds, batch_labels)
            losses = losses * batch_weights
            loss = losses.mean()

            loss.backward()

            self.optimizer.step()

            # Training statistics
            train_losses.append(loss.cpu().detach().numpy())

        return train_losses
