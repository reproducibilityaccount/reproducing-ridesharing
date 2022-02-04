from typing import List, Tuple

from src.Action import Action
from src.Experience import Experience
from src.value_functions.ValueFunction import ValueFunction


class GreedyValueFunction(ValueFunction):
    def __init__(self, envt, score_function, model_dir="../logs/ValueFunctionLogs/"):
        super(GreedyValueFunction, self).__init__(model_dir)
        self.envt = envt
        self.score_function = score_function

    def get_value(self, experiences: List[Experience], is_training: bool = False) -> List[List[Tuple[Action, float]]]:
        scored_actions_all_agents: List[List[Tuple[Action, float]]] = []
        for experience in experiences:
            for i, feasible_actions in enumerate(experience.feasible_actions_all_agents):
                scored_actions: List[Tuple[Action, float]] = []
                for action in feasible_actions:
                    assert action.new_path

                    # Takes in an environment, action, and a driver number
                    score = self.score_function(self.envt, action, experience.agents[i], i)
                    scored_actions.append((action, score))
                scored_actions_all_agents.append(scored_actions)

        return scored_actions_all_agents

    def update(self, *args, **kwargs):
        pass

    def remember(self, *args, **kwargs):
        pass
