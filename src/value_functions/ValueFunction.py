import os
from os import makedirs
from os.path import isdir
import time
from abc import ABC, abstractmethod
from typing import List, Tuple

from torch.utils.tensorboard import SummaryWriter

from src.Action import Action
from src.CentralAgent import CentralAgent
from src.Experience import Experience


class ValueFunction(ABC):
    """docstring for ValueFunction"""

    def __init__(self, model_dir: str = '../models/'):
        super(ValueFunction, self).__init__()

        self.model_dir = model_dir

        # Create directory of model
        model_time = round(time.time())
        model_nr = 0
        while True:
            model_path_candidate = os.path.join(model_dir, f'{type(self).__name__}_{model_time}_{model_nr}')
            if not isdir(model_path_candidate):
                break
            else:
                model_nr += 1

        self.model_path = model_path_candidate

        makedirs(self.model_path)

        self.summary_writer = SummaryWriter(os.path.join(self.model_path, 'logs'))

    def add_to_logs(self, tag: str, value: float, step: int, value_name: str = 'value') -> None:
        self.summary_writer.add_scalars(
            tag, {value_name: value}, step
        )
        self.summary_writer.flush()
    
    @abstractmethod
    def get_value(self, experiences: List[Experience], is_training: bool = False) -> List[List[Tuple[Action, float]]]:
        raise NotImplementedError

    @abstractmethod
    def update(self, central_agent: CentralAgent):
        raise NotImplementedError

    @abstractmethod
    def remember(self, experience: Experience):
        raise NotImplementedError
