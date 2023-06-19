from typing import Optional


import abc
import numpy as np
from typing import Union

class BaseMultiTaskSampler(metaclass=abc.ABCMeta):
    """
    Base class for Multi-Task sampler.

    This class provides a base structure and functionality for implementing
    custom Multi-Task sampling strategies in Multi-Task Learning frameworks.

    Args:
        task_dict (dict): A dictionary containing the task names and their corresponding datasets.
        rng (Union[int, np.random.RandomState, None]): Random number generator or seed value.

    """

    def __init__(self, task_dict: dict, rng: Union[int, np.random.RandomState, None]):
        self.task_dict = task_dict
        if isinstance(rng, int) or rng is None:
            rng = np.random.RandomState(rng)
        self.rng = rng
        self.task_names = list(task_dict.keys())

    @abc.abstractmethod
    def pop(self):
        """
        Returns the task name and dataset.
        """
        raise NotImplementedError()

    def iter(self):
        """
        Iterator method to yield the result of the pop method.
        """
        yield self.pop()


class SingleTaskSampler:
    """
    A sampler for a single task.

    Args:
        task_name (str): The name of the task.
        task: The task dataset.

    """
    def __init__(self, task_name, task):
        self.task_name = task_name
        self.task = task
        self.task_names = [task_name]

    def pop(self):
        return self.task_name, self.task

    def iter(self):
        yield self.pop()


class UniformMultiTaskSampler(BaseMultiTaskSampler):
    """
    A task sampler that uniformly samples tasks from a task dictionary.

    Inherits from:
        BaseMultiTaskSampler: A base class for Multi-Task samplers.

    Args:
        task_dict (dict): A dictionary containing the task names and their corresponding datasets.
        rng (Union[int, np.random.RandomState, None]): Random number generator or seed value.

    """
    def pop(self):
        task_name = self.rng.choice(list(self.task_dict))
        return task_name, self.task_dict[task_name]


class ProportionalMultiTaskSampler(BaseMultiTaskSampler):
    """
    A task sampler that samples tasks proportionally based on the number of examples in each task.

    Args:
        task_dict (dict): A dictionary mapping task names to task objects.
        rng (Union[int, np.random.RandomState]): Random number generator or seed.
        task_to_num_examples_dict (dict): A dictionary mapping task names to the number of examples in each task.
    """
    def __init__(
        self,
        task_dict: dict,
        rng: Union[int, np.random.RandomState],
        task_to_num_examples_dict: dict,
    ):
        super().__init__(task_dict=task_dict, rng=rng)
        assert task_dict.keys() == task_to_num_examples_dict.keys()
        self.task_to_examples_dict = task_to_num_examples_dict
        self.task_names = list(task_to_num_examples_dict.keys())
        self.task_num_examples = np.array([task_to_num_examples_dict[k] for k in self.task_names])
        self.task_p = self.task_num_examples / self.task_num_examples.sum()

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p)
        return task_name, self.task_dict[task_name]


class SpecifiedProbMultiTaskSampler(BaseMultiTaskSampler):
    """
    A Multi-Task sampler that samples tasks based on specified probabilities.

    Args:
        task_dict (dict): A dictionary mapping task names to task dataset.
        rng (Union[int, np.random.RandomState]): Random number generator or seed.
        task_to_unweighted_probs (dict): A dictionary mapping task names to unweighted probabilities.
    """
    def __init__(
        self,
        task_dict: dict,
        rng: Union[int, np.random.RandomState],
        task_to_unweighted_probs: dict,
    ):
        super().__init__(task_dict=task_dict, rng=rng)
        assert task_dict.keys() == task_to_unweighted_probs.keys()
        self.task_to_unweighted_probs = task_to_unweighted_probs
        self.task_names = list(task_to_unweighted_probs.keys())
        self.unweighted_probs_arr = np.array([task_to_unweighted_probs[k] for k in self.task_names])
        self.task_p = self.unweighted_probs_arr / self.unweighted_probs_arr.sum()

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p)
        return task_name, self.task_dict[task_name]


class TemperatureMultiTaskSampler(BaseMultiTaskSampler):
    """
    A Multi-Task sampler that samples tasks based on a temperature parameter.

    Args:
        task_dict (dict): A dictionary mapping task names to task data.
        rng (Union[int, np.random.RandomState]): Random number generator or seed.
        task_to_num_examples_dict (dict): A dictionary mapping task names to the number of examples for each task.
        temperature (float): The temperature parameter controlling the sampling distribution.
        examples_cap (Optional[int]): An optional maximum number of examples per task.
    """

    def __init__(
        self,
        task_dict: dict,
        rng: Union[int, np.random.RandomState],
        task_to_num_examples_dict: dict,
        temperature: float,
        examples_cap: Optional[int]=None,
    ):
        super().__init__(task_dict=task_dict, rng=rng)
        assert task_dict.keys() == task_to_num_examples_dict.keys()
        self.task_to_num_examples_dict = task_to_num_examples_dict
        self.temperature = temperature
        if examples_cap is None:
            self.examples_cap = max(list(self.task_to_num_examples_dict.values()))
        else:
            self.examples_cap = examples_cap
        self.task_names = list(task_to_num_examples_dict.keys())
        self.task_num_examples = np.array([task_to_num_examples_dict[k] for k in self.task_names])
        raw_n = self.task_num_examples.clip(max=self.examples_cap) ** (1 / self.temperature)
        self.task_p = raw_n / raw_n.sum()

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p)
        return task_name, self.task_dict[task_name]


class HybridMultiTaskSampler:
    """
    A Multi-Task sampler that combines multiple samplers using specified probabilities.

    Args:
        samplers (list): A list of individual task samplers.
        probabilities (Optional[ndarray]): An optional array of probabilities for each sampler.
        rng (Optional[Union[int, np.random.RandomState]]): Random number generator or seed.

    """
    def __init__(self, samplers, probabilities=None, rng=None):
        self.samplers = samplers
        if probabilities is None:
            self.probabilities = np.array([1. / len(samplers) for _ in samplers])
        else:
            self.probabilities = probabilities
        if isinstance(rng, int) or rng is None:
            rng = np.random.RandomState(rng)
        self.rng = rng
        self.task_names = sum([sampler.task_names for sampler in self.samplers], [])

    def pop(self):
        sampler = self.rng.choice(self.samplers, p=self.probabilities)
        return sampler.pop()

    def iter(self):
        yield self.pop()