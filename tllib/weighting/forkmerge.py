
def sum_all_weights(theta_dict, lambda_space, no_sum=[]):
    """
    Combines the model parameters from multiple tasks.

    Args:
        theta_dict (dict): A dictionary containing parameter dictionaries for each task.
        lambda_space (dict): A list of lambda values representing the weights for combining parameters.
        no_sum (list, optional): A list of parameter names that should not be included in the weighted combination.
            Defaults to an empty list.

    Returns:
        dict: The combined weighted parameter.

    """
    counter = {}
    for state_dict in theta_dict.values():
        for p_name in state_dict.keys():
            if p_name not in counter:
                counter[p_name] = 0
            counter[p_name] += 1

    shared_param = [p_name for p_name in counter.keys() if counter[p_name] == len(theta_dict)]
    for task, state_dict in theta_dict.items():
        state_dict = {
            p_name: state_dict[p_name] for p_name in state_dict.keys() if p_name in shared_param
        }
        theta_dict[task] = state_dict

    theta_0 = theta_dict[list(theta_dict.keys())[0]]
    lambda_space = {k: v / sum(lambda_space.values()) for k, v in lambda_space.items()}

    theta = {}
    for key in theta_0.keys():
        if not any(ns in key for ns in no_sum):
            theta[key] = 0
            for task_name in theta_dict.keys():
                theta[key] += lambda_space[task_name] * theta_dict[task_name][key]
        else:
            theta[key] = theta_0[key]
    return theta


class ForkMergeWeightedCombiner:
    """
    ForkMerge Combiner that combines parameters from multiple tasks  based on performance evaluations.

    Args:
        task_names (list): A list of task names.
        evaluate_function (callable): A function that evaluates the performance of a model parameter.
        lambda_space (list): A list of lambda values representing the weights for combining parameters.
        debug (bool, optional): Flag to enable debug mode. Defaults to False.

    """
    def __init__(self, task_names, evaluate_function, lambda_space, debug=False):
        self.task_names = task_names
        self.evaluate_function = evaluate_function
        self.lambda_space = lambda_space
        self.debug = debug

    def combine(self, theta_dict, performance_dict=None):
        """
        Args:
            theta_dict (dict): A dictionary containing parameter dictionaries for each task.
            performance_dict (dict, optional): A dictionary containing performance evaluations for each task. Defaults to None.

        Returns:
            tuple: A tuple containing the combined parameters (theta) and the corresponding lambda values used for combination.

        """
        if performance_dict is None:
            performance_dict = {}
            for task_name, theta in theta_dict.items():
                performance_dict[task_name] = self.evaluate_function(theta)

        # sort model parameters in decreasing order of performance
        ranked_performances = sorted(performance_dict.items(), key=lambda kv: kv[1], reverse=True)
        if self.debug:
            print("ranked_performances:", ranked_performances)
        lambda_dict = {name: 0 for name, _ in ranked_performances}
        lambda_dict[ranked_performances[0][0]] = 1

        criteria_dict = {}

        best_criteria = 0

        for i, (name, _) in enumerate(ranked_performances[1:]):
            best_lambda = 0
            upper_bound = sum(lambda_dict.values()) / (i + 1)
            for lambda_ in self.lambda_space:
                new_lambda = lambda_ * upper_bound
                lambda_dict[name] = new_lambda
                theta = sum_all_weights(theta_dict, lambda_dict)

                val_criteria = self.evaluate_function(theta)

                # remember best acc@1 and save checkpoint
                if val_criteria > best_criteria:
                    best_criteria = val_criteria
                    best_lambda = new_lambda
                if self.debug:
                    criteria_dict["{}_{}".format(name, new_lambda)] = val_criteria
                    print("lambda: {}, val_criteria: {}".format(new_lambda, val_criteria))
            lambda_dict[name] = best_lambda
        if self.debug:
            print("Combination Result: {} Best lambda: {} Best criteria: {}".
                  format(criteria_dict, lambda_dict, best_criteria))

        theta = sum_all_weights(theta_dict, lambda_dict)
        return theta, lambda_dict