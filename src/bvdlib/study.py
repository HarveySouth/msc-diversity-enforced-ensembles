from bvdlib.trial import Trial
from numpy.random import choice
from numpy.random import seed
from numpy import array
import numpy as np
import pickle

from tqdm import tqdm
from tqdm import trange

import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class Study:
    """
    Similar to Optuna's concept of Study and Trial, this class handles the running and results processing of bias-variance-diversity experiments

    Parameters
    ----------
    trial_space: ndarray of shape (n_parameter_values,)
        A list for all of the parameters to test in the experiment e.g. ensembles with 2, 3, 4, 5... members
    train_data: ndarray of shape (n_training_samples, n_dataset_features)
        The training data
    train_labels: ndarray of shape (n_training_samples,)
        The test labels
    test_data: ndarray of shape (n_test_samples, n_dataset_features)
        The test data
    test_labels: ndarray of shape (n_test_samples,)
        The test labels
    num_training: int
        The number of training samples to use for each trial in the experiment
    n_trials: int
        The number of trials per parameter
    decomp_fn: callable
        The decomposition function from decompose
    bootstrap: boolean
        If trial data sampling should be performed with or without replacement 
    exp_seed: int
        To seed generators for reproducible results

    Attributes
    ----------
    None

    """
    def __init__(self,
                 trial_space, 
                 train_data,
                 train_labels,
                 test_data,
                 test_labels,
                 num_training,
                 n_trials,
                 decomp_fn,
                 bootstrap = False,
                 exp_seed = 0) -> None:

        # Initialize experiment variables
        self.seed = exp_seed
        self.n_trials = n_trials
        self.trial_space = trial_space
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.decomp_fn = decomp_fn

        # Set the seed and generate subsets of the training data to use for each trial
        seed(self.seed) # np.random seed
        self.trial_idxs = self._gen_training_subsample_idxs(len(train_data), num_training, n_trials, replacement=bootstrap)
    
    def run_trials(self, trial_function):
        """
        Performs experiments required to calculate BVD decomposition_class and stores the results.

        Parameters
        ----------
        trial_function: callable
            a function using a trial object to take the trial data, train a model, 
            and return the relevant results of the trial on trial and test data

        Returns
        -------
        results_object: ResultsObject
            an instance of ResultsObject containing the results from the experiment

        """
        logger.info("Run Trial Start")
        # init results object
        results_object = ResultsObject("n_estimators", self.trial_space, self.decomp_fn,
                                                    n_test_splits=1)
        # for each parameter in the trial space
        for param_idx, param in enumerate(tqdm(self.trial_space, leave=True)):
            # initialize a list to contain each trial result
            trial_results = []
            # seed the trial
            seed(self.seed)
            # print the parameter index to display how many parameters are finished running
            info_str = "Parameter idx"  + str(param_idx)
            logger.info(info_str)
            # initialize the training loss
            total_train_loss = 0
            # for each trial
            for j in trange(self.n_trials):
                #create trial obj
                trial_idx = self.trial_idxs[j]
                param_dict = {"n_estimators": param}
                trial = Trial(self.train_data[trial_idx], 
                            self.train_labels[trial_idx],
                            param_dict)
                #run trial
                trial_result, train_loss = trial_function(trial)
                # update the average loss after a trial
                total_train_loss += (1/self.n_trials) * train_loss
                # append the trial result to the trial results list
                trial_results.append(trial_result)
            # after trials are finished for the parameter, convert to an array
            trial_results = array(trial_results)
            # perform a bvd decomposition on the trial results
            decomposition = self.decomp_fn(trial_results, self.test_labels)
            # format the errors for the result object
            errors = [total_train_loss, np.average(decomposition.expected_ensemble_loss)]
            # update the results object
            results_object.update_results(decomposition, param_idx, errors, split_idx=0)
        # after every trial for every parameter, return the results object
        logger.info("Run Trial End")
        return results_object


    def _gen_training_subsample_idxs(self, training_sample_n, trial_sample_n, trial_n, replacement=False):
        """
        generates sample indexes to be used in each trial.

        Parameters
        ----------
        training_sample_n: int
            number of training samples
        trial_sample_n: int
            number of samples to choose for a trial
        trial_n: int
            number of trials 
        replacement: boolean
            to sample training points for the 

        Returns
        -------
        all_indxs: list
            list of shape (trial_n, trial_sample_n)

        """
        all_indxs = []
        for i in range(trial_n):
            trial_idx = choice(training_sample_n, trial_sample_n, replace=replacement)
            all_indxs.append(trial_idx)
        return all_indxs
    


class ResultsObject(object):
    """
    Results from BVDExperiment are stored in ResultsObject instances.

    Parameters
    ----------
    parameter_name : str
        name of parameter that is varied over the course of the experiment
    parameter_values : list
        list of values that the varied parameter takes
    loss_func : str
        name of the loss function used for the decompose
    n_test_splits : int
        Number of separate folds unseen data is split into. Default is 2, with the first being the train split and the
        second being test

    Attributes
    ----------
    ensemble_risk : ndarray of shape (n_parameter_values, n_test_splits)
        The risk of the ensemble for each parameter value and test split
    ensemble_bias: ndarray of shape (n_parameter_values, n_test_splits)
        The biasof the ensemble for each paramter value and test split
    ensemble_variance: ndarray of shape (n_parameter_values, n_test_splits)
        The varianceof the ensemble for each parameter value and test split
    average_bias : ndarray of shape (n_parameter_values, n_test_splits)
        The average bias of the ensemble members for each parmater value and test split
    average_variance : ndarray of shape (n_parameter_values, n_test_splits)
        The average variance of the ensemble members for each parmater value and test split
    diversity : ndarray of shape (n_parameter_values, n_test_splits)
        The diversity for each parameter value and test split
    test_error : ndarray of shape (n_parameter_values, n_test_splits)
        The test error of the ensemble for each parameter value and test split
    train_error : ndarray of shape (n_parameter_values)
        The train error of the ensemble for each parameter value (each ensemble is evaluated only on data that it has seen
        during training.
    member_test_error : ndarray of shape (n_parameter_values, n_test_splits)
        The average test error of the ensemble for each parameter value and test split
    member_train_error : ndarray of shape (n_parameter_values)
        The average train error of the ensemble for each parameter value. Members are evaluated on data that was seen by the
        ensemble during training; there may be examples that were seen by the ensemble but not the individual member

    """

    def __init__(self, parameter_name, parameter_values, loss_func,
                 n_test_splits):
        n_parameter_values = len(parameter_values)
        self.loss_func = loss_func
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.n_test_splits = n_test_splits
        self.ensemble_risk = np.zeros((n_parameter_values, n_test_splits))
        self.ensemble_bias = np.zeros((n_parameter_values, n_test_splits))
        self.ensemble_variance = np.zeros((n_parameter_values, n_test_splits))
        self.average_bias = np.zeros((n_parameter_values, n_test_splits))
        self.average_variance = np.zeros((n_parameter_values, n_test_splits))
        self.diversity = np.zeros((n_parameter_values, n_test_splits))
        self.test_error = np.zeros((n_parameter_values, n_test_splits))
        self.train_error = np.zeros((n_parameter_values))
        self.member_test_error = np.zeros((n_parameter_values, n_test_splits))
        self.member_train_error = np.zeros((n_parameter_values))


    def update_results(self, decomp, param_idx, errors, split_idx=0, sample_weight=None):
        """
        Function used to update ResultsObject for a new parameter using Decomposition object and list of train/test errors

        Parameters
        ----------
        decomp : Decomposition
            Decomposition object for the experiment
        param_idx : int
            The index of the current parameter in the parameter_values
        errors : list of floats
            List containing (in order):
                Training error averaged over all runs of the experiment
                Test error averaged over all runs of the experiment
                (optional)
                Average member train error
                Average member test error

        Returns
        -------
        None

        """
        self.train_error[param_idx] = errors[0]  # overall error
        self.test_error[param_idx, split_idx] = errors[1][split_idx]  # overall error

        if len(errors) == 4:
            self.member_train_error[param_idx] = errors[2]  # avg member error
            self.member_test_error[param_idx, split_idx] = errors[3][split_idx]  # avg member error

        # We can tell if we have per trial test erorrs by checking if the length of the errors list is odd
        if len(errors) % 2 == 1:
            if not hasattr(self, "per_trial_test_errors"):
                self.per_trial_test_errors = np.zeros((len(self.parameter_values),
                                                       errors[-1].shape[0],
                                                       self.n_test_splits))
            # This also doesn't feel great, is it already filled?
            self.per_trial_test_errors[param_idx, :, split_idx] = errors[-1][:, split_idx]


        self.ensemble_bias[param_idx, split_idx] = np.average(decomp.ensemble_bias,
                                                              weights=sample_weight)

        self.ensemble_variance[param_idx, split_idx] = np.average(decomp.ensemble_variance,
                                                                                  weights=sample_weight)

        self.average_bias[param_idx, split_idx] = np.average(decomp.average_bias,
                                                             weights=sample_weight)

        self.average_variance[param_idx, split_idx] = np.average(decomp.average_variance,
                                                                 weights=sample_weight)

        self.diversity[param_idx, split_idx] = np.average(decomp.diversity, weights=sample_weight)

        self.ensemble_risk[param_idx, split_idx] = np.average(decomp.expected_ensemble_loss,
                                                              weights=sample_weight)
        logger.debug(f"Update Summary {param_idx},{split_idx}--"
                     f"ensemble bias: {self.ensemble_bias[param_idx, split_idx]},"
                     f" ensemble variance: {self.ensemble_variance[param_idx, split_idx]},"
                     f" average bias: {self.average_bias[param_idx, split_idx]},"
                     f"average variance: {self.average_variance[param_idx, split_idx]}, "
                     f"diversity: {self.diversity[param_idx, split_idx]},"
                     f" ensemble risk{self.ensemble_risk[param_idx, split_idx]},"
                     f" test error:{self.test_error[param_idx, split_idx]},"
                     f" train error{self.train_error[param_idx]}")
    def save_results(self, file_path):
            """
            Saves results object to pickle file for later use

            Parameters
            ----------
            file_path : str
                name of file (inlcuding directory) in which results are toe be stored

            Returns
            -------
            None

            """
            with open(file_path, "wb+") as file_:
                pickle.dump(self, file_)
