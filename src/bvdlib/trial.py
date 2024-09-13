class Trial:
    """
    Trial objects provide user access to relevant trial variables such as the data and trial settings during an experiment

    Parameters
    ----------
    x: ndarray of shape (trial_samples, n_features)
        training data for a trial
    y: ndarray of shape (trial_samples,)
        training labels for a trial
    param: dict
        dictionary with format {"parameter_name": parameter_object} 

    Attributes
    ----------
    x, y param. Identical to parameters

    """
    def __init__(self, x, y, param_dictionary) -> None:
        self.x = x
        self.y = y
        self.param_dict = param_dictionary

    @property
    def get_data(self):
        return self.x, self.y
    
    @property
    def get_params(self):
        return self.param_dict

    def get_singular_parameter(self):
        """
        Function to return a single parameter if there is only one entry in the parameter dictionary

        Returns
        ----------
        self.param_dict[self.param_dict.keys()[0]]: object
            the object within the single key dictionary

        """
        assert(len(self.param_dict.keys())==1)
        return self.param_dict[list(self.param_dict.keys())[0]]