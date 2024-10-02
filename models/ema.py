import torch

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        """
        Update the moving average of the model parameters.
        Args:
            ma_model: The model with moving average parameters.
            current_model: The current model being trained.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        Compute the updated average using exponential moving average formula.
        Args:
            old: Old parameter value.
            new: New parameter value.
        Returns:
            Updated parameter value.
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=1000):
        """
        Perform a step of EMA update.
        Args:
            ema_model: The model with moving average parameters.
            model: The current model being trained.
            step_start_ema: The step at which to start EMA updates.
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Reset the EMA model parameters to the current model parameters.
        Args:
            ema_model: The model with moving average parameters.
            model: The current model being trained.
        """
        ema_model.load_state_dict(model.state_dict())
