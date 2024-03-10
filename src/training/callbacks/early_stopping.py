from src.training.callbacks import base

class EarlyStoppingCallback(base.BaseCallback):
    """
    Early Stopping Regularization callback
    to prevent network from overfitting during training.

    Parameters:
    ----------
    """
    def __init__(self, min_diff: float, patience: int):
        super(EarlyStoppingCallback, self).__init__()

        self.min_diff = min_diff 
        self.default_patience = patience
        self.prev_metric = None
        self.curr_patience = patience 

    def on_validation_end(self, **kwargs):
        curr_metric = kwargs.get("prev_metric")

        if self.prev_metric is None:
            self.prev_metric = curr_metric
            return 
        else:
            diff = self.curr_metric - self.prev_metric 
            if diff < self.min_diff:
                self.curr_patience -= 1
            else:
                self.curr_patience = self.default_patience

        if self.curr_patience == 0:
            raise StopIteration()