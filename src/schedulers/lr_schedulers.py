from torch.optim.lr_scheduler import _LRScheduler 
from torch import nn

class StepLRScheduler(_LRScheduler):
    """
    Step Learning Rate Scheduler:
    
    Parameters:
    -----------
    optimizer - (nn.Module) - optimizer
    step_size: int - size of the step
    gamma - factor to reduce lr by.
    last_epoch - tracks current epoch.
    """
    def __init__(self, 
        optimizer: nn.Module, 
        step_size: int,
        gamma: float,
        last_epoch: int = -1
    ):
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)
        self.step_size = step_size 
        self.gamma = gamma 
        self.optimizer = optimizer
        self.current_step = 0

    def get_lr(self):

        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]
    
class PolyLRScheduler(_LRScheduler):

    """
    Polynomial Learning Rate Scheduler
    Parameters:
    -----------
    optimizer (nn.Module) - optimizer of the network
    max_iters (int) - maximum number of iterations
    power (float) - power of the polynomial, default 1.0.
    last_epoch (int) - tracks current epoch of the scheduler
    """

    def __init__(self, optimizer: nn.Module, max_iters=100, power=0.9, last_epoch: int = -1):
        self.optimizer = optimizer 
        self.max_iters = max_iters 
        self.power = power 
        self.last_epoch = last_epoch 
        super(PolyLRScheduler, self).__init__(optimizer=optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        self.last_epoch = (self.last_epoch) % self.max_iters 
        return [base_lr for base_lr in self.base_lrs]

class ExponentialLRScheduler(_LRScheduler):

    """
    Exponential Learning Rate Scheduler
    Parameters:
    ----------
    optimizer (nn.Module) - optimizer of the network 
    gamma (float) - multiplicative factor of learning rate decay
    last_epoch (int) current epoch of the scheduler
    """ 

    def __init__(self, optimizer: nn.Module, gamma: float, last_epoch: int = -1):
        self.gamma = gamma 
        super(ExponentialLRScheduler, self).__init__(optimizer=optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        if self.last_epoch <= 0:
            return self.base_lrs 

        return [(base_lr * self.gamma ** self.last_epoch) for base_lr in self.base_lrs]


