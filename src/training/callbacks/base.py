from abc import ABC, abstractmethod 
import typing
import pathlib
import os

class BaseCallback(ABC):
    """
    Base implementation of the training
    callback.
    """
    def __init__(self, log_dir: typing.Union[str, pathlib.Path] = None):
        if not os.path.exists(log_dir):
            os.makedirs(name=log_dir)

    @abstractmethod
    def tearDown(self, **kwargs):
        pass 

    @abstractmethod
    def on_init_start(self, **kwargs):
        pass 

    @abstractmethod
    def on_init_end(self, **kwargs):
        pass 

    @abstractmethod
    def on_train_batch_start(self, **kwargs):
        pass 

    @abstractmethod
    def on_train_batch_end(self, **kwargs):
        pass 
    
    @abstractmethod
    def on_train_epoch_end(self, **kwargs):
        pass 

    @abstractmethod
    def on_validation_start(self, **kwargs):
        pass 

    @abstractmethod
    def on_validation_end(self, **kwargs):
        pass

class TrainerCallbackMixin(ABC):

    callbacks: typing.List[BaseCallback] = []

    def tearDown(self, **kwargs):
        """
        Tear down callbacks after their execution
        """
        for callback in self.callbacks:
            callback.tearDown(**kwargs)

    def on_init_start(self, **kwargs):
        """
        Initialize callbacks when training starts
        """
        for callback in self.callbacks:
            callback.on_init_start(**kwargs)
    
    def on_init_end(self, **kwargs):
        """
        Triggers end of initialization for each callback
        after it ends
        """
        for callback in self.callbacks:
            callback.on_init_end(**kwargs)

    def on_train_batch_start(self, **kwargs):
        """
        Triggers each callback on start of the 
        training batch
        """
        for callback in self.callbacks:
            callback.on_batch_start(**kwargs)

    def on_train_batch_end(self, **kwargs):
        """
        Triggers each callback on ened of the
        training batch
        """
        for callback in self.callbacks:
            callback.on_batch_end(**kwargs)

    def on_train_epoch_end(self, **kwargs):
        """
        Triggers each callback to end the 
        training epoch
        """
        for callback in self.callbacks:
            callback.on_train_epoch_end(**kwargs)

    def on_validation_start(self, **kwargs):
        """
        Triggers each callback to activate
        on start of the model validation 
        """
        for callback in self.callbacks:
            callback.on_validation_start(**kwargs)

    def on_validation_end(self, **kwargs):
        """
        Triggers each callback to activate
        after validation of the network ends.
        """
        for callback in self.callbacks:
            callback.on_validation_end(**kwargs)
