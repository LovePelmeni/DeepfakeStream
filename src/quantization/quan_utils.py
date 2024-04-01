import torch

def get_observer_by_name(observer_name: str):
    if observer_name.lower() == "percentile":
        return torch.ao.quantization.observer.PercentileObserver

    if observer_name.lower() == "histogram":
        return torch.ao.quantization.observer.HistogramObserver

    if observer_name.lower() == "min_max":
        return torch.ao.quantization.observer.MinMaxObserver

    if observer_name.lower() == "moving_min_max":
        return torch.ao.quantization.observer.MovingAverageMinMaxObserver
    else:
        raise NotImplemented()