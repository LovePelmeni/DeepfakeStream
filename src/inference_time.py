from torch import cuda
from torch import nn 
import numpy

def measure_inference_time(
    network: nn.Module, 
    test_data: nn.Module, 
    repetitions: int = 100
):
    """
    Function measures inference time for a given 
    network 
    """
    network = network.cuda()
    test_data = test_data.cuda()
    starter, ender = cuda.Event(), cuda.Event()
    avg_time = []
    for _ in range(10):
        _ = network.forward(test_data)
    for _ in range(repetitions):
        starter.record()
        _ = network.forward(test_data)
        ender.record()
        inf_time_ms = starter.elapsed_time(ender)
        avg_time.append(inf_time_ms)
    return numpy.mean(avg_time)



