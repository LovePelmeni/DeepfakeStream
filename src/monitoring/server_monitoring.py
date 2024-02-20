import socket 
import psutil
import torch
import collections
import typing
import platform

def parse_model_consumption_info():
    """
    Returns information about model
    consumption of the resources.
    
    General Information:
        - number of model FLOPS
        - MAC
        - GPU's, used for inference
        - Percentage of GPU utilization for each instance
    """
    info = collections.defaultdict(typing.Any)
    info['gpus'] = []

    if torch.cuda.is_available():
        available_gpus = [
            torch.cuda.device(idx) 
            for idx in range(len(torch.cuda.device_count()))
        ]
        for gpu_device in available_gpus:
            dev_props = torch.cuda.get_device_properties(device=gpu_device)
            gpu_info = {
                "name": dev_props.name,
                "total_capacity": dev_props.total_memory,
                "multi_processor_count": dev_props.multi_processor_count,
                "reserved_memory": torch.cuda.memory_reserved(device=gpu_device),
                "allocated_memory_perc": torch.cuda.memory_allocated(device=gpu_device) / dev_props.total_memory,
            }
            info['gpus'].append(gpu_info)
    else:
        info['gpus'] = []
    
    return info

def parse_server_info():
    """
    Returns information about 
    current state of the system
    
    General information:
        1. name of the host machine.
        2. IP address of the machine
        3. Information about CPU
        4. Information about Virtual Memory (RAM)
        5. Information about GPU devices
    """
    return {
        "host_name": socket.gethostname(),
        "ip_address": socket.gethostbyname(socket.gethostname()),
        "host_os": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.architecture(),
        "ram": {
            "total_capacity": psutil.virtual_memory().total,
            "used": psutil.virtual_memory().percent,
            "available": psutil.virtual_memory().available * 100 / psutil.virtual_memory().total,
        },
        "cpu": {
            "total_capacity": psutil.cpu_percent(),
            "total_cpus": psutil.cpu_count(),
            "cpu_frequency_info": psutil.cpu_freq()
        }
    }



