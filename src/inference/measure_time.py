from torch import nn 
import typing 
import torch 
import numpy 
import facenet_pytorch

def measure_classifier_inference_time(
    network: nn.Module, 
    batch_size: int, 
    img_shape: typing.Tuple,
    train_device: typing.Literal['cuda', 'mps', 'cpu'],
    total_repetitions: int = 100,
    warmup_iters: int = 10
):
    """
    Function responsible for measuring inference time
    of the network.
    
    Parameters:
    -----------
    
    network: nn.Module - neural network for testing
    batch_size: (int) - size of the batch of images you want to test on
    img_shape - (tuple) - shape of the image inside the batch
    train_device - str - device for training 
    total_repetitions - (int) - total number of measure repetitions to make
    warmup_iters: (int) - number of iterations to warmup GPU
    
    Returns:
        - average time across all repetitions in milliseconds.
    """
    # generating data for running inference
    gpu_data = torch.stack([
        torch.randn(size=img_shape).to(torch.float32).permute(2, 0, 1) 
        for _ in range(batch_size)
    ])
    
    # connecting neural network
    gpu_network = network.to(train_device)
     
    # running warmup repetitions
    for _ in range(warmup_iters):
        _ = gpu_network.forward(gpu_data.to(train_device))
        
    # running repetitions
    
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    times = []
    
    with torch.no_grad():
        for _ in range(total_repetitions):
            starter.record()
            _ = gpu_network.forward(gpu_data.to(train_device))
            ender.record()
            torch.cuda.synchronize()
            time = starter.elapsed_time(ender)
            times.append(time / 100) # converting to seconds by dividing by 100
    return numpy.mean(times)

def measure_face_detector_inference_time(
    detector: facenet_pytorch.MTCNN,
    input_images: list,
    total_repetitions: int,
    warmup_iters: int,
    inf_device: str
):
    """
    Measures the approximate
    inference time of the face detector
    on a given set of input images
    
    Parameters:
    -----------
    detector: MTCNN face detector
    input_images - list of numpy images
    total_repetitions - total number of times to repeat measure iteration
    warmup_iters - number of iterations for gpu warmup
    inf_device - device, used for inference test (usually the one used in production env)
    """
    detector.device = torch.device(inf_device)
    gpu_data = [torch.from_numpy(img).permute(2, 0, 1) for img in input_images]

    for _ in range(warmup_iters):
        _, _ = detector.detect-(gpu_data[0].unsqueeze(0).to(inf_device))

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    avg_times = []

    gpu_data = torch.stack(gpu_data).to(inf_device)

    for _ in range(total_repetitions):
        starter.record()
        _, _ = detector.detect(gpu_data)
        ender.record()
        torch.cuda.synchronize()
        total_time = ender.elapsed_time(starter) / len(input_images)
        avg_times.append(total_time)
    return numpy.mean(avg_times)
        