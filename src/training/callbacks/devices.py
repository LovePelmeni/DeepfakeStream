from src.training.callbacks import base
import time
import psutil
from torch.utils.tensorboard.writer import SummaryWriter
import numpy
import torch.cuda

class CPUInferenceCallback(base.BaseCallback):
    """
    Performance callback for tracking CPU latency and
    average execution time for one batch of image data.
    NOTE:
        use this callback if you train network on CPU.
        otherwise consider 'GPUInferenceCallback' class
    """
    def on_init_start(self, **kwargs):
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.cpu_latency_measures = []
        self.cpu_time_measures = []

    def tearDown(self, **kwargs):
        if hasattr(self, 'writer'):
            self.writer.close()

    def on_train_batch_start(self, **kwargs):
        self._start_timer = time.time()
        self._start_latency_timer = psutil.cpu_percent()

    def on_train_batch_end(self, ):
        self._end_timer = time.time()
        self._end_latency_timer = psutil.cpu_percent()
        curr_diff_time = self._end_timer - self._start_timer 
        curr_latency_diff_time = self._end_latency_timer - self._start_latency_timer
        self.cpu_time_measures.append(curr_diff_time)
        self.cpu_latency_measures.append(curr_latency_diff_time)

    def on_train_epoch_end(self, **kwargs):

        global_step = kwargs.get("global_step", None)
        avg_cpu_time = numpy.mean(self.cpu_time_measures)
        avg_cpu_latency_time = numpy.mean(self.cpu_latency_measures)

        if hasattr(self, 'curr_diff_time'):
            self.writer.add_scalar(
                tag='cpu_time', 
                scalar_value=avg_cpu_time,
                global_step=global_step
            )

        if hasattr(self, 'curr_latency_diff_time'):
            self.writer.add_scalar(
                tag='cpu_latency', 
                scalar_value=avg_cpu_latency_time,
                global_step=global_step
            )
            
class GPUInferenceCallback(base.BaseCallback):
    """
    Performance callback for tracking GPU latency and
    average execution time of one batch of image data
    NOTE:
        use this callback if you are training on GPU
        which supports CUDA. 
        If you train on CPU, then consider 'CPUInferenceCallback' class.
    """
    def on_init_start(self, **kwargs):
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def on_train_batch_start(self, **kwargs):
        self.start_event.record()

    def on_train_batch_end(self, **kwargs):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time = self.end_event.elapsed_time(other=self.start_event)
        self.gpu_measures.append(elapsed_time)
        
    def on_train_epoch_end(self, **kwargs):
        avg_gpu_time = numpy.mean(self.gpu_measures)
        global_step = kwargs.get("global_step", None)
        self.writer.add_scalar(
            tag='gpu_time', 
            scalar_value=avg_gpu_time, 
            global_step=global_step
        )

    def tearDown(self, **kwargs):
        if hasattr(self, 'writer'):
            self.writer.close()