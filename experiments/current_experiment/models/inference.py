import tensorrt as trt
import typing
from torch import nn
import torch
from pycuda.driver import cuda
import numpy


INFERENCE_TYPE = typing.TypeAlias(
    typing.Literal[
        'int',
        'fp',
        'fp8',
        'fp16',
        'fp32',
        'int8',
        'int16',
        'int32'
        'uint8',
        'uint16',
        'uint32'
    ],
)


class InferenceBuilder(object):
    """
    Builder class for optimizing model inference 
    using tensorRT acceleration library
    """

    def __init__(self,
                 network: nn.Module,
                 input_dtype: INFERENCE_TYPE,
                 output_dtype: INFERENCE_TYPE,
                 input_size: tuple,
                 output_size: tuple,
                 maximum_batch_size: int,
                 ):
        self._input_shape = input_size 
        self._output_shape = output_size
        self._logger = trt.Logger(trt.Logger.INFO)
        self._builder = trt.Builder(self._logger)
        self._network = self._builder.create_network()

        # configuring inference settings for the model
        self.configure(
            input_dtype,
            output_dtype,
            input_size,
            output_size,
            maximum_batch_size
        )

    def _configure(
        self,
        input_dtype,
        output_dtype,
        input_size,
        output_size,
        maximum_batch_size
    ):
        in_dtype = self._get_prec(input_dtype)
        out_dtype = self._get_prec(output_dtype)
        self._builder.maximum_batch_size = maximum_batch_size

        self._network.add_input(name='input_tensor',
                                shape=input_size, dtype=in_dtype)
        self._network.add_output(name='output_tensor',
                                 shape=output_size, dtype=out_dtype)

    def _get_prec(self, prec):
        match(prec):
            case "fp16":
                trt.fp16_mode = True
                return trt.float16
            case "fp32":
                trt.fp32_mode = True
                return trt.float32
            case "uint8":
                trt.uint8_mode = True
                return trt.uint8
            case "uint16":
                trt.uint16_mode = True
                return trt.uint16
            case "int16":
                trt.int16_mode = True
                return trt.int16
            case "int8":
                trt.int8_mode = True
                return trt.int8

    def _test_inference(self, test_input_tensor: torch.Tensor):
        """
        Function runs testing inference to ensure that model
        has increased it's performance afterwards
        """
        if not len(test_input_tensor):
            raise ValueError('tensor should not be empty')

        # rearraning to CHW, because it context of CUDA processing it works reasonably better

        test_input_tensor = test_input_tensor.transpose(2, 0, 1)
        self.engine = trt.create_cuda_engine(self._network)
        self.engine.profiling = True

        with self.engine.create_execution_context() as context:

            # allocating memory for the inference to be executed

            allocated_input_mem = cuda.mem_alloc(test_input_tensor.nbytes)

            cuda.memcpy_htod(
                allocated_input_mem, 
                test_input_tensor.contiguous().numpy()
            )
            # allocating memory for the output on the GPU using cuda
            allocated_output_mem = cuda.mem_alloc(self._output_shape)
            context.execute_v2(
                bindings=[
                    int(allocated_input_mem), 
                    int(allocated_output_mem)
                ]
            )
            # transfering output back to the cpu
            output = numpy.zeros(shape=self._output_shape, dtype=numpy.float32)
            cuda.memcpy_dtoh(output, allocated_output_mem)


    def get_benchmarks(self):
        """
        Function return benchmarks 
        for tensorRT accelerated inference
        """
        profile = self.engine.get_profile()
        return {
            'average_exec_time': profile.get('average_time', None),
            'max_exec_time': profile.get('max_time', None),
            'min_exec_time': profile.get('min_time', None),
            'working_memory_used': profile.get('working_memory', None),
            'device_memory_used': profile.get('device_memory', None)
        }

    def _export_optimized_network(self, model, input_example=None):
        """
        Function, which exports optimized Network
        to ONNX format
        """
        if not input_example:
            input_example = torch.randn(size=self._input_shape)

        torch.onnx.export(
            model=model,
            args=(input_example,),
            verbose=True,
            f='models/inf_deepfake_model.onnx',
            opset_version='9'
        )