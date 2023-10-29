import tensorrt as trt
import typing
from torch import nn
import torch
from pycuda.driver import cuda
import numpy
import logging

logger = logging.getLogger(__name__)
handler = logging.FileHandler("inference.log")
logger.addHandler(logger)


INFERENCE_TYPE = typing.Literal[
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
    ]

def load_engine(engine_path):
    """
    Function loads engine from the specified file path
    location
    """
    with open(engine_path, mode='rb') as engine:
        engine_content = engine.read()
    runtime = trt.Runtime(cuda.Device(0).make_context())
    loaded_engine = runtime.deserialize_cuda_engine(engine_content)
    return loaded_engine

class HostMemoryBinding(object):
    """
    Class represents binding unit
    for a given image object for executing in parallel
    """
    def __init__(self, input_device, out_device, img: numpy.ndarray):
        self.input_device = input_device 
        self.out_device = out_device 
        self.img = img 

def run_inference(
    images: typing.List[torch.Tensor],
    engine,
    input_dtype,
    output_dtype,
    profiler=None
):
        """
        Function runs testing inference to ensure that model
        has increased it's performance afterwards
        """

        if not len(input_data):
            raise ValueError('tensor should not be empty')

        # rearraning to CHW, because it context of CUDA processing it works reasonably better
        input_data = [
            img.permute(2, 0, 1) for img in input_data
        ]

        with engine.create_execution_context() as context:

            if profiler is not None:
                context.set_profiler(profiler)


            input_bindings = []
            output_bindings = []
            host_bindings = []

            for img in images:
                try:
                    host_input = numpy.asarray(img, dtype=input_dtype)
                    host_output = 
                    gpu_input = cuda.mem_alloc(host_input.nbytes)
                    gpu_output = cuda.mem_alloc(host_output.nbytes)
                    obj = HostMemoryBinding(
                        host_output=host_output, 
                        gpu_input=int(gpu_input), 
                        gpu_output=int(gpu_output)
                    )
                    input_bindings.append(int(gpu_input))
                    output_bindings.append(int(gpu_output))
                    host_bindings.append(obj)

                except(cuda.OutOfMemoryError) as mem_err:
                    raise RuntimeError("not enough memory to allocate for img")

            for obj in host_bindings:
                cuda.mempy_htod(obj.img, obj.gpu_input)

            context.execute(
                bindings=input_bindings.extend(output_bindings)
            )

            # initializing array of predictions 

            predictions = []

            for obj in host_bindings:
                cuda.memcpy_dtoh(obj.host_output, obj.gpu_output)
                prediction = torch.from_numpy(obj.host_output).softmax(dim=0)
                predictions.append(prediction)
            return predictions 
class InferenceBuilder(object):
    """
    Builder class for optimizing model inference 
    using tensorRT acceleration library
    """

    def __init__(self,
                 model_path: str,
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
        self.profiler = trt.Profiler()

        parser = trt.OnnxParser(self._network)
        try:
            with open(model_path, mode='rb') as model_file:
                parser.parse(model_file.read())
        except(Exception) as err:
            logger.debug(err)
            raise RuntimeError('failed to load model, invalid path')
        
        self.engine = trt.create_cuda_engine(self._network)
        self.engine.profiling = True

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

        self.input_dtype = self._get_prec(input_dtype)
        self.output_dtype = self._get_prec(output_dtype)
        self._builder.maximum_batch_size = maximum_batch_size

        self._network.add_input(name='input_tensor',
                                shape=input_size, dtype=self.input_dtype)
        self._network.add_output(name='output_tensor',
                                 shape=output_size, dtype=self.output_dtype)

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

    def run_test_inference(self, input_data):
        predictions = run_inference(
            engine=self.engine, 
            input_dtype=self.input_dtype, 
            output=self.output_dtype,
            input_data=input_data
        )
        return predictions

    def _export_optimized_network(self, input_example=None):
        """
        Function, which exports optimized Network
        to ONNX format
        """
        if not input_example:
            input_example = torch.randn(
                size=tuple([1] + [param for param in self._input_shape])
            )
        with open("./models/optimized_model.trt", mode='wb') as model_file:
            model_file.write(self.engine.serialize())
        model_file.close()
        