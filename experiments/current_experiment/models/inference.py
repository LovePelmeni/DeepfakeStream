import tensorrt as trt
import typing

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
                 input_dtype: INFERENCE_TYPE,
                 output_dtype: INFERENCE_TYPE,
                 input_size: tuple,
                 output_size: tuple,
                 maximum_batch_size: int,
                 ):
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

    def _test_inference(self):
        pass

    def export_network(self,):
        pass
