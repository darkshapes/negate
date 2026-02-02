# SPDX-License-Identifier: Apache-2.0
# adapted from cap-ntu/ML-Model-CI

#!/usr/bin/python3
# -*- coding: utf-8 -*-
#  Copyright (c) NTU_CAP 2021. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

import numpy as np
import onnx
import onnxconverter_common
import onnxmltools
import torch
import torch.onnx
from mongoengine.fields import EmbeddedDocument, IntField, ListField, StringField
from onnx import TensorProto
from onnxoptimizer import optimize


class ModelInputFormat(Enum):
    FORMAT_NONE = 0
    FORMAT_NHWC = 1
    FORMAT_NCHW = 2


class DataType(Enum):
    """
    @@@@.. cpp:enum:: DataType@@@@   Data types supported for input and output
    tensors.@@
    """

    # @@  .. cpp:enumerator:: DataType::INVALID = 0
    TYPE_INVALID = 0
    # @@  .. cpp:enumerator:: DataType::BOOL = 1
    TYPE_BOOL = 1
    # @@  .. cpp:enumerator:: DataType::UINT8 = 2
    TYPE_UINT8 = 2
    # @@  .. cpp:enumerator:: DataType::UINT16 = 3
    TYPE_UINT16 = 3
    # @@  .. cpp:enumerator:: DataType::UINT32 = 4
    TYPE_UINT32 = 4
    # @@  .. cpp:enumerator:: DataType::UINT64 = 5
    TYPE_UINT64 = 5
    # @@  .. cpp:enumerator:: DataType::INT8 = 6
    TYPE_INT8 = 6
    # @@  .. cpp:enumerator:: DataType::INT16 = 7
    TYPE_INT16 = 7
    # @@  .. cpp:enumerator:: DataType::INT32 = 8
    TYPE_INT32 = 8
    # @@  .. cpp:enumerator:: DataType::INT64 = 9
    TYPE_INT64 = 9
    # @@  .. cpp:enumerator:: DataType::FP16 = 10
    TYPE_FP16 = 10
    # @@  .. cpp:enumerator:: DataType::FP32 = 11
    TYPE_FP32 = 11
    # @@  .. cpp:enumerator:: DataType::FP64 = 12
    TYPE_FP64 = 12
    # @@  .. cpp:enumerator:: DataType::STRING = 13
    TYPE_STRING = 13


class IOShapeDO(EmbeddedDocument):
    name = StringField()
    shape = ListField(IntField(), required=True)
    dtype = StringField(required=True)
    format = IntField(required=True)


def type_to_data_type(tensor_type):
    import torch

    mapper = defaultdict(
        lambda: DataType.TYPE_INVALID,
        {
            bool: DataType.TYPE_BOOL,
            int: DataType.TYPE_INT32,
            float: DataType.TYPE_FP32,
            str: DataType.TYPE_STRING,
            torch.bool: DataType.TYPE_BOOL,
            torch.uint8: DataType.TYPE_UINT8,
            torch.int: DataType.TYPE_INT32,
            torch.int8: DataType.TYPE_INT8,
            torch.int16: DataType.TYPE_INT16,
            torch.int32: DataType.TYPE_INT32,
            torch.int64: DataType.TYPE_INT64,
            torch.float: DataType.TYPE_FP32,
            torch.float16: DataType.TYPE_FP16,
            torch.float32: DataType.TYPE_FP32,
            torch.float64: DataType.TYPE_FP64,
            np.dtype(np.bool): DataType.TYPE_BOOL,
            np.dtype(np.uint8): DataType.TYPE_UINT8,
            np.dtype(np.uint16): DataType.TYPE_UINT16,
            np.dtype(np.uint32): DataType.TYPE_UINT32,
            np.dtype(np.uint64): DataType.TYPE_UINT64,
            np.dtype(np.float16): DataType.TYPE_FP16,
            np.dtype(np.float32): DataType.TYPE_FP32,
            np.dtype(np.float64): DataType.TYPE_FP64,
            np.dtype(np.str_): DataType.TYPE_STRING,
            TensorProto.UNDEFINED: DataType.TYPE_INVALID,
            TensorProto.FLOAT: DataType.TYPE_FP32,
            TensorProto.UINT8: DataType.TYPE_UINT8,
            TensorProto.INT8: DataType.TYPE_INT8,
            TensorProto.UINT16: DataType.TYPE_UINT16,
            TensorProto.INT16: DataType.TYPE_INT16,
            TensorProto.INT32: DataType.TYPE_INT32,
            TensorProto.INT64: DataType.TYPE_INT64,
            TensorProto.STRING: DataType.TYPE_STRING,
            TensorProto.BOOL: DataType.TYPE_BOOL,
            TensorProto.FLOAT16: DataType.TYPE_FP16,
            TensorProto.DOUBLE: DataType.TYPE_FP64,
            TensorProto.UINT32: DataType.TYPE_UINT32,
            TensorProto.UINT64: DataType.TYPE_UINT64,
        },
    )

    return mapper[tensor_type]


def model_data_type_to_onnx(model_dtype):
    mapper = {
        DataType.TYPE_INVALID: onnxconverter_common,
        DataType.TYPE_BOOL: onnxmltools.convert.common.data_types.BooleanTensorType,
        DataType.TYPE_INT32: onnxmltools.convert.common.data_types.Int32TensorType,
        DataType.TYPE_INT64: onnxmltools.convert.common.data_types.Int64TensorType,
        DataType.TYPE_FP32: onnxmltools.convert.common.data_types.FloatTensorType,
        DataType.TYPE_FP64: onnxmltools.convert.common.data_types.DoubleTensorType,
        DataType.TYPE_STRING: onnxmltools.convert.common.data_types.StringType,
    }
    if isinstance(model_dtype, int):
        model_dtype = DataType(model_dtype)
    elif isinstance(model_dtype, str):
        model_dtype = DataType[model_dtype]
    elif not isinstance(model_dtype, DataType):
        raise TypeError(f"model_dtype is expecting one of the type: `int`, `str`, or `DataType` but got {type(model_dtype)}")
    return mapper[model_dtype]


def model_data_type_to_np(model_dtype):
    mapper = {
        DataType.TYPE_INVALID: None,
        DataType.TYPE_BOOL: np.bool,
        DataType.TYPE_UINT8: np.uint8,
        DataType.TYPE_UINT16: np.uint16,
        DataType.TYPE_UINT32: np.uint32,
        DataType.TYPE_UINT64: np.uint64,
        DataType.TYPE_INT8: np.int8,
        DataType.TYPE_INT16: np.int16,
        DataType.TYPE_INT32: np.int32,
        DataType.TYPE_INT64: np.int64,
        DataType.TYPE_FP16: np.float16,
        DataType.TYPE_FP32: np.float32,
        DataType.TYPE_FP64: np.float64,
        DataType.TYPE_STRING: np.dtype(object),
    }

    if isinstance(model_dtype, int):
        model_dtype = DataType(model_dtype)
    elif isinstance(model_dtype, str):
        model_dtype = DataType[model_dtype]
    elif not isinstance(model_dtype, DataType):
        raise TypeError(f"model_dtype is expecting one of the type: `int`, `str`, or `DataType` but got {type(model_dtype)}")
    return mapper[model_dtype]


def model_data_type_to_torch(model_dtype):
    import torch

    mapper = {
        DataType.TYPE_INVALID: None,
        DataType.TYPE_BOOL: torch.bool,
        DataType.TYPE_UINT8: torch.uint8,
        DataType.TYPE_INT8: torch.int8,
        DataType.TYPE_INT16: torch.int16,
        DataType.TYPE_INT32: torch.int32,
        DataType.TYPE_INT64: torch.int64,
        DataType.TYPE_FP16: torch.float16,
        DataType.TYPE_FP32: torch.float32,
        DataType.TYPE_FP64: torch.float64,
    }

    if isinstance(model_dtype, int):
        model_dtype = DataType(model_dtype)
    elif isinstance(model_dtype, str):
        model_dtype = DataType[model_dtype]
    elif not isinstance(model_dtype, DataType):
        raise TypeError(f"model_dtype is expecting one of the type: `int`, `str`, or `DataType` but got {type(model_dtype)}")
    return mapper[model_dtype]


class IOShape:
    """Class for recording input and output shape with their data type.

    Args:
        shape (List[int]): the shape of the input or output tensor.
        dtype (DataType, type, str): The data type of the input or output tensor.
        name (str): Tensor name. Default to None.
        format (ModelInputFormat): Input format, used for TensorRT currently.
            Default to `ModelInputFormat.FORMAT_NONE`.
    """

    def __init__(self, shape: List[int], dtype: Union[type, int, str, DataType], name: str | None = None, format: ModelInputFormat = ModelInputFormat.FORMAT_NONE):
        """Initializer of input/output shape."""

        import numpy as np
        import torch

        # input / output name
        self.name = name
        # input / output tensor shape
        self.shape = shape
        # input format
        if isinstance(format, str):
            # TODO: for the convenience of `model.hub.manager`, to be removed
            format = ModelInputFormat[format.upper()]
        self.format = format
        if isinstance(dtype, str):
            try:
                # if the type name is unified python type
                dtype = type_to_data_type(eval(dtype))
            except NameError:
                # try if the dtype is `DataType`
                dtype = DataType[dtype.upper()]
        elif isinstance(dtype, (type, int)):
            dtype = type_to_data_type(dtype)
        elif isinstance(dtype, (torch.dtype, np.dtype)):
            dtype = type_to_data_type(dtype)
        elif isinstance(dtype, DataType):
            pass
        else:
            raise ValueError(f"data type should be an instance of `type`, type name or `DataType`, but got {type(dtype)}")

        # warning if the dtype is DataType.TYPE_INVALID
        if dtype == DataType.TYPE_INVALID:
            print("W: `dtype` is converted to invalid.")

        # input / output datatype
        self.dtype = dtype

    @property
    def batch_size(self) -> int:
        return self.shape[0]

    @property
    def example_shape(self):
        return self.shape[1:]

    @property
    def height(self):
        if self.format == ModelInputFormat.FORMAT_NONE:
            raise ValueError("No height for shape format of `ModelInputFormat.FORMAT_NONE`.")
        if self.format == ModelInputFormat.FORMAT_NCHW:
            return self.shape[2]
        if self.format == ModelInputFormat.FORMAT_NHWC:
            return self.shape[1]

    @property
    def width(self):
        if self.format == ModelInputFormat.FORMAT_NONE:
            raise ValueError("No width for shape format of `ModelInputFormat.FORMAT_NONE`.")
        if self.format == ModelInputFormat.FORMAT_NCHW:
            return self.shape[3]
        if self.format == ModelInputFormat.FORMAT_NHWC:
            return self.shape[2]

    @property
    def channel(self):
        if self.format == ModelInputFormat.FORMAT_NONE:
            raise ValueError("No channel for shape format of `ModelInputFormat.FORMAT_NONE`.")
        if self.format == ModelInputFormat.FORMAT_NCHW:
            return self.shape[1]
        if self.format == ModelInputFormat.FORMAT_NHWC:
            return self.shape[3]

    def to_io_shape_po(self):
        """Convert IO shape business object to IO shape plain object."""
        return IOShapeDO(name=self.name, shape=self.shape, dtype=self.dtype.name, format=self.format)

    @staticmethod
    def from_io_shape_po(io_shape_po: IOShapeDO):
        """Create IO shape business object from IO shape plain object."""
        io_shape_bo = IOShape(name=io_shape_po.name, shape=io_shape_po.shape, dtype=io_shape_po.dtype, format=ModelInputFormat(io_shape_po.format))

        return io_shape_bo

    def __str__(self):
        return "{}, dtype={}, format={}".format(self.shape, self.dtype, self.format.name)


class ONNXConverter(object):
    """Convert model to ONNX format."""

    DEFAULT_OPSET = 10
    supported_framework = ["pytorch", "sklearn", "xgboost", "lightgbm"]

    class _Wrapper(object):
        @staticmethod
        def save(converter: Callable[..., "onnx.ModelProto"]):
            def wrap(*args, save_path: Path | None = None, optimize: bool = True, override: bool = False, **kwargs) -> "onnx.ModelProto":
                onnx_model = None
                save_path_with_ext = None

                if save_path is not None:
                    save_path = Path(save_path)
                    save_path_with_ext = save_path.with_suffix(".onnx")
                    if save_path_with_ext.exists() and not override:
                        # file exist yet override flag is not set
                        print("Use cached model")
                        onnx_model = onnx.load(str(save_path_with_ext))

                if onnx_model is None:
                    # otherwise, convert model
                    onnx_model = converter(*args, **kwargs)

                if optimize:
                    # optimize ONNX model
                    onnx_model = ONNXConverter.optim_onnx(onnx_model)

                if save_path_with_ext:
                    # save to disk
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    onnxmltools.utils.save_model(onnx_model, save_path_with_ext)

                return onnx_model

            return wrap

    @staticmethod
    def from_pytorch(
        model: torch.nn.Module,
        save_path: Path,
        inputs: Iterable[IOShape],
        outputs: Iterable[IOShape],
        model_input: Optional[List] = None,
        opset: int = 10,
        optimize: bool = True,
        override: bool = False,
    ):
        """Save a loaded model in ONNX.
            TODO: reuse inputs to pass model_input parameter later

        Arguments:
            model (nn.Module): PyTorch model.
            save_path (Path): ONNX saved path.
            inputs (Iterable[IOShape]): Model input shapes. Batch size is indicated at the dimension.
            outputs (Iterable[IOShape]): Model output shapes.
            model_input (Optional[List]) : Sample Model input data
            opset (int): ONNX op set version.
            optimize (bool): Flag to optimize ONNX network. Default to `True`.
            override (bool): Flag to override if the file with path to `save_path` has existed. Default to `False`.
        """
        if save_path.with_suffix(".onnx").exists():
            if not override:  # file exist yet override flag is not set
                print("Use cached model")
                return True

        export_kwargs = dict()

        # assert batch size
        batch_sizes = list(map(lambda x: x.shape[0], inputs))
        if not all(batch_size == batch_sizes[0] for batch_size in batch_sizes):
            raise ValueError("batch size for inputs (i.e. the first dimensions of `input.shape` are not consistent.")
        batch_size = batch_sizes[0]

        if batch_size == -1:
            export_kwargs["dynamic_axes"] = {
                "input": {0: "batch_size"},  # variable length axes
                "output": {0: "batch_size"},
            }
            batch_size = 1
        else:
            assert batch_size > 0

        model.eval()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path_with_ext = save_path.with_suffix(".onnx")

        dummy_tensors, input_names, output_names = list(), list(), list()
        for input_ in inputs:
            dtype = model_data_type_to_torch(input_.dtype)
            dummy_tensors.append(torch.rand(batch_size, *input_.shape[1:], requires_grad=True, dtype=dtype))
            input_names.append(input_.name)
        for output_ in outputs:
            output_names.append(output_.name)
        if model_input is None:
            model_input = tuple(dummy_tensors)
        try:
            torch.onnx.export(
                model,  # model being run
                model_input,  # model input (or a tuple for multiple inputs)
                save_path_with_ext,  # where to save the model (can be a file or file-like object)
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=opset,  # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=input_names,  # the model's input names
                output_names=output_names,  # the model's output names
                keep_initializers_as_inputs=True,
                **export_kwargs,
            )

            if optimize:
                onnx_model = onnx.load(str(save_path_with_ext))
                network = ONNXConverter.optim_onnx(onnx_model)
                onnx.save(network, str(save_path_with_ext))

            print("ONNX format converted successfully")
            return True
        except Exception as e:
            # TODO catch different types of error
            print("Unable to convert to ONNX format, reason:")
            print(e)
            return False

    @staticmethod
    @_Wrapper.save
    def from_sklearn(
        model,
        inputs: Iterable[IOShape],
        opset: int = DEFAULT_OPSET,
    ):
        initial_type = ONNXConverter.convert_initial_type(inputs)
        onnx_model = onnxmltools.convert_sklearn(model, initial_types=initial_type, target_opset=opset)
        print("sklearn to onnx converted successfully")
        return onnx_model

    @staticmethod
    @_Wrapper.save
    def from_xgboost(model, inputs: Iterable[IOShape], opset: int = DEFAULT_OPSET):
        initial_type = ONNXConverter.convert_initial_type(inputs)
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type, target_opset=opset)
        print("xgboost to onnx converted successfully")
        return onnx_model

    @staticmethod
    @_Wrapper.save
    def from_lightgbm(model, inputs: Iterable[IOShape], opset: int = DEFAULT_OPSET):
        initial_type = ONNXConverter.convert_initial_type(inputs)
        onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type, target_opset=opset)
        print("lightgbm to onnx converted successfully")
        return onnx_model

    @staticmethod
    def convert_initial_type(inputs: Iterable[IOShape]):
        # assert batch size
        batch_sizes = list(map(lambda x: x.shape[0], inputs))
        if not all(batch_size == batch_sizes[0] for batch_size in batch_sizes):
            raise ValueError("batch size for inputs (i.e. the first dimensions of `input.shape` are not consistent.")
        batch_size = batch_sizes[0]

        if batch_size == -1:
            batch_size = None
        else:
            assert batch_size > 0

        initial_type = list()
        for input_ in inputs:
            initial_type.append((input_.name, model_data_type_to_onnx(input_.dtype)([batch_size, *input_.shape[1:]])))
        return initial_type

    @staticmethod
    def optim_onnx(model: onnx.ModelProto, verbose=False):
        """Optimize ONNX network"""
        print("Begin Simplify ONNX Model ...")
        passes = [
            "eliminate_deadend",
            "eliminate_identity",
            "extract_constant_to_initializer",
            "eliminate_unused_initializer",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_matmul_add_bias_into_gemm",
        ]
        model = optimize(model, passes=passes)

        print(f"Finished optimizing ONNX model. Total nodes: {len(model.graph.node)}")
        return model
