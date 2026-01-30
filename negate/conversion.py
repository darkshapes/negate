# adapted from ML-Model-CI

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
from typing import List, Union

import numpy as np
import onnxconverter_common
from mongoengine.fields import IntField, ListField, StringField, EmbeddedDocument
from onnx import TensorProto


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
        DataType.TYPE_BOOL: onnxconverter_common.BooleanTensorType,
        DataType.TYPE_INT32: onnxconverter_common.Int32TensorType,
        DataType.TYPE_INT64: onnxconverter_common.Int64TensorType,
        DataType.TYPE_FP32: onnxconverter_common.FloatTensorType,
        DataType.TYPE_FP64: onnxconverter_common.DoubleTensorType,
        DataType.TYPE_STRING: onnxconverter_common.StringType,
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
