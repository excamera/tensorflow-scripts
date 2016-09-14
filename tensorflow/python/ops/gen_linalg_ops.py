"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def batch_cholesky(input, name=None):
  r"""Calculates the Cholesky decomposition of a batch of square matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices, with the same constraints as the single matrix Cholesky
  decomposition above. The output is a tensor of the same shape as the input
  containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[..., M, M]`.
  """
  return _op_def_lib.apply_op("BatchCholesky", input=input, name=name)


def batch_matrix_determinant(input, name=None):
  r"""Calculates the determinants for a batch of square matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a 1-D tensor containing the determinants
  for all input submatrices `[..., :, :]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[...]`.
  """
  return _op_def_lib.apply_op("BatchMatrixDeterminant", input=input,
                              name=name)


def batch_matrix_inverse(input, name=None):
  r"""Calculates the inverse of square invertible matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a tensor of the same shape as the input
  containing the inverse for all input submatrices `[..., :, :]`.

  The op uses the Cholesky decomposition if the matrices are symmetric positive
  definite and LU decomposition with partial pivoting otherwise.

  If a matrix is not invertible there is no guarantee what the op does. It
  may detect the condition and raise an exception or it may simply return a
  garbage result.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[..., M, M]`.
  """
  return _op_def_lib.apply_op("BatchMatrixInverse", input=input, name=name)


def batch_self_adjoint_eig(input, name=None):
  r"""Calculates the Eigen Decomposition of a batch of square self-adjoint matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices, with the same constraints as the single matrix
  SelfAdjointEig.

  The result is a '[..., M+1, M] matrix with [..., 0,:] containing the
  eigenvalues, and subsequent [...,1:, :] containing the eigenvectors.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[..., M+1, M]`.
  """
  return _op_def_lib.apply_op("BatchSelfAdjointEig", input=input, name=name)


def cholesky(input, name=None):
  r"""Calculates the Cholesky decomposition of a square matrix.

  The input has to be symmetric and positive definite. Only the lower-triangular
  part of the input will be used for this operation. The upper-triangular part
  will not be read.

  The result is the lower-triangular matrix of the Cholesky decomposition of the
  input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[M, M]`.
  """
  return _op_def_lib.apply_op("Cholesky", input=input, name=name)


def matrix_determinant(input, name=None):
  r"""Calculates the determinant of a square matrix.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A tensor of shape `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    A scalar, equal to the determinant of the input.
  """
  return _op_def_lib.apply_op("MatrixDeterminant", input=input, name=name)


def matrix_inverse(input, name=None):
  r"""Calculates the inverse of a square invertible matrix.

  The op uses the Cholesky decomposition if the matrix is symmetric positive
  definite and LU decomposition with partial pivoting otherwise.

  If the matrix is not invertible there is no guarantee what the op does. It
  may detect the condition and raise an exception or it may simply return a
  garbage result.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Shape is `[M, M]` containing the matrix inverse of the input.
  """
  return _op_def_lib.apply_op("MatrixInverse", input=input, name=name)


def self_adjoint_eig(input, name=None):
  r"""Calculates the Eigen Decomposition of a square Self-Adjoint matrix.

  Only the lower-triangular part of the input will be used in this case. The
  upper-triangular part will not be read.

  The result is a M+1 x M matrix whose first row is the eigenvalues, and
  subsequent rows are eigenvectors.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[M+1, M]`.
  """
  return _op_def_lib.apply_op("SelfAdjointEig", input=input, name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "BatchCholesky"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
}
op {
  name: "BatchMatrixDeterminant"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "BatchMatrixInverse"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "BatchSelfAdjointEig"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
}
op {
  name: "Cholesky"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
}
op {
  name: "MatrixDeterminant"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "MatrixInverse"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "SelfAdjointEig"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
