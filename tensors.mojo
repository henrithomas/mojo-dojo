from tensor import Tensor, TensorSpec, TensorShape
from algorithm import parallelize, vectorize
from utils.index import Index
from random import rand
from pathlib import path
alias type = DType.float32
alias simdwidth = simdwidthof[type]()
"""
def matmul_python(C, A, B):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]
"""
# using mojo docs tutorial: https://docs.modular.com/mojo/notebooks/Matmul.html#python-implementation

fn matmul_0(t1: Tensor[type], t2: Tensor[type]) -> Tensor[type]:
    print(str(t1.shape()))
    print(str(t2.shape()))
    var t_mul: Tensor[type] = Tensor[type](TensorShape(t1.shape()[0],t2.shape()[1]))

    for i in range(t_mul.shape()[0]):
        for j in range(t1.shape()[1]):
            for k in range(t_mul.shape()[1]):
                t_mul[Index(i, k)] += t1[Index(i,j)] * t2[Index(j,k)]
                
    return t_mul        

fn matmul_vectorized(t1: Tensor[type], t2: Tensor[type]) -> Tensor[type]:
    var t_mul: Tensor[type] = Tensor[type](TensorShape(t1.shape()[0],t2.shape()[1]))

    for i in range(t_mul.shape()[0]):
        for j in range(t1.shape()[1]):
            @parameter
            fn dot[simd_width: Int](k: Int):
                t_mul.simd_store[simd_width](
                    i * t_mul.shape()[1] + k, 
                    t_mul.simd_load[simd_width](i * t_mul.shape()[1] + k) + t1[Index(i,j)] * t2.simd_load[simd_width](j * t_mul.shape()[1] + k)
                )
            vectorize[dot, simdwidth](t_mul.shape()[1])

    return t_mul 

fn matmul_parallelized(t1: Tensor[type], t2: Tensor[type]) -> Tensor[type]:
    var t_mul: Tensor[type] = Tensor[type](TensorShape(t1.shape()[0],t2.shape()[1]))

    @parameter
    fn calc_row(i: Int):
        for j in range(t1.shape()[1]):
            @parameter
            fn dot[simd_width: Int](k: Int):
                t_mul.simd_store[simd_width](
                    i * t_mul.shape()[1] + k, 
                    t_mul.simd_load[simd_width](i * t_mul.shape()[1] + k) + t1[Index(i,j)] * t2.simd_load[simd_width](j * t_mul.shape()[1] + k)
                )
            vectorize[dot, simdwidth](t_mul.shape()[1])

    parallelize[calc_row](t_mul.shape()[0], t_mul.shape()[0])

    return t_mul 


fn dot(t1: Tensor[type], t2: Tensor[type]) -> Float32:
    var vec_dot: Float32 = 0.0
    var temp_vec: Tensor[type] = Tensor[type](t1.shape())
    var sum_vec: Tensor[type] = Tensor[type](simdwidth)
    var sum_simd = SIMD[type, simdwidth](0.0)

    @parameter
    fn compute_mul[simd_width: Int](idx: Int):
        temp_vec.simd_store[simd_width](idx, 
            t1.simd_load[simd_width](idx) * t2.simd_load[simd_width](idx))

    vectorize[compute_mul, simdwidth](t1.shape()[1])

    for i in range(temp_vec.shape()[1]):
        vec_dot += temp_vec[i]

    return vec_dot

fn main() raises:
    var height = 150
    var width = 125
    var channels = 3
    var length = 100
    var tensor_file = path.Path("./tensor_test")
    print("simd width:", simdwidth)
 
    var vals = rand[DType.float32](length)
    var vals2 = rand[DType.float32](length)

    var spec = TensorSpec(DType.float32, height, width)
    var spec2 = TensorSpec(DType.float32, width, height)
    var tshape = TensorShape(2,3)
    var tensor1 = Tensor[DType.float32](spec)
    var tensor2 = Tensor[DType.float32](spec2)
    var tensor_mul = Tensor[DType.float32](TensorSpec(DType.float32, height, height))
    var vector1 = Tensor[DType.float32](TensorSpec(DType.float32, 1, length))
    var vector2 = Tensor[DType.float32](TensorSpec(DType.float32, 1, length))

    for i in range(length):
        vector1[i] = vals[i]
        vector2[i] = vals2[i]

    vector1 = vector1.__radd__(1)
    vector2 = vector2.__radd__(1)

    for y in range(height):
        for x in range(width):
            tensor1[Index(y,x)] = 1
            tensor2[Index(x,y)] = 2

    tensor1[Index(0,1)] = 2
    tensor1[Index(1,1)] = 7
    tensor1[Index(2,1)] = 5

    tensor2[Index(1,1)] = 4
    tensor2[Index(0,1)] = 3
    tensor2[Index(1,2)] = 5

    print(str(tensor1))
    print(str(tensor2))
    tensor_mul = matmul_parallelized(tensor1, tensor2)
    print(str(tensor_mul))

    print(str(vector1))
    print(str(vector2))
    var vecdot = dot(vector1, vector2)
    print(vecdot)