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
fn matmul(t1: Tensor[type], t2: Tensor[type]) -> Tensor[type]:
    print(str(t1.shape()))
    print(str(t2.shape()))
    var t_mul: Tensor[type] = Tensor[type](TensorShape(t1.shape()[0],t2.shape()[1]))

    for i in range(t_mul.shape()[0]):
        for j in range(t1.shape()[1]):
            for k in range(t_mul.shape()[1]):
                t_mul[Index(i, k)] += t1[Index(i,j)] * t2[Index(j,k)]
                
    return t_mul        

fn dot(t1: Tensor[type], t2: Tensor[type]) -> Float32:
    var vec_dot: Float32 = 0.0
    var temp_vec: Tensor[type] = Tensor[type](t1.shape())

    @parameter
    fn compute_mul[simd_width: Int](idx: Int):
        var testsimd = SIMD[type, simd_width](0)
        testsimd = t1.simd_load[simd_width](idx)
        temp_vec.data().simd_store[simd_width](idx, t1.simd_load[simd_width](idx).__mul__(t2.simd_load[simd_width](idx)))
    
    vectorize[simdwidth, compute_mul](t1.shape()[1])

    print(str(temp_vec))

    @unroll(100)
    for i in range(temp_vec.shape()[1]):
        vec_dot += temp_vec[i]
    print(vec_dot)
    return vec_dot

fn main() raises:
    let height = 3
    let width = 2
    let channels = 3
    let length = 1000
    let tensor_file = Path("./tensor_test")
    print("simd width:", simdwidth)
 
    let vals = rand[DType.float32](length)
    let vals2 = rand[DType.float32](length)

    let spec = TensorSpec(DType.float32, height, width)
    let spec2 = TensorSpec(DType.float32, width, height)
    let tshape = TensorShape(2,3)
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
    tensor_mul = matmul(tensor1, tensor2)
    print(str(tensor_mul))

    print(str(vector1))
    print(str(vector2))
    _ = dot(vector1, vector2)



    # print(gray_scale_image.shape().__str__())


    # try:
    #     gray_scale_image.save(tensor_file)
    # except e:
    #     print("failed to save tensor:", e)
    # finally:
    #     print("done saving tensor")


    # try:
    #     tensor_load = tensor_load.load(tensor_file)
    # except e:
    #     print("failed to load tensor:", e)
    # finally:
    #     print("done loading tensor")

    # print(tensor_load.shape().__str__())
    # print(str(tensor_load))

    # tensor_load = tensor_load + gray_scale_image
    # var tensor_transpose = tensor_load.reshape(tshape)
    # var tensor_mul = Tensor[type](TensorShape(tensor_load.shape()[0],tensor_transpose.shape()[1]))
    # print(str(tensor_load))
    # print(str(tensor_transpose))
    # print(str(tensor_mul))
    # print("file tensor shape: ", tensor_load.shape())
    # print("tensore transpose shape: ", tensor_transpose.shape())

    # for i in range(tensor_load.shape()[0]):
    #     for j in range(tensor_transpose.shape()[1]):
    #         var tload_row = Tensor[type](tensor_load.shape[1])
    #         var ttrans_col = Tensor[type](tensor_transpose.shape[0])
    #         for k in range(tensor_load.shape()[1]):
    #             tload_row[]
    #             # var str = "t(" + str(i) + ", " + str(j) + ")"
    #             # print(str, tensor_load[i,j])


