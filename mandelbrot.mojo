import benchmark
from complex import ComplexSIMD, ComplexFloat64
from math import iota
from python import Python
from sys.info import num_physical_cores, num_performance_cores, num_logical_cores
from algorithm import parallelize, vectorize
from tensor import Tensor
from utils.index import Index 

alias float_type = DType.float64
alias simd_width = 2 * simdwidthof[float_type]()
alias width = 1440
alias height = 1440
alias MAX_ITERS = 200
alias min_x = -2.0
alias max_x = 0.6
alias min_y = -1.5
alias max_y = 1.5

fn info():
    print("physical cores: " + str(num_physical_cores()))
    print("performance cores: " + str(num_performance_cores()))
    print("logical cores: " + str(num_logical_cores()))
    print("simd width: " + str(simd_width))

fn mandelbrot_kernel(borrowed c: ComplexFloat64) -> Int:
    var z = c
    for i in range(MAX_ITERS):
        z = z * z + c
        if z.squared_norm() > 4:
            return i

    return MAX_ITERS

fn mandelbrot_kernel_SIMD[simd_width: Int](c: ComplexSIMD[float_type, simd_width]) -> SIMD[float_type, simd_width]:
    let cx = c.re
    let cy = c.im

    var x = SIMD[float_type, simd_width] (0)
    var y = SIMD[float_type, simd_width] (0)
    var y2 = SIMD[float_type, simd_width] (0)
    var iters = SIMD[float_type, simd_width] (0)
    var t: SIMD[DType.bool, simd_width] = True 

    for i in range(MAX_ITERS):
        if not t.reduce_or():
            break
        y2 = y*y
        y = x.fma(y + y, cy)
        t = x.fma(x, y2) <= 4
        x = x.fma(x, cx - y2)
        iters = t.select(iters + 1, iters)

    return iters

fn compute_mandelbrot() -> Tensor[float_type]:
    var t = Tensor[float_type] (height, width)

    let dx = (max_x - min_x) / width
    let dy = (max_y - min_y) / height

    var y = min_y
    for row in range(height):
        var x = min_x
        for col in range(width):
            t[Index(row, col)] = mandelbrot_kernel(ComplexFloat64(x, y))
            x += dx
        y += dy

    return t

def show_plot(tensor: Tensor[float_type]):
    alias scale = 10
    alias dpi = 64

    np = Python.import_module("numpy")
    matplotlib = Python.import_module("matplotlib")
    plt = Python.import_module("matplotlib.pyplot")
    colors = Python.import_module("matplotlib.colors")
    matplotlib.use('TkAgg')
    numpy_array = np.zeros((height, width), np.float64)

    for row in range(height):
        for col in range(width):
            numpy_array.itemset((col, row), tensor[col, row])

    fig = plt.figure(1, [scale, scale * height // width], dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], False, 1)
    light = colors.LightSource(315, 10, 0, 1, 1, 0)

    image = light.shade(numpy_array, plt.cm.hot, colors.PowerNorm(0.3), "hsv", 0, 0, .75)
    plt.imshow(image)
    plt.axis("off")
    # plt.savefig("mandelbrot.png")
    plt.show()

fn compute_mandelbrot_vectorized() -> Tensor[float_type]:
    let t = Tensor[float_type] (height, width)

    @parameter
    fn worker(row: Int):
        let scale_x = (max_x - min_x) / width
        let scale_y = (max_y - min_y) / height

        @parameter
        fn compute_vector[simd_width: Int] (col: Int):
            let cx = min_x + (col + iota[float_type, simd_width] ()) * scale_x
            let cy = min_y + row * scale_y
            let c = ComplexSIMD[float_type, simd_width] (cx, cy)

            t.data().simd_store[simd_width](
                row * width + col, mandelbrot_kernel_SIMD[simd_width] (c)
            )

        vectorize[simd_width, compute_vector](width)

    @parameter
    fn bench[simd_width: Int]():
        for row in range(height):
            worker(row)

    @parameter
    fn bench_parallel[simd_width: Int] ():
        parallelize[worker] (height, height)

    bench_parallel[simd_width]()

    # let vectorized = benchmark.run[bench[simd_width]](
    #     max_iters=1000,
    #     max_runtime_secs=0.5
    # ).mean(benchmark.Unit.ms)

    # let parallelized = benchmark.run[bench_parallel[simd_width]](
    #     max_iters=1000,
    #     max_runtime_secs=0.5
    # ).mean(benchmark.Unit.ms)

    # print("Vectorized time", ":", vectorized, "ms")
    # print("Parallelized time", ":", parallelized, "ms")
    return t

fn main():
    info()
    
    try:
        _ = show_plot(compute_mandelbrot_vectorized())
    except e:
        print("compute and plot failed: ", e)
