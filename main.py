import numpy as np
from numba import cuda, float32

@cuda.jit(device=True)
def mandelbrot(c_real, c_imag, max_iters):
    z_real = c_real
    z_imag = c_imag
    for i in range(max_iters):
        if z_real * z_real + z_imag * z_imag > 4.0:
            return i
        temp_real = z_real * z_real - z_imag * z_imag + c_real
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = temp_real
    return max_iters

@cuda.jit
def mandelbrot_set(xmin, xmax, ymin, ymax, image):
    height, width = image.shape
    pixel_size_x = (xmax - xmin) / width
    pixel_size_y = (ymax - ymin) / height

    # Copy frequently accessed data to shared memory
    sm_xmin = cuda.shared.array(shape=1, dtype=float32)
    sm_xmax = cuda.shared.array(shape=1, dtype=float32)
    sm_ymin = cuda.shared.array(shape=1, dtype=float32)
    sm_ymax = cuda.shared.array(shape=1, dtype=float32)
    sm_pixel_size_x = cuda.shared.array(shape=1, dtype=float32)
    sm_pixel_size_y = cuda.shared.array(shape=1, dtype=float32)
    tid = cuda.threadIdx.x
    if tid == 0:
        sm_xmin[0] = xmin
        sm_xmax[0] = xmax
        sm_ymin[0] = ymin
        sm_ymax[0] = ymax
        sm_pixel_size_x[0] = pixel_size_x
        sm_pixel_size_y[0] = pixel_size_y
    cuda.syncthreads()

    # Copy constant data to constant memory
    cuda.const.array_like(max_iters)(max_iters)

    startX, startY = cuda.grid(2)
    gridX, gridY = cuda.gridsize(2)
    for x in range(startX, width, gridX):
        real = sm_x
