# Mandelbrot Set Generator using CUDA

This program generates an image of the Mandelbrot set using NVIDIA's CUDA parallel computing platform. The Mandelbrot set is a fractal that is generated by iterating a simple mathematical formula for each point in the complex plane. The color of each pixel in the image corresponds to the number of iterations required to determine whether the point belongs to the set.

## Requirements

- Python 3.6 or higher
- NumPy
- Numba
- CUDA-enabled GPU with compute capability 2.0 or higher

## Usage

To run the program, simply execute the `mandelbrot.py` file. The program will generate an image of the Mandelbrot set with the default parameters and save it to the current directory as `mandelbrot.png`.

You can also modify the parameters of the Mandelbrot set by modifying the constants at the top of the `mandelbrot.py` file:

- `WIDTH` and `HEIGHT`: The width and height of the image, in pixels.
- `XMIN`, `XMAX`, `YMIN`, and `YMAX`: The coordinates of the rectangular region in the complex plane to be plotted.
- `MAX_ITERS`: The maximum number of iterations to be performed for each point.

## Benefits of CUDA

CUDA allows us to harness the power of a GPU to perform massively parallel computations. This can result in significant performance gains over traditional CPU-based computations. In the case of this program, the Mandelbrot set is generated by computing the color of each pixel independently of all other pixels. By utilizing the thousands of processing cores available on a modern GPU, we can generate the Mandelbrot set much more quickly than with a traditional CPU-based implementation.

CUDA also provides low-level control over the GPU, allowing us to optimize our code for the specific hardware we are targeting. For example, we can make use of shared memory and constant memory to reduce the number of memory accesses and improve performance. Additionally, CUDA provides tools for profiling and debugging GPU code, making it easier to identify and optimize performance bottlenecks.

## Additional Notes
- Numba only supports CUDA Toolkit 11 as of 2/26/23.
- If using Windows, ensure CUDA_PATH is set under: `rundll32 sysdm.cpl,EditEnvironmentVariables`
