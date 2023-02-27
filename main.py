import numpy as np
from numba import cuda

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
def mandelbrot_set(xmin, xmax, ymin, ymax, image, max_iters):
    height, width = image.shape
    pixel_size_x = (xmax - xmin) / width
    pixel_size_y = (ymax - ymin) / height
    startX, startY = cuda.grid(2)
    gridX, gridY = cuda.gridsize(2)
    for x in range(startX, width, gridX):
        real = xmin + x * pixel_size_x
        for y in range(startY, height, gridY):
            imag = ymin + y * pixel_size_y
            image[y, x] = mandelbrot(real, imag, max_iters)

if __name__ == '__main__':
    # Define the parameters of the Mandelbrot set
    xmin, xmax, ymin, ymax = -2, 2, -2, 2
    max_iters = 1000
    
    # Define the dimensions of the image to generate
    width, height = 1024, 1024
    
    # Allocate memory on the device for the image
    image = np.zeros((height, width), dtype=np.uint16)
    d_image = cuda.to_device(image)
    
    # Set up the CUDA kernel
    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # Generate the Mandelbrot set using CUDA
    mandelbrot_set[blockspergrid, threadsperblock](xmin, xmax, ymin, ymax, d_image, max_iters)
    
    # Copy the image data back to the host and save the image
    d_image.copy_to_host()
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap='gray', extent=[xmin, xmax, ymin, ymax])
    plt.axis('off')
    plt.show()
