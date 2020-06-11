import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import sys

imageFileName = sys.argv[1]
image = io.imread(imageFileName)

# Convert the image to grayscale if it is not gray
if len(image.shape) > 2:
    image = rgb2gray(image)
    image = img_as_ubyte(image)

image_filtered_gpu = np.zeros_like(image)

# I refer to the matrix for shared memory as "tile". Although, technically it is not a matrix.
# Actually, it is a current block with boundaries of the neighboring blocks.
# It is necessary for counting values in pixels that are boundaries of the block.
# Thus, tile size is (blockDim.x + 2, blockDim.y + 2)
kernelCode = """
#include <math.h> // It is needed for usage of the 'sqrt' function.
#define m 34 // size of the tile
__constant__ int Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
__global__ void sobel_filter(
    unsigned char *input,
    unsigned char *output,
    int height,
    int width
    )
{
    extern __shared__ unsigned char shared_mat[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockDim.x;
    int by = blockDim.y;
    int x = tx + blockIdx.x * bx;
    int y = ty + blockIdx.y * by;
    if ((x < 1) || (y < 1) || (x > height - 2) || (y > width - 2))
        return;

    // fill the tile with values of the current block
    shared_mat[m * (tx + 1) + ty + 1] = input[x * width + y];

    // fill 4 corners of the tile
    if ((tx == 0) && (ty == 0))
        shared_mat[0] = input[x * width + y - width - 1];
    if ((tx == 0) && (ty == by - 1))
        shared_mat[m - 1] = input[x * width + y - width + 1];
    if ((tx == bx - 1) && (ty == 0))
        shared_mat[m * (m - 1)] = input[x * width + y + width - 1];
    if ((tx == bx - 1) && (ty == by - 1))
        shared_mat[m * m - 1] = input[x * width + y + width + 1];

    // fill boundaries of the tile except for corners
    if (tx == 0)
        shared_mat[ty + 1] = input[x * width + y - width];
    if (tx == bx - 1)
        shared_mat[m * (m - 1) + ty + 1] = input[x * width + y + width];
    if (ty == 0)
        shared_mat[m * (tx + 1)] = input[x * width + y - 1];
    if (ty == by - 1)
        shared_mat[m * (tx + 1) + m - 1] = input[x * width + y + 1];

    __syncthreads();

    float s1 = 0;
    float s2 = 0;
    for (int s = -1; s < 2; s++)
    {
        s1 += shared_mat[m * tx + (ty + 1) + s] * Gx[s + 1];
        s1 += shared_mat[m * (tx + 1) + (ty + 1) + s] * Gx[s + 4];
        s1 += shared_mat[m * (tx + 2) + (ty + 1) + s] * Gx[s + 7];
        s2 += shared_mat[m * tx + (ty + 1) + s] * Gy[s + 1];
        s2 += shared_mat[m * (tx + 1) + (ty + 1) + s] * Gy[s + 4];
        s2 += shared_mat[m * (tx + 2) + (ty + 1) + s] * Gy[s + 7];
    }

    float mag = sqrt(s1 * s1 + s2 * s2);
    output[x * width + y] = mag > 70 ? mag : 0;
}
"""

# The option "--expt-relaxed-constexpr" is needed for including the library <math.h>.
mod = SourceModule(kernelCode, options=["--expt-relaxed-constexpr"])
func = mod.get_function("sobel_filter")

blockSize = 32
block = (blockSize, blockSize, 1)
grid = (image.shape[0] // blockSize, image.shape[1] // blockSize, 1)

input_gpu = cuda.mem_alloc(image.nbytes)
cuda.memcpy_htod(input_gpu, image)
output_gpu = cuda.mem_alloc(image_filtered_gpu.nbytes)

# Size of shared memory, i.e. of the tile.
mat = np.zeros((blockSize + 2) * (blockSize + 2))
func(
    input_gpu,
    output_gpu,
    np.int32(image.shape[0]),
    np.int32(image.shape[1]),
    block=block,
    grid=grid,
    shared=mat.nbytes)
cuda.Context.synchronize()

cuda.memcpy_dtoh(image_filtered_gpu, output_gpu)
io.imsave('gpu_filtered_' + imageFileName, image_filtered_gpu)