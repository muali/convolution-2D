#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <CL/cl.h>
#include "cl.hpp"

#include <cassert>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

struct data
{
    std::vector<float> matrix;
    std::vector<float> mask;
    int matrix_size;
    int mask_size;
};

data read_data();
void write_result(int matrix_size, std::vector<float> result);
void cpu_convolution_2d(float* matrix, float* mask, int matrix_size, int mask_size, float* result);

void test(data input, std::vector<float> result)
{
    const float epsilon = 1e-4;
    std::vector<float> cpu_result(input.matrix_size * input.matrix_size);
    cpu_convolution_2d(&input.matrix[0], &input.mask[0], input.matrix_size, input.mask_size, &cpu_result[0]);
    for (int i = 0; i < cpu_result.size(); ++i)
    {
        assert(abs(cpu_result[i] - result[i]) < epsilon);
    }
}

void generate(int matrix_size, int mask_size)
{
    std::ofstream out("input.txt");
    out << matrix_size << ' ' << mask_size << std::endl;
    for (int i = 0; i < matrix_size; ++i)
    {
        for (int j = 0; j < matrix_size; ++j)
        {
            out << 1 << (j != matrix_size - 1 ? ' ' : '\n');
        }
    }
    for (int i = 0; i < mask_size; ++i)
    {
        for (int j = 0; j < mask_size; ++j)
        {
            out << 1 << (j != mask_size - 1 ? ' ' : '\n');
        }
    }
}

int main()
{
    //generate(1024, 3);
    //generate(1024, 9);
    //generate(1, 9);
    //generate(31, 9);
    //generate(1023, 9);

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("convolution2d.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
            cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(devices, "-D BLOCK_SIZE=16");

        data input = read_data();
        size_t matrix_mem_size = sizeof(float) * input.matrix_size * input.matrix_size;
        size_t mask_mem_size = sizeof(float) * input.mask_size * input.mask_size;
        size_t block_size = min(input.matrix_size, 16U);

        size_t global_size = input.matrix_size / block_size;
        if (input.matrix_size % block_size)
            global_size++;
        global_size *= block_size;

        // allocate device buffer to hold message
        cl::Buffer dev_matrix(context, CL_MEM_READ_ONLY , matrix_mem_size);
        cl::Buffer dev_mask  (context, CL_MEM_READ_ONLY , mask_mem_size);
        cl::Buffer dev_result(context, CL_MEM_WRITE_ONLY, matrix_mem_size);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_matrix, CL_TRUE, 0, matrix_mem_size, &input.matrix[0]);
        queue.enqueueWriteBuffer(dev_mask,   CL_TRUE, 0, mask_mem_size, &input.mask[0]);

        // load named kernel from opencl source
        cl::Kernel kernel(program, "convolution_2d_gpu_global");
        cl::KernelFunctor convolution_global(kernel, queue, cl::NullRange, cl::NDRange(global_size, global_size), 
            cl::NDRange(block_size, block_size));
        convolution_global(dev_matrix, dev_mask, input.matrix_size, input.mask_size, dev_result);

        //cl::Kernel kernel_shared(program, "matrix_mult_shared");
        //cl::KernelFunctor matrix_mult_shared(kernel_shared, queue, cl::NullRange,
        //    cl::NDRange(N, N), cl::NDRange(block_size, block_size));
        ////matrix_mult_shared(dev_a, dev_b, dev_c, (int)N);

        std::vector<float> result(input.matrix_size * input.matrix_size);
        queue.enqueueReadBuffer(dev_result, CL_TRUE, 0, matrix_mem_size, &result[0]);

        write_result(input.matrix_size, result);

        //test(input, result);
    }
    catch (cl::Error e)
    {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}

data read_data()
{
    std::ifstream in("input.txt");

    int matrix_size, mask_size;
    in >> matrix_size >> mask_size;
    std::vector<float> matrix(matrix_size * matrix_size);
    std::vector<float> mask(mask_size * mask_size);
    for (int i = 0; i < matrix_size * matrix_size; ++i)
        in >> matrix[i];
    for (int i = 0; i < mask_size * mask_size; ++i)
        in >> mask[i];
    
    data result;
    result.matrix      = matrix;
    result.mask        = mask;
    result.matrix_size = matrix_size;
    result.mask_size   = mask_size;
    return result;
}

void write_result(int matrix_size, std::vector<float> result)
{
    std::ofstream out("output.txt");

    out.setf(std::ios::fixed);
    out.precision(3);
    for (int i = 0; i < matrix_size; ++i)
    {
        for (int j = 0; j < matrix_size; ++j)
        {
            out << result[matrix_size * i + j] << (j != matrix_size - 1 ? ' ' : '\n');
        }
    }
}

void cpu_convolution_2d(float* matrix, float* mask, int x, int y, int matrix_size, int mask_size, float* result)
{
    result[matrix_size * x + y] = 0;
    for (int i = 0; i < mask_size; ++i)
    {
        for (int j = 0; j < mask_size; ++j)
        {
            int cx = x + i - mask_size / 2;
            int cy = y + j - mask_size / 2;
            if (cx >= 0 && cx < matrix_size &&
                cy >= 0 && cy < matrix_size)
                result[matrix_size * x + y] += matrix[matrix_size * cx + cy] * mask[mask_size * i + j];
        }
    }
}

void cpu_convolution_2d(float* matrix, float* mask, int matrix_size, int mask_size, float* result)
{
    for (int i = 0; i < matrix_size; ++i)
    {
        for (int j = 0; j < matrix_size; ++j)
        {
            cpu_convolution_2d(matrix, mask, i, j, matrix_size, mask_size, result);
        }
    }
}
