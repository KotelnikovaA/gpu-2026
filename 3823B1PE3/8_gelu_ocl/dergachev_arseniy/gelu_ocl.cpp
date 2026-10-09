#include "gelu_ocl.h"
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <memory>

namespace {

constexpr char GELU_KERNEL_SOURCE[] = R"CLC(
__kernel void gelu(__global float* values, ulong element_count) {
    size_t index = get_global_id(0);
    if (index < element_count) {
        float value = values[index];
        float exponent = 1.5957691216f * value * (1.0f + 0.044715f * value * value);
        float decay = native_exp(-fabs(exponent));
        float sigmoid = exponent >= 0.0f ? 1.0f / (1.0f + decay) : decay / (1.0f + decay);
        values[index] = value * sigmoid;
    }
}
)CLC";

struct OpenCLState {
    int platform_index;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem device_values = nullptr;
    size_t buffer_capacity = 0;

    explicit OpenCLState(int selected_platform) : platform_index(selected_platform) {
        cl_uint platform_count = 0;
        clGetPlatformIDs(0, nullptr, &platform_count);
        std::vector<cl_platform_id> platforms(platform_count);
        clGetPlatformIDs(platform_count, platforms.data(), nullptr);

        cl_device_id device;
        clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueue(context, device, 0, nullptr);
        const char* kernel_source = GELU_KERNEL_SOURCE;
        program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, nullptr);
        clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        kernel = clCreateKernel(program, "gelu", nullptr);
    }

    ~OpenCLState() {
        if (device_values) {
            clReleaseMemObject(device_values);
        }
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }

    void Reserve(size_t byte_count) {
        if (byte_count > buffer_capacity) {
            if (device_values) {
                clReleaseMemObject(device_values);
            }
            device_values = clCreateBuffer(context, CL_MEM_READ_WRITE, byte_count, nullptr, nullptr);
            buffer_capacity = byte_count;
        }
    }
};

}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }

    static std::unique_ptr<OpenCLState> state;
    if (!state || state->platform_index != platform) {
        state = std::make_unique<OpenCLState>(platform);
    }

    size_t byte_count = input.size() * sizeof(float);
    state->Reserve(byte_count);
    cl_ulong element_count = input.size();
    clSetKernelArg(state->kernel, 0, sizeof(cl_mem), &state->device_values);
    clSetKernelArg(state->kernel, 1, sizeof(element_count), &element_count);

    size_t local_size = 256;
    size_t global_size = (input.size() + local_size - 1) / local_size * local_size;
    clEnqueueWriteBuffer(state->queue, state->device_values, CL_FALSE, 0, byte_count,
                         input.data(), 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(state->queue, state->kernel, 1, nullptr, &global_size,
                          &local_size, 0, nullptr, nullptr);
    clFlush(state->queue);

    std::vector<float> result(input.size());
    clEnqueueReadBuffer(state->queue, state->device_values, CL_TRUE, 0, byte_count,
                        result.data(), 0, nullptr, nullptr);
    return result;
}
