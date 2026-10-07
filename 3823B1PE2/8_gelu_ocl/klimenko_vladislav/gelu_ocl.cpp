#include "gelu_ocl.h"

#include <CL/cl.h>

#include <cstddef>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace
{

    const char *kKernelSource = R"CLC(
#ifndef SQRT_2_OVER_PI
#define SQRT_2_OVER_PI 0.7978845608028654f
#endif

#ifndef GELU_COEF
#define GELU_COEF 0.044715f
#endif

inline float gelu_scalar(float x)
{
    float x3 = x * x * x;
    float two_z = 2.0f * SQRT_2_OVER_PI * (x + GELU_COEF * x3);
    return x / (1.0f + exp(-two_z));
}

// -------- scalar version --------
__kernel void gelu_kernel(
    __global const float* __restrict input,
    __global float* __restrict output,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    output[i] = gelu_scalar(input[i]);
}

// -------- vectorized version (n % 4 == 0) --------
__kernel void gelu_kernel_vec4(
    __global const float4* __restrict input,
    __global float4* __restrict output,
    const int n4)
{
    int i = get_global_id(0);
    if (i >= n4) return;

    float4 v = input[i];
    float4 r;
    r.x = gelu_scalar(v.x);
    r.y = gelu_scalar(v.y);
    r.z = gelu_scalar(v.z);
    r.w = gelu_scalar(v.w);
    output[i] = r;
}
)CLC";

    struct OCLState
    {
        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_program program = nullptr;
        cl_kernel kernel_scalar = nullptr;
        cl_kernel kernel_vec4 = nullptr;
        cl_device_id device = nullptr;

        cl_mem buf_in = nullptr;
        cl_mem buf_out = nullptr;
        size_t buf_bytes = 0;
    };

    std::unordered_map<int, OCLState> &states()
    {
        static std::unordered_map<int, OCLState> map;
        return map;
    }

    std::mutex &states_mutex()
    {
        static std::mutex m;
        return m;
    }

    void check(cl_int err, const char *what)
    {
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error(std::string("OpenCL error in ") + what +
                                     ": " + std::to_string(err));
        }
    }

    void init_state(int platform_index, OCLState &st)
    {
        cl_uint num_platforms = 0;
        check(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs(count)");
        if (num_platforms == 0)
        {
            throw std::runtime_error("No OpenCL platforms found");
        }
        if (platform_index < 0 ||
            static_cast<cl_uint>(platform_index) >= num_platforms)
        {
            throw std::runtime_error("Invalid platform index");
        }

        cl_platform_id platform_id = nullptr;
        check(clGetPlatformIDs(1, &platform_id, nullptr), "clGetPlatformIDs(get)");
        std::vector<cl_platform_id> platforms(num_platforms);
        check(clGetPlatformIDs(num_platforms, platforms.data(), nullptr),
              "clGetPlatformIDs(all)");
        platform_id = platforms[platform_index];

        cl_device_id device = nullptr;
        cl_int err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS)
        {
            check(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device, nullptr),
                  "clGetDeviceIDs");
        }
        st.device = device;

        cl_int ctx_err = CL_SUCCESS;
        st.context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ctx_err);
        check(ctx_err, "clCreateContext");

        cl_int q_err = CL_SUCCESS;
#ifdef CL_VERSION_2_0
        st.queue = clCreateCommandQueueWithProperties(st.context, device, nullptr, &q_err);
#else
        st.queue = clCreateCommandQueue(st.context, device, 0, &q_err);
#endif
        check(q_err, "clCreateCommandQueue");

        const char *src = kKernelSource;
        size_t src_len = std::strlen(src);
        cl_int prog_err = CL_SUCCESS;
        st.program = clCreateProgramWithSource(st.context, 1, &src, &src_len, &prog_err);
        check(prog_err, "clCreateProgramWithSource");

        const char *build_opts = "-cl-fast-relaxed-math";
        cl_int build_err = clBuildProgram(st.program, 1, &device, build_opts, nullptr, nullptr);
        if (build_err != CL_SUCCESS)
        {
            size_t log_size = 0;
            clGetProgramBuildInfo(st.program, device, CL_PROGRAM_BUILD_LOG,
                                  0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(st.program, device, CL_PROGRAM_BUILD_LOG,
                                  log_size, log.data(), nullptr);
            throw std::runtime_error("OpenCL build failed: " + log);
        }

        cl_int k1_err = CL_SUCCESS;
        st.kernel_scalar = clCreateKernel(st.program, "gelu_kernel", &k1_err);
        check(k1_err, "clCreateKernel(gelu_kernel)");

        cl_int k2_err = CL_SUCCESS;
        st.kernel_vec4 = clCreateKernel(st.program, "gelu_kernel_vec4", &k2_err);
        check(k2_err, "clCreateKernel(gelu_kernel_vec4)");
    }

    void ensure_buffers(OCLState &st, size_t bytes)
    {
        if (bytes <= st.buf_bytes)
            return;

        if (st.buf_in)
            clReleaseMemObject(st.buf_in);
        if (st.buf_out)
            clReleaseMemObject(st.buf_out);

        cl_int err1 = CL_SUCCESS, err2 = CL_SUCCESS;
        st.buf_in = clCreateBuffer(st.context, CL_MEM_READ_ONLY, bytes, nullptr, &err1);
        st.buf_out = clCreateBuffer(st.context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err2);
        check(err1, "clCreateBuffer(in)");
        check(err2, "clCreateBuffer(out)");

        st.buf_bytes = bytes;
    }

} // namespace

std::vector<float> GeluOCL(const std::vector<float> &input, int platform)
{
    const size_t n = input.size();
    std::vector<float> output(n);

    if (n == 0)
    {
        return output;
    }

    OCLState *st = nullptr;
    {
        std::lock_guard<std::mutex> lock(states_mutex());
        auto &map = states();
        auto it = map.find(platform);
        if (it == map.end())
        {
            OCLState fresh;
            init_state(platform, fresh);
            it = map.emplace(platform, std::move(fresh)).first;
        }
        st = &it->second;
    }

    const size_t bytes = n * sizeof(float);
    ensure_buffers(*st, bytes);

    check(clEnqueueWriteBuffer(st->queue, st->buf_in, CL_TRUE, 0, bytes,
                               input.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer");

    const bool vec4_ok = (n % 4 == 0);
    const int n_int = static_cast<int>(n);
    const int n4_int = static_cast<int>(n / 4);

    cl_int err = CL_SUCCESS;
    size_t global_size = 0;

    if (vec4_ok)
    {
        err = clSetKernelArg(st->kernel_vec4, 0, sizeof(cl_mem), &st->buf_in);
        check(err, "clSetKernelArg(vec4, 0)");
        err = clSetKernelArg(st->kernel_vec4, 1, sizeof(cl_mem), &st->buf_out);
        check(err, "clSetKernelArg(vec4, 1)");
        err = clSetKernelArg(st->kernel_vec4, 2, sizeof(int), &n4_int);
        check(err, "clSetKernelArg(vec4, 2)");

        global_size = static_cast<size_t>(n4_int);
        check(clEnqueueNDRangeKernel(st->queue, st->kernel_vec4, 1, nullptr,
                                     &global_size, nullptr, 0, nullptr, nullptr),
              "clEnqueueNDRangeKernel(vec4)");
    }
    else
    {
        err = clSetKernelArg(st->kernel_scalar, 0, sizeof(cl_mem), &st->buf_in);
        check(err, "clSetKernelArg(scalar, 0)");
        err = clSetKernelArg(st->kernel_scalar, 1, sizeof(cl_mem), &st->buf_out);
        check(err, "clSetKernelArg(scalar, 1)");
        err = clSetKernelArg(st->kernel_scalar, 2, sizeof(int), &n_int);
        check(err, "clSetKernelArg(scalar, 2)");

        global_size = static_cast<size_t>(n_int);
        check(clEnqueueNDRangeKernel(st->queue, st->kernel_scalar, 1, nullptr,
                                     &global_size, nullptr, 0, nullptr, nullptr),
              "clEnqueueNDRangeKernel(scalar)");
    }

    check(clEnqueueReadBuffer(st->queue, st->buf_out, CL_TRUE, 0, bytes,
                              output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer");

    return output;
}