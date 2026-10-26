#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include "ternix/bitpack.hpp"
#include "ternix/mul_mat_ternary.h"
#ifdef TERNIX_HAS_CUDA
#include "ternix/cuda_kernels.cuh"
#endif

namespace py = pybind11;

namespace {

using i8_array = py::array_t<int8_t, py::array::c_style | py::array::forcecast>;
using u8_array = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// Pack a (K, N) int8 matrix with values in {-1, 0, +1} into the (K, N/4) uint8 layout the kernels read.
u8_array pack_ternary(i8_array w) {
    if (w.ndim() != 2) throw std::invalid_argument("pack_ternary expects a 2-D (K, N) array");
    const py::ssize_t K = w.shape(0), N = w.shape(1);
    if (N % 4 != 0) throw std::invalid_argument("N must be a multiple of 4");
    const int8_t* src = w.data();
    for (py::ssize_t i = 0; i < K * N; ++i)
        if (src[i] < -1 || src[i] > 1) throw std::invalid_argument("weights must be in {-1, 0, +1}");
    u8_array out({K, N / 4});
    ternix::pack_ternary_weights(out.mutable_data(), src, static_cast<size_t>(K * N));
    return out;
}

i8_array unpack_ternary(u8_array packed) {
    if (packed.ndim() != 2) throw std::invalid_argument("unpack_ternary expects a 2-D (K, N/4) array");
    const py::ssize_t K = packed.shape(0), N4 = packed.shape(1);
    i8_array out({K, N4 * 4});
    ternix::unpack_ternary_weights(out.mutable_data(), packed.data(), static_cast<size_t>(K * N4 * 4));
    return out;
}

py::array_t<int32_t> ternary_matmul(i8_array a, u8_array b_packed, const std::string& impl) {
    if (a.ndim() != 2 || b_packed.ndim() != 2) throw std::invalid_argument("A must be (M, K) and B_packed (K, N/4)");
    const size_t M = a.shape(0), K = a.shape(1), N = static_cast<size_t>(b_packed.shape(1)) * 4;
    if (static_cast<size_t>(b_packed.shape(0)) != K) throw std::invalid_argument("inner dimensions do not match");
    py::array_t<int32_t> c({static_cast<py::ssize_t>(M), static_cast<py::ssize_t>(N)});
    const int8_t* pa = a.data();
    const uint8_t* pb = b_packed.data();
    int32_t* pc = c.mutable_data();
    if (impl == "naive") {
        ternix::mul_mat_ternary_naive(pa, pb, pc, M, K, N);
    } else if (impl == "scalar") {
        ternix::mul_mat_ternary_scalar(pa, pb, pc, M, K, N);
    } else if (impl == "avx2" || impl == "avx2_blocked") {
        if (N % 8 != 0) throw std::invalid_argument("AVX2 kernels need N to be a multiple of 8");
        if (impl == "avx2") ternix::mul_mat_ternary_avx2(pa, pb, pc, M, K, N);
        else ternix::mul_mat_ternary_avx2_blocked(pa, pb, pc, M, K, N);
#ifdef TERNIX_HAS_CUDA
    } else if (impl == "cuda_naive" || impl == "cuda_tiled") {
        py::gil_scoped_release release;
        const cudaError_t e = ternix::mul_mat_ternary_cuda_host(pa, pb, pc, static_cast<int>(M), static_cast<int>(K),
                                                                 static_cast<int>(N), impl == "cuda_tiled");
        if (e != cudaSuccess) {
            py::gil_scoped_acquire acquire;
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e));
        }
#endif
    } else {
        throw std::invalid_argument("unknown impl '" + impl + "'");
    }
    return c;
}

} // namespace

PYBIND11_MODULE(ternix_engine, m) {
    m.doc() = "TernixEngine: 1.58-bit ternary GEMM kernels (CPU AVX2, optional CUDA) and packing utilities";

    m.def("pack_ternary", &pack_ternary, py::arg("w"), "Pack a (K, N) int8 array in {-1,0,1} into a (K, N/4) uint8 array");
    m.def("unpack_ternary", &unpack_ternary, py::arg("packed"), "Inverse of pack_ternary");
    m.def("ternary_matmul", &ternary_matmul, py::arg("a"), py::arg("b_packed"), py::arg("impl") = "avx2_blocked",
          "C = A (M,K int8) @ W (K,N ternary packed) -> (M,N int32). impl: naive | scalar | avx2 | avx2_blocked"
#ifdef TERNIX_HAS_CUDA
          " | cuda_naive | cuda_tiled"
#endif
          );
#ifdef TERNIX_HAS_CUDA
    m.attr("has_cuda") = true;
#else
    m.attr("has_cuda") = false;
#endif
}
