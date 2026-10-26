#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include "ternix/bitpack.hpp"
#include "ternix/format.hpp"
#include "ternix/model.hpp"
#include "ternix/mul_mat_ternary.h"
#include "ternix/sampler.hpp"
#include "ternix/tokenizer.hpp"
#ifdef TERNIX_HAS_CUDA
#include "ternix/cuda_kernels.cuh"
#include "ternix/cuda_engine.hpp"
#endif

namespace py = pybind11;

namespace {

// Python-facing engine: owns the mapped model, tokenizer and either the CPU or the CUDA backend.
class PyEngine {
public:
    PyEngine(const std::string& path, bool gpu, int threads, int ctx) : file_(path), tok_(file_.tokenizer_data(), file_.tokenizer_size()) {
        if (gpu) {
#ifdef TERNIX_HAS_CUDA
            gpu_ = std::make_unique<ternix::CudaModel>(file_, ctx);
#else
            throw std::runtime_error("built without CUDA");
#endif
        } else {
            file_.prefault();
            cpu_ = std::make_unique<ternix::Model>(file_, threads, ctx);
        }
    }
    std::vector<int32_t> encode(const std::string& text, bool add_bos, bool parse_special) const { return tok_.encode(text, add_bos, parse_special); }
    py::bytes decode(const std::vector<int32_t>& ids) const { return py::bytes(tok_.decode(ids)); }
    size_t vocab_size() const { return file_.config().vocab; }
    int max_ctx() const { return gpu_ ? gpu_max_ctx() : cpu_->max_ctx(); }
    bool is_gpu() const { return gpu_ != nullptr; }

    // Logits for a fresh sequence: last position only (1-D) or every position (2-D, all=true).
    py::array_t<float> logits(const std::vector<int32_t>& ids, bool all) {
        const size_t V = vocab_size();
        const size_t rows = all ? ids.size() : 1;
        py::array_t<float> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(V)});
        {
            py::gil_scoped_release release;
            reset();
            forward(ids.data(), static_cast<int>(ids.size()), out.mutable_data(), all ? 2 : 1);
        }
        if (!all) out.resize({static_cast<py::ssize_t>(V)});
        return out;
    }
    // Teacher forcing on a fresh sequence: (nll[i], argmax[i]) for the prediction after tokens[i] of targets[i] (<0 skips).
    std::pair<std::vector<float>, std::vector<int32_t>> nll(const std::vector<int32_t>& tokens, const std::vector<int32_t>& targets) {
        if (tokens.size() != targets.size()) throw std::invalid_argument("tokens and targets must have equal length");
        std::vector<float> nll(tokens.size());
        std::vector<int32_t> am(tokens.size());
        py::gil_scoped_release release;
        reset();
        bool ok = gpu_ ? forward_nll_gpu(tokens, targets, nll, am) : cpu_->forward_nll(tokens.data(), static_cast<int>(tokens.size()), targets.data(), nll.data(), am.data());
        if (!ok) throw std::runtime_error("sequence exceeds the context window");
        return {nll, am};
    }
    // Incremental interface (used by the HTTP server): prefill resets the sequence, step appends one token. Both return the
    // next-token logits.
    py::array_t<float> prefill(const std::vector<int32_t>& ids) {
        py::array_t<float> out(static_cast<py::ssize_t>(vocab_size()));
        py::gil_scoped_release release;
        reset();
        if (!forward(ids.data(), static_cast<int>(ids.size()), out.mutable_data(), 1)) throw std::runtime_error("sequence exceeds the context window");
        return out;
    }
    py::array_t<float> step(int32_t token) {
        py::array_t<float> out(static_cast<py::ssize_t>(vocab_size()));
        py::gil_scoped_release release;
        if (!forward(&token, 1, out.mutable_data(), 1)) throw std::runtime_error("context window full");
        return out;
    }
    std::string generate(const std::string& prompt, int n, float temperature, int top_k, float top_p, uint64_t seed) {
        std::vector<int32_t> ids = tok_.encode(prompt, true, true);
        ternix::SamplerConfig sc;
        sc.temperature = temperature; sc.top_k = top_k; sc.top_p = top_p; sc.seed = seed;
        ternix::Sampler sampler(sc);
        std::vector<float> logits(vocab_size());
        std::string out;
        py::gil_scoped_release release;
        reset();
        std::vector<int32_t> hist = ids;
        forward(ids.data(), static_cast<int>(ids.size()), logits.data(), 1);
        for (int i = 0; i < n; ++i) {
            const int32_t next = sampler.sample(logits.data(), logits.size(), hist);
            if (next == static_cast<int32_t>(file_.config().eos) || next == static_cast<int32_t>(file_.config().eot) || next == 128001) break;
            hist.push_back(next);
            out += tok_.token_bytes(next);
            if (!forward(&next, 1, logits.data(), 1)) break;
        }
        return out;
    }

private:
    void reset() { if (gpu_) { gpu_reset(); } else cpu_->reset(); }
    bool forward(const int32_t* t, int n, float* logits, int mode) {
#ifdef TERNIX_HAS_CUDA
        if (gpu_) return gpu_->forward(t, n, logits, mode);
#endif
        return cpu_->forward(t, n, logits, mode);
    }
#ifdef TERNIX_HAS_CUDA
    void gpu_reset() { gpu_->reset(); }
    int gpu_max_ctx() const { return gpu_->max_ctx(); }
    bool forward_nll_gpu(const std::vector<int32_t>& t, const std::vector<int32_t>& tg, std::vector<float>& nll, std::vector<int32_t>& am) {
        return gpu_->forward_nll(t.data(), static_cast<int>(t.size()), tg.data(), nll.data(), am.data());
    }
#else
    void gpu_reset() {}
    int gpu_max_ctx() const { return 0; }
    bool forward_nll_gpu(const std::vector<int32_t>&, const std::vector<int32_t>&, std::vector<float>&, std::vector<int32_t>&) { return false; }
#endif
    ternix::ModelFile file_;
    ternix::Tokenizer tok_;
    std::unique_ptr<ternix::Model> cpu_;
#ifdef TERNIX_HAS_CUDA
    std::unique_ptr<ternix::CudaModel> gpu_;
#else
    std::unique_ptr<int> gpu_;
#endif
};

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
    py::class_<PyEngine>(m, "Engine")
        .def(py::init<const std::string&, bool, int, int>(), py::arg("path"), py::arg("gpu") = false, py::arg("threads") = 0, py::arg("ctx") = 4096)
        .def("encode", &PyEngine::encode, py::arg("text"), py::arg("add_bos") = true, py::arg("parse_special") = true)
        .def("decode", &PyEngine::decode)
        .def("logits", &PyEngine::logits, py::arg("ids"), py::arg("all") = false)
        .def("nll", &PyEngine::nll, py::arg("tokens"), py::arg("targets"))
        .def("prefill", &PyEngine::prefill, py::arg("ids"))
        .def("step", &PyEngine::step, py::arg("token"))
        .def("generate", &PyEngine::generate, py::arg("prompt"), py::arg("n") = 64, py::arg("temperature") = 0.f, py::arg("top_k") = 0,
             py::arg("top_p") = 1.f, py::arg("seed") = 1)
        .def_property_readonly("vocab_size", &PyEngine::vocab_size)
        .def_property_readonly("max_ctx", &PyEngine::max_ctx)
        .def_property_readonly("is_gpu", &PyEngine::is_gpu);
#ifdef TERNIX_HAS_CUDA
    m.attr("has_cuda") = true;
#else
    m.attr("has_cuda") = false;
#endif
}
