#pragma once
// .ternix container reader (layout in docs/FORMAT.md; writer in scripts/ternix_format.py).
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ternix {

enum class DType : uint32_t { F32 = 0, BF16 = 1, F16 = 2, I2K128 = 3 };

struct ModelConfig {
    uint32_t arch = 0, vocab = 0, hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, ffn = 0, max_pos = 0;
    bool tie_embeddings = true;
    enum class Act : uint32_t { SiluGlu = 0, Relu2Glu = 1 } act = Act::Relu2Glu;
    bool subln = true;
    uint32_t bos = 0, eos = 0, eot = 0;
    float rms_eps = 1e-5f, rope_theta = 10000.f;
};

struct TensorView {
    DType dtype = DType::F32;
    std::vector<uint64_t> dims;
    float scale = 1.f;               // per-tensor weight scale (mean |W|) for I2K128 tensors
    const uint8_t* data = nullptr;
    uint64_t nbytes = 0;
};

class ModelFile {
public:
    explicit ModelFile(const std::string& path);   // memory-maps the file; throws std::runtime_error on any format error
    ~ModelFile();
    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;

    const ModelConfig& config() const { return cfg_; }
    const TensorView* find(const std::string& name) const;
    const TensorView& get(const std::string& name) const;   // throws if missing
    const uint8_t* tokenizer_data() const { return base_ + tok_off_; }
    size_t tokenizer_size() const { return tok_bytes_; }
    size_t file_size() const { return size_; }
    // Touch every page once so the first forward pass is not paid for in page faults.
    void prefault() const;

private:
    const uint8_t* base_ = nullptr;
    size_t size_ = 0;
    void* handle_ = nullptr;     // platform mapping handles
    void* map_ = nullptr;
    ModelConfig cfg_;
    uint64_t tok_off_ = 0, tok_bytes_ = 0;
    std::unordered_map<std::string, TensorView> tensors_;
};

} // namespace ternix
