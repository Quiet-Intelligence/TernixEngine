#include "ternix/format.hpp"
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ternix {
namespace {

constexpr char kMagic[8] = {'T', 'E', 'R', 'N', 'I', 'X', '0', '1'};
constexpr size_t kCfgU32 = 40, kCfgF32 = 16;

struct Reader {
    const uint8_t* p;
    size_t size, pos = 0;
    template <class T>
    T take() {
        if (pos + sizeof(T) > size) throw std::runtime_error("ternix: truncated header");
        T v;
        std::memcpy(&v, p + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }
    const uint8_t* bytes(size_t n) {
        if (pos + n > size) throw std::runtime_error("ternix: truncated header");
        const uint8_t* r = p + pos;
        pos += n;
        return r;
    }
};

} // namespace

ModelFile::ModelFile(const std::string& path) {
#ifdef _WIN32
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("ternix: cannot open " + path);
    LARGE_INTEGER sz;
    GetFileSizeEx(f, &sz);
    size_ = static_cast<size_t>(sz.QuadPart);
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) {
        CloseHandle(f);
        throw std::runtime_error("ternix: CreateFileMapping failed");
    }
    base_ = static_cast<const uint8_t*>(MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0));
    handle_ = f;
    map_ = m;
    if (!base_) throw std::runtime_error("ternix: MapViewOfFile failed");
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("ternix: cannot open " + path);
    struct stat st;
    fstat(fd, &st);
    size_ = static_cast<size_t>(st.st_size);
    void* m = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("ternix: mmap failed");
    base_ = static_cast<const uint8_t*>(m);
#endif
    Reader r{base_, size_};
    if (size_ < 16 || std::memcmp(base_, kMagic, 8) != 0) throw std::runtime_error("ternix: bad magic (not a .ternix file)");
    r.pos = 8;
    const uint64_t header_size = r.take<uint64_t>();
    uint32_t u[kCfgU32];
    float fl[kCfgF32];
    for (auto& x : u) x = r.take<uint32_t>();
    for (auto& x : fl) x = r.take<float>();
    cfg_.arch = u[0];
    cfg_.vocab = u[1];
    cfg_.hidden = u[2];
    cfg_.layers = u[3];
    cfg_.heads = u[4];
    cfg_.kv_heads = u[5];
    cfg_.head_dim = u[6];
    cfg_.ffn = u[7];
    cfg_.max_pos = u[8];
    cfg_.tie_embeddings = u[9] != 0;
    cfg_.act = static_cast<ModelConfig::Act>(u[10]);
    cfg_.subln = u[11] != 0;
    cfg_.bos = u[12];
    cfg_.eos = u[13];
    cfg_.eot = u[14];
    const uint32_t n_tensors = u[15];
    tok_off_ = u[16];
    tok_bytes_ = u[17];
    cfg_.rms_eps = fl[0];
    cfg_.rope_theta = fl[1];
    for (uint32_t i = 0; i < n_tensors; ++i) {
        const uint16_t nl = r.take<uint16_t>();
        std::string name(reinterpret_cast<const char*>(r.bytes(nl)), nl);
        TensorView t;
        t.dtype = static_cast<DType>(r.take<uint32_t>());
        const uint32_t nd = r.take<uint32_t>();
        for (uint32_t d = 0; d < nd; ++d) t.dims.push_back(r.take<uint64_t>());
        t.scale = r.take<float>();
        const uint64_t off = r.take<uint64_t>();
        t.nbytes = r.take<uint64_t>();
        if (off + t.nbytes > size_) throw std::runtime_error("ternix: tensor '" + name + "' extends past end of file");
        t.data = base_ + off;
        tensors_.emplace(std::move(name), std::move(t));
    }
    if (r.pos != header_size) throw std::runtime_error("ternix: header size mismatch");
    if (tok_off_ + tok_bytes_ > size_) throw std::runtime_error("ternix: tokenizer section out of range");
}

ModelFile::~ModelFile() {
#ifdef _WIN32
    if (base_) UnmapViewOfFile(base_);
    if (map_) CloseHandle(static_cast<HANDLE>(map_));
    if (handle_) CloseHandle(static_cast<HANDLE>(handle_));
#else
    if (base_) munmap(const_cast<uint8_t*>(base_), size_);
#endif
}

const TensorView* ModelFile::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const TensorView& ModelFile::get(const std::string& name) const {
    if (const TensorView* t = find(name)) return *t;
    throw std::runtime_error("ternix: missing tensor '" + name + "'");
}

void ModelFile::prefault() const {
    volatile uint8_t sink = 0;
    for (size_t i = 0; i < size_; i += 4096) sink = sink + base_[i];
    (void)sink;
}

} // namespace ternix
