#include "ternix/tokenizer.hpp"
#include <algorithm>
#include <cstring>
#include <queue>
#include <stdexcept>

namespace ternix {
namespace {

struct Blob {
    const uint8_t* p;
    size_t size, pos = 0;
    uint32_t u32() {
        if (pos + 4 > size) throw std::runtime_error("tokenizer: truncated blob");
        uint32_t v;
        std::memcpy(&v, p + pos, 4);
        pos += 4;
        return v;
    }
    std::string str(size_t n) {
        if (pos + n > size) throw std::runtime_error("tokenizer: truncated blob");
        std::string s(reinterpret_cast<const char*>(p + pos), n);
        pos += n;
        return s;
    }
};

// Decodes one UTF-8 code point at s[i]; invalid bytes decode as themselves (one byte, value >= 0x80).
uint32_t next_cp(std::string_view s, size_t& i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    auto cont = [&](size_t k) { return i + k < s.size() && (static_cast<unsigned char>(s[i + k]) & 0xC0) == 0x80; };
    if (c < 0x80) { ++i; return c; }
    if ((c & 0xE0) == 0xC0 && cont(1)) {
        const uint32_t v = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
        i += 2;
        return v;
    }
    if ((c & 0xF0) == 0xE0 && cont(1) && cont(2)) {
        const uint32_t v = ((c & 0x0Fu) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
                           (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
        i += 3;
        return v;
    }
    if ((c & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
        const uint32_t v = ((c & 0x07u) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12) |
                           ((static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6) | (static_cast<unsigned char>(s[i + 3]) & 0x3Fu);
        i += 4;
        return v;
    }
    ++i;
    return c;
}

} // namespace

Tokenizer::Tokenizer(const uint8_t* blob, size_t size) {
    Blob b{blob, size};
    const uint32_t n = b.u32();
    tokens_.resize(n);
    special_flag_.assign(n, 0);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t len = b.u32();
        tokens_[i] = b.str(len);
    }
    const uint32_t ns = b.u32();
    for (uint32_t i = 0; i < ns; ++i) {
        const int32_t id = static_cast<int32_t>(b.u32());
        special_ids_.push_back(id);
        special_flag_[static_cast<size_t>(id)] = 1;
    }
    for (uint32_t i = 0; i < n; ++i) token_to_id_.emplace(tokens_[i], static_cast<int32_t>(i));
    for (int c = 0; c < 256; ++c) {
        auto it = token_to_id_.find(std::string(1, static_cast<char>(c)));
        byte_id_[c] = it == token_to_id_.end() ? -1 : it->second;
    }
    const uint32_t nm = b.u32();
    merge_rank_.reserve(nm * 2);
    merge_result_.resize(nm, -1);
    for (uint32_t r = 0; r < nm; ++r) {
        const uint32_t a = b.u32(), c = b.u32();
        merge_rank_.emplace((static_cast<uint64_t>(a) << 32) | c, r);
        auto it = token_to_id_.find(tokens_[a] + tokens_[c]);
        if (it != token_to_id_.end()) merge_result_[r] = it->second;
    }
    for (auto* ranges : {&letters_, &numbers_, &spaces_}) {
        const uint32_t k = b.u32();
        ranges->resize(k);
        for (uint32_t i = 0; i < k; ++i) {
            const uint32_t lo = b.u32(), hi = b.u32();
            (*ranges)[i] = {lo, hi};
        }
    }
}

bool Tokenizer::in_ranges(const std::vector<std::pair<uint32_t, uint32_t>>& r, uint32_t c) {
    auto it = std::upper_bound(r.begin(), r.end(), c, [](uint32_t v, const std::pair<uint32_t, uint32_t>& p) { return v < p.first; });
    return it != r.begin() && c <= std::prev(it)->second;
}
bool Tokenizer::is_letter(uint32_t c) const { return in_ranges(letters_, c); }
bool Tokenizer::is_number(uint32_t c) const { return in_ranges(numbers_, c); }
bool Tokenizer::is_space(uint32_t c) const { return in_ranges(spaces_, c); }

// Hand-written equivalent of
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
std::vector<std::string_view> Tokenizer::pretokenize(std::string_view text) const {
    // decode once, remembering each code point's byte offset
    std::vector<uint32_t> cp;
    std::vector<size_t> off;
    for (size_t i = 0; i < text.size();) {
        off.push_back(i);
        cp.push_back(next_cp(text, i));
    }
    off.push_back(text.size());
    const size_t m = cp.size();
    std::vector<std::string_view> out;
    auto lower = [](uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };
    auto crlf = [](uint32_t c) { return c == '\r' || c == '\n'; };

    size_t i = 0;
    while (i < m) {
        size_t end = i;
        const uint32_t c = cp[i];
        // 1. contractions
        if (c == '\'' && i + 1 < m) {
            const uint32_t d = lower(cp[i + 1]);
            if (d == 's' || d == 't' || d == 'm' || d == 'd') end = i + 2;
            else if (i + 2 < m) {
                const uint32_t e = lower(cp[i + 2]);
                if ((d == 'r' && e == 'e') || (d == 'v' && e == 'e') || (d == 'l' && e == 'l')) end = i + 3;
            }
        }
        // 2. optional non-letter/number/newline char followed by letters
        if (end == i) {
            size_t j = i;
            if (!crlf(c) && !is_letter(c) && !is_number(c) && i + 1 < m && is_letter(cp[i + 1])) j = i + 1;
            if (is_letter(cp[j])) {
                while (j < m && is_letter(cp[j])) ++j;
                end = j;
            }
        }
        // 3. up to three digits
        if (end == i && is_number(c)) {
            size_t j = i;
            while (j < m && j - i < 3 && is_number(cp[j])) ++j;
            end = j;
        }
        // 4. optional space, run of symbols, trailing newlines
        if (end == i) {
            size_t j = i;
            if (c == ' ') ++j;
            const size_t s = j;
            while (j < m && !is_space(cp[j]) && !is_letter(cp[j]) && !is_number(cp[j])) ++j;
            if (j > s) {
                while (j < m && crlf(cp[j])) ++j;
                end = j;
            }
        }
        // 5-7. whitespace runs
        if (end == i && is_space(c)) {
            size_t j = i;
            while (j < m && is_space(cp[j])) ++j;        // [i, j) is the whole whitespace run
            size_t last_nl = j;
            for (size_t k = j; k > i; --k)
                if (crlf(cp[k - 1])) { last_nl = k; break; }
            if (last_nl != j || (j > i && crlf(cp[j - 1]))) {
                end = last_nl;                              // \s*[\r\n]+ : up to the last newline in the run
            } else if (j == m) {
                end = j;                                    // \s+(?!\S) at end of text
            } else if (j - i >= 2) {
                end = j - 1;                                // \s+(?!\S): leave the last space for the next token
            } else {
                end = j;                                    // \s+
            }
        }
        if (end == i) end = i + 1;                          // unreachable for valid input; guarantees progress
        out.push_back(text.substr(off[i], off[end] - off[i]));
        i = end;
    }
    return out;
}

void Tokenizer::bpe_word(std::string_view word, std::vector<int32_t>& out) const {
    auto whole = token_to_id_.find(std::string(word));
    if (whole != token_to_id_.end()) {
        out.push_back(whole->second);
        return;
    }
    const size_t n = word.size();
    std::vector<int32_t> id(n), prev(n), next(n);
    for (size_t i = 0; i < n; ++i) {
        id[i] = byte_id_[static_cast<unsigned char>(word[i])];
        prev[i] = static_cast<int32_t>(i) - 1;
        next[i] = i + 1 < n ? static_cast<int32_t>(i + 1) : -1;
    }
    struct Cand {
        uint32_t rank;
        int32_t left, right;
        int32_t id_l, id_r;       // token ids at push time, to detect stale entries
        bool operator>(const Cand& o) const { return rank != o.rank ? rank > o.rank : left > o.left; }
    };
    std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> q;
    auto push = [&](int32_t l, int32_t r) {
        if (l < 0 || r < 0) return;
        auto it = merge_rank_.find((static_cast<uint64_t>(static_cast<uint32_t>(id[l])) << 32) | static_cast<uint32_t>(id[r]));
        if (it != merge_rank_.end()) q.push({it->second, l, r, id[l], id[r]});
    };
    for (size_t i = 0; i + 1 < n; ++i) push(static_cast<int32_t>(i), static_cast<int32_t>(i + 1));
    while (!q.empty()) {
        const Cand c = q.top();
        q.pop();
        if (id[c.left] != c.id_l || id[c.right] != c.id_r || next[c.left] != c.right) continue;   // stale
        const int32_t merged = merge_result_[c.rank];
        if (merged < 0) continue;
        id[c.left] = merged;
        const int32_t nr = next[c.right];
        next[c.left] = nr;
        if (nr >= 0) prev[nr] = c.left;
        id[c.right] = -1;
        push(prev[c.left], c.left);
        push(c.left, nr);
    }
    for (int32_t i = 0; i >= 0 && i < static_cast<int32_t>(n); i = next[i]) {
        if (id[i] >= 0) out.push_back(id[i]);
        if (next[i] < 0) break;
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool add_bos, bool parse_special) const {
    std::vector<int32_t> out;
    if (add_bos) out.push_back(bos_);
    size_t seg_start = 0, i = 0;
    auto flush = [&](size_t end) {
        if (end > seg_start)
            for (std::string_view w : pretokenize(text.substr(seg_start, end - seg_start))) bpe_word(w, out);
    };
    while (i < text.size()) {
        bool matched = false;
        if (parse_special && text[i] == '<') {
            int32_t best = -1;
            size_t best_len = 0;
            for (int32_t sid : special_ids_) {
                const std::string& s = tokens_[static_cast<size_t>(sid)];
                if (s.size() > best_len && text.compare(i, s.size(), s) == 0) { best = sid; best_len = s.size(); }
            }
            if (best >= 0) {
                flush(i);
                out.push_back(best);
                i += best_len;
                seg_start = i;
                matched = true;
            }
        }
        if (!matched) ++i;
    }
    flush(text.size());
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string s;
    for (int32_t id : ids) s += tokens_[static_cast<size_t>(id)];
    return s;
}

} // namespace ternix
