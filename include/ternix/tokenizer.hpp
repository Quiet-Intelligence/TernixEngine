#pragma once
// Byte-level BPE tokenizer with the Llama-3 pre-tokenizer split, loaded from the tokenizer section of a .ternix file.
// Matches Hugging Face `tokenizers` (ignore_merges = true): a pre-token that is itself a vocabulary entry is emitted
// whole, otherwise merges are applied by rank.
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ternix {

class Tokenizer {
public:
    Tokenizer(const uint8_t* blob, size_t size);

    // parse_special: literal "<|eot_id|>"-style strings in `text` become their special token ids.
    std::vector<int32_t> encode(std::string_view text, bool add_bos, bool parse_special) const;
    // Raw UTF-8 bytes of one token (special tokens return their literal text).
    const std::string& token_bytes(int32_t id) const { return tokens_[static_cast<size_t>(id)]; }
    std::string decode(const std::vector<int32_t>& ids) const;
    size_t vocab_size() const { return tokens_.size(); }
    bool is_special(int32_t id) const { return special_flag_[static_cast<size_t>(id)]; }
    int32_t bos() const { return bos_; }

    // Exposed for tests: the Llama-3 regex split of UTF-8 text into pre-tokens.
    std::vector<std::string_view> pretokenize(std::string_view text) const;

private:
    void bpe_word(std::string_view word, std::vector<int32_t>& out) const;
    bool is_letter(uint32_t c) const;
    bool is_number(uint32_t c) const;
    bool is_space(uint32_t c) const;
    static bool in_ranges(const std::vector<std::pair<uint32_t, uint32_t>>& r, uint32_t c);

    std::vector<std::string> tokens_;
    std::vector<uint8_t> special_flag_;
    std::vector<int32_t> special_ids_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<uint64_t, uint32_t> merge_rank_;      // (a << 32 | b) -> rank
    std::vector<int32_t> merge_result_;                      // rank -> merged token id
    int32_t byte_id_[256];
    std::vector<std::pair<uint32_t, uint32_t>> letters_, numbers_, spaces_;
    int32_t bos_ = 128000;
};

} // namespace ternix
