"""Reader/writer helpers for the .ternix container (see docs/FORMAT.md). Stdlib + numpy only.

Layout of a ternary weight (dtype I2K128): logical W[n][k] (n = output row, k = input column), K % 128 == 0.
Row n occupies K/4 bytes: for each 128-column block b, 32 bytes; byte l holds the codes of columns
b*128 + 32*j + l for j = 0..3 in bit pairs j (bits 2j+1:2j). Code = w + 1, so {-1, 0, +1} -> {0, 1, 2} (3 unused).
A 32-byte load, masked with 0x03/0x0C/0x30/0xC0 (or shifted), yields four 32-lane vectors whose activation
counterparts are four *contiguous* 32-byte loads: no lane shuffles are needed on the activation side.
"""
from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"TERNIX01"
ALIGN = 64
CONFIG_U32 = 40   # fixed-size config block: 40 x u32 then 16 x f32
CONFIG_F32 = 16

DT_F32, DT_BF16, DT_F16, DT_I2K128 = 0, 1, 2, 3
DT_NAMES = {DT_F32: "f32", DT_BF16: "bf16", DT_F16: "f16", DT_I2K128: "i2k128"}

# u32 config slots (keep in sync with include/ternix/format.hpp). Slots 18-24 (arch 2, Qwen3-style group-scaled models):
# QKNORM 1 = RMSNorm on q/k heads; GROUP = quantisation group (128) with fp16 scales in "<name>.gscale"; ACTBLOCK 1 = activations are
# quantised per 128-block instead of per token; HEAD_DTYPE 1 bf16 / 2 f16; ADD_BOS; DIGITS = max digits per pre-token (3 Llama-3, 1 Qwen);
# IGNORE_MERGES 1 = a pre-token that is a vocabulary entry is emitted whole (Llama-3), 0 = always merge (Qwen).
(C_ARCH, C_VOCAB, C_HIDDEN, C_LAYERS, C_HEADS, C_KV_HEADS, C_HEAD_DIM, C_FFN, C_MAX_POS, C_TIE, C_ACT, C_SUBLN,
 C_BOS, C_EOS, C_EOT, C_NTENSORS, C_TOK_OFF, C_TOK_BYTES,
 C_QKNORM, C_GROUP, C_ACTBLOCK, C_HEAD_DTYPE, C_ADD_BOS, C_DIGITS, C_IGNORE_MERGES) = range(25)
# f32 config slots
F_EPS, F_ROPE_THETA, F_ROPE_ATTN = 0, 1, 2


def pack_i2k128(w_ternary: np.ndarray) -> np.ndarray:
    """w_ternary: int8 array [N, K] with values in {-1,0,1} -> uint8 [N, K/4] in the I2K128 layout."""
    n, k = w_ternary.shape
    assert k % 128 == 0, "K must be a multiple of 128"
    codes = (w_ternary.astype(np.int16) + 1).astype(np.uint8)            # 0,1,2
    blocks = codes.reshape(n, k // 128, 4, 32)                           # [n, b, j, l]
    packed = (blocks[:, :, 0] | (blocks[:, :, 1] << 2) | (blocks[:, :, 2] << 4) | (blocks[:, :, 3] << 6))
    return np.ascontiguousarray(packed.reshape(n, k // 4))


def unpack_i2k128(packed: np.ndarray, k: int) -> np.ndarray:
    n = packed.shape[0]
    p = packed.reshape(n, k // 128, 1, 32)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    codes = (p >> shifts) & 3
    return (codes.astype(np.int8) - 1).reshape(n, k)


@dataclass
class TensorEntry:
    name: str
    dtype: int
    dims: tuple[int, ...]
    scale: float
    offset: int
    nbytes: int


def write_ternix(path: Path, cfg_u32: dict[int, int], cfg_f32: dict[int, float],
                 tensors: list[tuple[str, int, tuple[int, ...], float, bytes | np.ndarray]], tokenizer_blob: bytes) -> None:
    """tensors: (name, dtype, dims, scale, data). Writes header, tensor table, 64-byte aligned data, then the tokenizer."""
    u32 = [0] * CONFIG_U32
    f32 = [0.0] * CONFIG_F32
    for k, v in cfg_u32.items():
        u32[k] = v
    for k, v in cfg_f32.items():
        f32[k] = v
    u32[C_NTENSORS] = len(tensors)

    table_size = 0
    for name, _, dims, _, _ in tensors:
        table_size += 2 + len(name.encode()) + 4 + 4 + 8 * len(dims) + 4 + 8 + 8
    header_size = len(MAGIC) + 8 + CONFIG_U32 * 4 + CONFIG_F32 * 4 + table_size
    off = (header_size + ALIGN - 1) // ALIGN * ALIGN
    entries = []
    for name, dtype, dims, scale, data in tensors:
        nbytes = data.nbytes if isinstance(data, np.ndarray) else len(data)
        entries.append(TensorEntry(name, dtype, dims, scale, off, nbytes))
        off = (off + nbytes + ALIGN - 1) // ALIGN * ALIGN
    tok_off = off
    u32[C_TOK_OFF], u32[C_TOK_BYTES] = tok_off, len(tokenizer_blob)

    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", header_size))
        f.write(struct.pack(f"<{CONFIG_U32}I", *u32))
        f.write(struct.pack(f"<{CONFIG_F32}f", *f32))
        for e in entries:
            nb = e.name.encode()
            f.write(struct.pack("<H", len(nb)) + nb)
            f.write(struct.pack("<II", e.dtype, len(e.dims)))
            f.write(struct.pack(f"<{len(e.dims)}Q", *e.dims))
            f.write(struct.pack("<fQQ", e.scale, e.offset, e.nbytes))
        for (name, _, _, _, data), e in zip(tensors, entries):
            f.write(b"\0" * (e.offset - f.tell()))
            f.write(data.tobytes() if isinstance(data, np.ndarray) else data)
        f.write(b"\0" * (tok_off - f.tell()))
        f.write(tokenizer_blob)


def read_ternix(path: Path):
    """Returns (cfg_u32, cfg_f32, {name: TensorEntry}, mmap). Used by tests and scripts."""
    f = path.open("rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    assert mm[:8] == MAGIC, "bad magic"
    (header_size,) = struct.unpack_from("<Q", mm, 8)
    pos = 16
    u32 = struct.unpack_from(f"<{CONFIG_U32}I", mm, pos); pos += CONFIG_U32 * 4
    f32 = struct.unpack_from(f"<{CONFIG_F32}f", mm, pos); pos += CONFIG_F32 * 4
    entries = {}
    for _ in range(u32[C_NTENSORS]):
        (nl,) = struct.unpack_from("<H", mm, pos); pos += 2
        name = mm[pos:pos + nl].decode(); pos += nl
        dtype, nd = struct.unpack_from("<II", mm, pos); pos += 8
        dims = struct.unpack_from(f"<{nd}Q", mm, pos); pos += 8 * nd
        scale, off, nbytes = struct.unpack_from("<fQQ", mm, pos); pos += 20
        entries[name] = TensorEntry(name, dtype, tuple(dims), scale, off, nbytes)
    assert pos == header_size
    return u32, f32, entries, mm


# ---- safetensors (own minimal reader so bf16 needs no torch) -------------------------------------------------
_ST_DTYPES = {"F32": (np.float32, 4), "F16": (np.float16, 2), "BF16": (np.uint16, 2), "I64": (np.int64, 8),
              "U8": (np.uint8, 1), "I8": (np.int8, 1)}


class SafeTensors:
    def __init__(self, path: Path):
        self._f = path.open("rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        (n,) = struct.unpack_from("<Q", self._mm, 0)
        self.meta = json.loads(self._mm[8:8 + n])
        self._base = 8 + n
        self.meta.pop("__metadata__", None)

    def names(self):
        return list(self.meta)

    def get(self, name: str):
        """Returns (array, dtype_str). BF16 comes back as uint16 bit patterns."""
        m = self.meta[name]
        dt, sz = _ST_DTYPES[m["dtype"]]
        a, b = m["data_offsets"]
        arr = np.frombuffer(self._mm, dtype=dt, count=(b - a) // sz, offset=self._base + a).reshape(m["shape"])
        return arr, m["dtype"]


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)
