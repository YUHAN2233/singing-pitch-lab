"""Local reader for the classic CTENFDAM NCM container.

The supplied audio is never sent to a converter service. Unsupported variants
and malformed lengths fail explicitly. This module only handles the container;
ffmpeg still validates and decodes the resulting MP3 or FLAC stream.
"""
from pathlib import Path
import struct

MAGIC = b"CTENFDAM"
CORE_KEY = bytes.fromhex("687A4852416D736F356B496E62617857")


def _xtime(value):
    return ((value << 1) ^ (0x11B if value & 0x80 else 0)) & 255


def _multiply(a, b):
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        a = _xtime(a)
        b >>= 1
    return result


def _sbox_item(value):
    inverse = next((candidate for candidate in range(1, 256) if _multiply(value, candidate) == 1), 0)
    rotate = lambda shift: ((inverse << shift) | (inverse >> (8 - shift))) & 255
    return inverse ^ rotate(1) ^ rotate(2) ^ rotate(3) ^ rotate(4) ^ 0x63


SBOX = bytes(_sbox_item(value) for value in range(256))
INV_SBOX = bytes(SBOX.index(value) for value in range(256))


def _aes128_decrypt(block, key):
    if len(block) != 16 or len(key) != 16:
        raise ValueError("NCM 密钥块长度无效")
    expanded = list(key)
    rcon = 1
    while len(expanded) < 176:
        temp = expanded[-4:]
        if len(expanded) % 16 == 0:
            temp = [SBOX[item] for item in temp[1:] + temp[:1]]
            temp[0] ^= rcon
            rcon = _xtime(rcon)
        start = len(expanded)
        expanded.extend(expanded[start - 16 + i] ^ temp[i] for i in range(4))
    state = list(block)
    def add_key(round_number):
        for i in range(16):
            state[i] ^= expanded[round_number * 16 + i]
    def shift_rows():
        for row in range(1, 4):
            positions = [row + 4 * column for column in range(4)]
            values = [state[position] for position in positions]
            for column, position in enumerate(positions):
                state[position] = values[(column - row) % 4]
    add_key(10)
    for round_number in range(9, -1, -1):
        shift_rows()
        state[:] = [INV_SBOX[item] for item in state]
        add_key(round_number)
        if round_number:
            for column in range(4):
                offset = column * 4
                values = state[offset:offset + 4]
                for row, coefficients in enumerate(((14, 11, 13, 9), (9, 14, 11, 13), (13, 9, 14, 11), (11, 13, 9, 14))):
                    state[offset + row] = 0
                    for coefficient, value in zip(coefficients, values):
                        state[offset + row] ^= _multiply(coefficient, value)
    return bytes(state)


def _read_exact(source, size, limit):
    if size > limit:
        raise ValueError("NCM 区块长度超出安全限制")
    data = source.read(size)
    if len(data) != size:
        raise ValueError("NCM 文件被截断")
    return data


def _length(source):
    return struct.unpack("<I", _read_exact(source, 4, 4))[0]


def _unpad(data):
    if not data or len(data) % 16:
        raise ValueError("NCM 密钥密文无效")
    amount = data[-1]
    if amount < 1 or amount > 16 or data[-amount:] != bytes([amount]) * amount:
        raise ValueError("NCM 密钥填充无效")
    return data[:-amount]


def _audio_kind(head):
    if head.startswith(b"fLaC"):
        return "flac"
    if head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 255 and head[1] & 0xE0 == 0xE0):
        return "mp3"
    raise ValueError("NCM 内部音频不是可识别的 MP3 或 FLAC")


def extract_ncm(source_path: Path, output_path: Path, cancel=lambda: False):
    """Decode one local NCM into a temporary audio file; return detected codec."""
    with source_path.open("rb") as source:
        if _read_exact(source, 8, 8) != MAGIC:
            raise ValueError("不是受支持的 NCM 容器（文件头不符）")
        _read_exact(source, 2, 2)  # NCM header flags vary; key block validates the layout.
        key_size = _length(source)
        if key_size == 0 or key_size > 4096 or key_size % 16:
            raise ValueError("NCM 密钥长度无效")
        encrypted_key = bytes(value ^ 0x64 for value in _read_exact(source, key_size, 4096))
        plain_key = _unpad(b"".join(_aes128_decrypt(encrypted_key[i:i + 16], CORE_KEY)
                                     for i in range(0, key_size, 16)))
        prefix = b"neteasecloudmusic"
        if not plain_key.startswith(prefix) or len(plain_key) <= len(prefix):
            raise ValueError("无法读取 NCM 音频密钥；文件版本可能不受支持")
        key = plain_key[len(prefix):]
        box = list(range(256)); last = 0
        for i in range(256):
            last = (box[i] + last + key[i % len(key)]) & 255
            box[i], box[last] = box[last], box[i]
        metadata_size = _length(source)
        _read_exact(source, metadata_size, 2_000_000)
        _read_exact(source, 9, 9)  # CRC32 + two reserved bytes + artwork length follows.
        image_size = struct.unpack("<I", _read_exact(source, 4, 4))[0]
        _read_exact(source, image_size, 20_000_000)
        position = source.tell()
        encrypted_head = _read_exact(source, 16, 16)
        def mask(index):
            i = (index + 1) & 255
            return box[(box[i] + box[(box[i] + i) & 255]) & 255]
        head = bytes(value ^ mask(i) for i, value in enumerate(encrypted_head))
        kind = _audio_kind(head)
        source.seek(position)
        index = 0
        try:
            with output_path.open("wb") as target:
                while chunk := source.read(131072):
                    if cancel():
                        raise InterruptedError("已取消 NCM 导入")
                    decoded = bytearray(chunk)
                    for j in range(len(decoded)):
                        decoded[j] ^= mask(index + j)
                    target.write(decoded)
                    index += len(decoded)
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise
    return kind
