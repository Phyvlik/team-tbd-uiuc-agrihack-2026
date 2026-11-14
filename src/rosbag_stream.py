"""Minimal streaming reader for ROS1 bag v2.0 files.

CitrusFarm ``base_*.bag`` files are 2.5–4.2 GB, almost all of it LiDAR. This
reader walks the bag *sequentially* (from a local file or an HTTP stream) and
yields only messages on requested topics, so IMU / GPS / receiver-state data
can be extracted without storing the whole bag. It tolerates truncated files
(stops at the last complete record), which allows inspecting a partial
download. Deserialisation uses the ``rosbags`` ROS1 typestore, with custom
types (e.g. piksi_rtk_msgs) registered from the definitions stored in the bag.
"""
from __future__ import annotations

import bz2
import io
import struct
from typing import BinaryIO, Iterator

OP_MSG, OP_BAG_HEADER, OP_INDEX, OP_CHUNK, OP_CHUNK_INFO, OP_CONNECTION = 0x02, 0x03, 0x04, 0x05, 0x06, 0x07


def _read_exact(f: BinaryIO, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        b = f.read(n - len(buf))
        if not b:
            return None
        buf += b
    return bytes(buf)


def _parse_header(h: bytes) -> dict[str, bytes]:
    out, i = {}, 0
    while i < len(h):
        (ln,) = struct.unpack_from("<I", h, i)
        i += 4
        k, _, v = h[i:i + ln].partition(b"=")
        out[k.decode()] = v
        i += ln
    return out


def _records(f: BinaryIO) -> Iterator[tuple[dict, bytes]]:
    while True:
        b = _read_exact(f, 4)
        if b is None:
            return
        (hl,) = struct.unpack("<I", b)
        h = _read_exact(f, hl)
        if h is None:
            return
        b = _read_exact(f, 4)
        if b is None:
            return
        (dl,) = struct.unpack("<I", b)
        d = _read_exact(f, dl)
        if d is None:
            return
        yield _parse_header(h), d


def _decompress(comp: str, data: bytes, size: int) -> bytes:
    if comp == "none":
        return data
    if comp == "bz2":
        return bz2.decompress(data)
    if comp == "lz4":
        import lz4.frame  # optional dependency
        return lz4.frame.decompress(data)
    raise ValueError(f"unsupported chunk compression {comp}")


class BagStream:
    """Iterate (topic, msgtype, t_record, raw_bytes, connection_header) for wanted topics."""

    def __init__(self, f: BinaryIO, topics: set[str] | None = None):
        self.f, self.topics = f, topics
        self.connections: dict[int, dict] = {}
        self.bytes_hint = 0
        magic = _read_exact(f, 13)
        if magic != b"#ROSBAG V2.0\n":
            raise ValueError(f"not a ROS1 v2.0 bag: {magic!r}")

    def _conn(self, hdr: dict, data: bytes):
        cid = struct.unpack("<I", hdr["conn"])[0]
        if cid not in self.connections:
            ch = _parse_header(data)
            self.connections[cid] = {"topic": hdr["topic"].decode(), "type": ch["type"].decode(),
                                     "msgdef": ch.get("message_definition", b"").decode(errors="replace")}

    def __iter__(self):
        for hdr, data in _records(self.f):
            op = hdr["op"][0]
            if op == OP_CONNECTION:
                self._conn(hdr, data)
            elif op == OP_CHUNK:
                comp = hdr["compression"].decode()
                size = struct.unpack("<I", hdr["size"])[0]
                inner = io.BytesIO(_decompress(comp, data, size))
                for h2, d2 in _records(inner):
                    op2 = h2["op"][0]
                    if op2 == OP_CONNECTION:
                        self._conn(h2, d2)
                    elif op2 == OP_MSG:
                        yield from self._msg(h2, d2)
            elif op == OP_MSG:
                yield from self._msg(hdr, data)

    def _msg(self, hdr, data):
        cid = struct.unpack("<I", hdr["conn"])[0]
        c = self.connections.get(cid)
        if c is None or (self.topics and c["topic"] not in self.topics):
            return
        sec, nsec = struct.unpack("<II", hdr["time"])
        yield c["topic"], c["type"], sec + nsec * 1e-9, data, c


class Ros1Decoder:
    """Deserialise ROS1 messages, registering custom types from bag definitions."""

    def __init__(self):
        from rosbags.typesys import Stores, get_typestore
        self.ts = get_typestore(Stores.ROS1_NOETIC)
        self.known = set(self.ts.types)

    def decode(self, raw: bytes, conn: dict):
        from rosbags.typesys import get_types_from_msg
        t = conn["type"]
        if "/msg/" not in t:  # ROS1 'pkg/Type' -> rosbags 'pkg/msg/Type'
            t = t.replace("/", "/msg/", 1)
        if t not in self.known:
            self.ts.register(get_types_from_msg(conn["msgdef"], t))
            self.known = set(self.ts.types)
        return self.ts.deserialize_ros1(raw, t)
