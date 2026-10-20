"""See the communication XLA inserted: count collectives in compiled HLO, per mesh axis.

    python train.py --cpu_devices 8 --dp 2 --pp 2 --tp 2 --dry_run --show_collectives

prints something like

    collective           mesh axes     ops   bytes (static)
    all-reduce           tp             17      1.31 MiB
    all-reduce           dp             12    330.00 KiB
    collective-permute   pp              2    256.00 KiB

Each collective's replica groups (or source/target pairs) are mapped back to
mesh coordinates; the "mesh axes" column lists the axes along which members of
a group differ. Counts are static: an op inside a loop (layer scan, gradient
accumulation, pipeline iterations) is counted once even though it runs many
times.
"""
from __future__ import annotations

import collections
import re
from collections.abc import Iterable

import numpy as np
from jax.sharding import Mesh

COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "collective-permute", "all-to-all")

_OP_RE = re.compile(r"\b(" + "|".join(COLLECTIVES) + r")(-start)?\(")
_GROUPS_LIST_RE = re.compile(r"replica_groups=\{((?:\{[\d,\s]*\},?\s*)*)\}")
_GROUPS_MESH_RE = re.compile(r"replica_groups=mesh\[([^\]]*)\]\s*\{([^}]*)\}")
_GROUPS_IOTA_RE = re.compile(r"replica_groups=\[([\d,]+)\]<=\[([\d,]+)\](?:T\(([\d,]+)\))?")
_PAIRS_RE = re.compile(r"source_target_pairs=\{((?:\{\d+,\d+\},?)*)\}")
_TYPE_RE = re.compile(r"\b(pred|s8|u8|s16|u16|s32|u32|s64|u64|bf16|f16|f32|f64)\[([\d,]*)\]")
_BYTES = {"pred": 1, "s8": 1, "u8": 1, "s16": 2, "u16": 2, "bf16": 2, "f16": 2,
          "s32": 4, "u32": 4, "f32": 4, "s64": 8, "u64": 8, "f64": 8}


def _parse_groups(line: str) -> list[list[int]]:
    m = _GROUPS_MESH_RE.search(line)
    if m:  # newer XLA: mesh['a'=2,'b'=4] {'b'} = groups along the listed axes of a device grid
        dims = re.findall(r"'([^']+)'=(\d+)", m.group(1))
        names, sizes = [n for n, _ in dims], [int(s) for _, s in dims]
        selected = [names.index(a) for a in re.findall(r"'([^']+)'", m.group(2)) if a in names]
        ids = np.arange(int(np.prod(sizes))).reshape(sizes)
        rest = [i for i in range(len(sizes)) if i not in selected]
        group_size = int(np.prod([sizes[i] for i in selected])) if selected else 1
        return ids.transpose(rest + selected).reshape(-1, group_size).tolist()
    m = _GROUPS_IOTA_RE.search(line)
    if m:  # iota format: [G,S]<=[dims]T(perm)
        out_shape = [int(x) for x in m.group(1).split(",")]
        dims = [int(x) for x in m.group(2).split(",")]
        ids = np.arange(int(np.prod(dims))).reshape(dims)
        if m.group(3):
            ids = ids.transpose([int(x) for x in m.group(3).split(",")])
        return ids.reshape(out_shape[0], -1).tolist()
    m = _GROUPS_LIST_RE.search(line)
    if m:
        return [[int(x) for x in g.split(",") if x.strip()] for g in re.findall(r"\{([\d,\s]*)\}", m.group(1))]
    m = _PAIRS_RE.search(line)
    if m:
        return [[int(a), int(b)] for a, b in re.findall(r"\{(\d+),(\d+)\}", m.group(1))]
    return []


def _axes_for_groups(groups: Iterable[list[int]], mesh: Mesh) -> tuple[str, ...]:
    shape = mesh.devices.shape
    varying = set()
    for g in groups:
        if len(g) < 2 or max(g) >= mesh.size:
            continue
        coords = np.array([np.unravel_index(i, shape) for i in g])
        for axis, name in enumerate(mesh.axis_names):
            if len(set(coords[:, axis])) > 1:
                varying.add(name)
    return tuple(n for n in mesh.axis_names if n in varying)


def _result_bytes(line: str) -> int:
    lhs = line.split("=", 1)[1] if "=" in line else line
    head = lhs.split("(", 1)[0] if not lhs.strip().startswith("(") else lhs.split(")", 1)[0]
    total = 0
    for dtype, dims in _TYPE_RE.findall(head):
        n = int(np.prod([int(d) for d in dims.split(",") if d])) if dims else 1
        total += n * _BYTES[dtype]
    return total


def count_collectives(hlo_text: str, mesh: Mesh) -> dict[tuple[str, tuple[str, ...]], dict[str, int]]:
    """{(op, mesh_axes): {"count": n, "bytes": b}} for every collective in the HLO text."""
    stats: dict = collections.defaultdict(lambda: {"count": 0, "bytes": 0})
    for line in hlo_text.splitlines():
        m = _OP_RE.search(line)
        if not m or "-done(" in line:
            continue
        key = (m.group(1), _axes_for_groups(_parse_groups(line), mesh))
        stats[key]["count"] += 1
        stats[key]["bytes"] += _result_bytes(line)
    return dict(stats)


def _human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} GiB"


def collectives_table(stats: dict) -> str:
    if not stats:
        return "No collectives: this layout needs no communication (single device or pure replication)."
    rows = [("collective", "mesh axes", "ops", "bytes (static)")]
    for (op, axes), s in sorted(stats.items(), key=lambda kv: (COLLECTIVES.index(kv[0][0]), kv[0][1])):
        rows.append((op, "+".join(axes) or "-", str(s["count"]), _human_bytes(s["bytes"])))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = [f"{r[0]:<{widths[0]}}   {r[1]:<{widths[1]}}   {r[2]:>{widths[2]}}   {r[3]:>{widths[3]}}" for r in rows]
    lines.insert(1, "-" * len(lines[0]))
    lines.append("(static counts: an op inside a loop is listed once but runs on every iteration)")
    return "\n".join(lines)
