"""Per-block measurements between consecutive horizontal grooves.

Enclosure is the share of a block's outline (top groove, bottom groove,
linea alba, lateral edge) where a valley is actually present. It is the
continuous counterpart of T-Muscle Detector's "open angle":
open_angle = 360 * (1 - enclosure).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .grooves import GROOVE_FLOOR, Alba, Groove, _sample
from .workspace import Workspace

BLOCK_HEIGHT = (0.10, 0.45)   # allowed block height, fraction of the waist
MEDIAL_GAP = 0.04             # outline starts this far from the alba
EDGE_REACH = 2                # pixels searched either side of an outline sample
ENCLOSURE_THRESHOLD = GROOVE_FLOOR
DEPTH_REF = 0.05              # groove response treated as "fully deep"
MIN_BLOCK_SKIN = 0.7
# The rectus abdominis has at most four rows per side (an "8-pack"). Extra rows
# come from the lower pectoral edge, the rib line or folds below the navel.
MAX_ROWS_PER_SIDE = 4


@dataclass(frozen=True)
class Block:
    side: str
    index: int
    top: float
    bottom: float
    lateral: float
    enclosure: float
    depth: float
    relief: float
    sides_closed: dict

    @property
    def height(self) -> float:
        return self.bottom - self.top


def _edge_hits(field: np.ndarray, ws: Workspace, xs: np.ndarray, ys: np.ndarray, across_rows: bool):
    """Best response within EDGE_REACH of each outline sample."""
    offsets = np.arange(-EDGE_REACH, EDGE_REACH + 1)
    if across_rows:
        rows = np.rint(ys)[:, None].astype(int) + offsets[None, :]
        cols = np.broadcast_to(xs[:, None], rows.shape)
    else:
        rows = np.broadcast_to(np.rint(ys)[:, None].astype(int), (len(ys), offsets.size))
        cols = xs[:, None] + offsets[None, :]
    vals, ok = _sample(field, ws.valid, rows, cols)
    return np.where(ok, vals, 0.0).max(axis=1)


def _groove_segment(groove: Groove, alba: Alba, sign: int, lateral: float, waist: float) -> np.ndarray:
    pts = groove.points
    offset = sign * (pts[:, 0] - alba.x(pts[:, 1]))
    keep = (offset >= MEDIAL_GAP * waist) & (offset <= lateral)
    return pts[keep]


def measure_block(
    ws: Workspace, alba: Alba, side: str, sign: int, index: int,
    upper: Groove, lower: Groove, lateral: float,
) -> Block | None:
    w = ws.waist
    top_pts = _groove_segment(upper, alba, sign, lateral, w)
    bottom_pts = _groove_segment(lower, alba, sign, lateral, w)
    if len(top_pts) < 2 or len(bottom_pts) < 2:
        return None
    top, bottom = float(top_pts[:, 1].mean()), float(bottom_pts[:, 1].mean())
    rows = np.arange(int(np.ceil(top)) + EDGE_REACH, int(bottom) - EDGE_REACH + 1)
    if rows.size < 3:
        return None

    inner_rows = rows[(rows >= top + 0.25 * (bottom - top)) & (rows <= bottom - 0.25 * (bottom - top))]
    inner_x = alba.x(inner_rows)[:, None] + sign * np.linspace(0.25, 0.75, 9)[None, :] * lateral
    relief_vals, relief_ok = _sample(ws.relief, ws.valid, inner_rows[:, None], inner_x)
    body_x = alba.x(rows)[:, None] + sign * np.linspace(MEDIAL_GAP * w, lateral, 12)[None, :]
    _, body_ok = _sample(ws.relief, ws.surface, rows[:, None], body_x)
    if body_ok.mean() < MIN_BLOCK_SKIN or not relief_ok.any():
        return None

    edges = {
        "top": _edge_hits(ws.horizontal, ws, top_pts[:, 0], top_pts[:, 1], across_rows=True),
        "bottom": _edge_hits(ws.horizontal, ws, bottom_pts[:, 0], bottom_pts[:, 1], across_rows=True),
        "medial": _edge_hits(ws.vertical, ws, alba.x(rows), rows, across_rows=False),
        "lateral": _edge_hits(ws.vertical, ws, alba.x(rows) + sign * lateral, rows, across_rows=False),
    }
    lengths = {"top": lateral, "bottom": lateral, "medial": bottom - top, "lateral": bottom - top}
    closed = {k: float(np.mean(v >= ENCLOSURE_THRESHOLD)) for k, v in edges.items()}
    enclosure = sum(closed[k] * lengths[k] for k in edges) / sum(lengths.values())
    outline = np.concatenate(list(edges.values()))
    depth = float(np.mean(np.minimum(outline, 2 * DEPTH_REF)))

    rim_x = np.concatenate([top_pts[:, 0], bottom_pts[:, 0]])
    rim_y = np.concatenate([top_pts[:, 1], bottom_pts[:, 1]])
    rim_vals, rim_ok = _sample(ws.relief, ws.valid, np.rint(rim_y).astype(int), rim_x)
    rim = float(rim_vals[rim_ok].mean()) if rim_ok.any() else 0.0
    relief = float(relief_vals[relief_ok].mean()) - rim
    return Block(
        side=side, index=index, top=top, bottom=bottom, lateral=lateral,
        enclosure=float(enclosure), depth=depth, relief=relief, sides_closed=closed,
    )


def side_blocks(
    ws: Workspace, alba: Alba, side: str, sign: int, grooves: list[Groove], lateral: float
) -> list[Block]:
    lo, hi = BLOCK_HEIGHT[0] * ws.waist, BLOCK_HEIGHT[1] * ws.waist
    blocks: list[Block] = []
    for upper, lower in zip(grooves, grooves[1:]):
        if not lo <= lower.row - upper.row <= hi:
            continue
        block = measure_block(ws, alba, side, sign, len(blocks), upper, lower, lateral)
        if block is not None:
            blocks.append(block)
    return _best_rows(blocks)


def _best_rows(blocks: list[Block], limit: int = MAX_ROWS_PER_SIDE) -> list[Block]:
    """Keep the run of consecutive rows that is most clearly enclosed."""
    if len(blocks) <= limit:
        return blocks
    start = max(range(len(blocks) - limit + 1), key=lambda i: sum(b.enclosure for b in blocks[i:i + limit]))
    return [replace(b, index=i) for i, b in enumerate(blocks[start:start + limit])]


def definition_index(blocks: list[Block], best_strength: float, blocks_per_side: float) -> float:
    """0..1 summary of how clearly the abdominal blocks are separated."""
    if not blocks:
        return 0.35 * min(1.0, best_strength / DEPTH_REF)
    per_block = [0.65 * b.enclosure + 0.35 * min(1.0, b.depth / DEPTH_REF) for b in blocks]
    count_factor = min(1.0, 0.7 + 0.1 * blocks_per_side)
    return float(np.mean(per_block) * count_factor)
