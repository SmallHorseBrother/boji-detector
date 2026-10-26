from dataclasses import replace

import pytest

from boji_detector.abdomen.blocks import MAX_ROWS_PER_SIDE, Block, _best_rows
from boji_detector.types import STATUS_OK
from synth import SynthSpec
from test_abdomen import _abs


def _block(index: int, enclosure: float) -> Block:
    return Block(
        side="image_left", index=index, top=index * 10.0, bottom=index * 10.0 + 9, lateral=30.0,
        enclosure=enclosure, depth=0.03, relief=0.0, sides_closed={},
    )


def test_keeps_short_lists_untouched():
    blocks = [_block(i, 0.9) for i in range(3)]
    assert _best_rows(blocks) == blocks


def test_keeps_the_most_enclosed_run_of_four_and_reindexes():
    # A weak extra row on top (pectoral edge) and a strong 4-row run below it.
    blocks = [_block(0, 0.2)] + [_block(i, 0.9) for i in range(1, 5)]
    kept = _best_rows(blocks)
    assert len(kept) == MAX_ROWS_PER_SIDE
    assert [b.top for b in kept] == [10.0, 20.0, 30.0, 40.0]
    assert [b.index for b in kept] == [0, 1, 2, 3]


@pytest.mark.parametrize("rows", [5, 6])
def test_never_reports_more_than_eight_packs(rows):
    m = _abs(replace(SynthSpec(), abs_rows=rows))
    assert m.status == STATUS_OK
    assert m.values["blocks_image_left"] <= MAX_ROWS_PER_SIDE
    assert m.values["blocks_image_right"] <= MAX_ROWS_PER_SIDE
    assert m.values["pack_count"] <= 2 * MAX_ROWS_PER_SIDE
