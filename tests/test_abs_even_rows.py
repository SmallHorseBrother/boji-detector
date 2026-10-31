from dataclasses import replace

from boji_detector.abdomen.blocks import Block, even_out_rows
from boji_detector.types import STATUS_OK
from synth import SynthSpec
from test_abdomen import _abs


def _column(side: str, spans: list[tuple[float, float]], enclosure: float = 0.8) -> list[Block]:
    return [
        Block(side=side, index=i, top=t, bottom=b, lateral=30.0, enclosure=enclosure, depth=0.03, relief=0.0, sides_closed={})
        for i, (t, b) in enumerate(spans)
    ]


def test_reported_photo_seven_blocks_become_a_six_pack():
    # 4 on the left and 3 on the right; the top left block is the pectoral edge.
    left = _column("image_left", [(0, 40), (42, 70), (72, 100), (102, 130)])
    right = _column("image_right", [(44, 70), (73, 101), (103, 131)])
    kept_left, kept_right = even_out_rows(left, right)
    assert [b.top for b in kept_left] == [42, 72, 102]
    assert [b.index for b in kept_left] == [0, 1, 2]
    assert kept_right == right


def test_works_when_the_right_side_is_longer():
    left = _column("image_left", [(10, 40)])
    right = _column("image_right", [(12, 42), (100, 130)])
    kept_left, kept_right = even_out_rows(left, right)
    assert kept_left == left
    assert [b.top for b in kept_right] == [12]


def test_ties_drop_the_least_enclosed_block():
    left = _column("image_left", [(0, 30), (40, 70)]) + _column("image_left", [(200, 230)], enclosure=0.5)
    left = [replace(b, index=i) for i, b in enumerate(left)]
    left[0] = replace(left[0], top=300, bottom=330)  # no overlap either, but clearer
    right = _column("image_right", [(41, 71), (500, 530)])
    kept_left, _ = even_out_rows(left, right[:1] + right[1:])
    assert 200 not in [b.top for b in kept_left]


def test_even_totals_and_a_lone_block_are_untouched():
    a = _column("image_left", [(0, 30), (40, 70)])
    b = _column("image_right", [(100, 130), (140, 170)])
    assert even_out_rows(a, b) == (a, b)   # offset rows are not removed
    assert even_out_rows(a[:1], []) == (a[:1], [])
    assert even_out_rows([], []) == ([], [])


def test_large_imbalance_only_loses_one_block():
    left = _column("image_left", [(0, 30)])
    right = _column("image_right", [(50, 80), (90, 120), (130, 160), (170, 200)])
    kept_left, kept_right = even_out_rows(left, right)
    assert len(kept_left) + len(kept_right) == 4


def test_symmetric_photo_is_unchanged():
    m = _abs(replace(SynthSpec(), abs_rows=3))
    assert m.status == STATUS_OK
    assert m.values["pack_count"] == 6
