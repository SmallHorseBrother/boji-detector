# Contributing

```bash
pip install -e ".[dev]"
pytest                      # synthetic tests, no models or photos needed
boji fetch-models
python scripts/robustness.py path/to/your/photo.jpg   # stability check, stays local
```

Rules:

- **Never commit photos of people**, including test inputs and overlays. Use
  `tests/synth.py` to draw the structure you need, and keep real photos in the
  ignored `samples/` or `out/` folders.
- Every metric reports raw `values`, pixel `geometry` and `reasons`; a score
  alone is not accepted. New reason codes need Chinese and English text in
  `boji_detector/messages.py` (a test enforces this).
- Express sizes as fractions of the waist or limb length, never fixed pixels.
- Scoring curves are placeholders until calibrated; change them only together
  with the data that justifies the change.
- New regions: see "扩展一个新区域" in `README.md`.
