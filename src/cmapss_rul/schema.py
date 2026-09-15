"""Inference-safe schema constants for NASA C-MAPSS files."""

CMAPSS_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{index}" for index in range(1, 4)]
    + [f"sensor_{index}" for index in range(1, 22)]
)
