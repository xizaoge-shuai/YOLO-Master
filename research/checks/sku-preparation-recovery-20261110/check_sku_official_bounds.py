"""CPU-only source-grounded SKU boundary contracts; no image/model training."""

import argparse
import ast
import csv
import importlib.util
import json
import math
import os
import tempfile
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--raw", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
a = parser.parse_args()
here = Path(__file__).parent
source = here / "prepare_sku_official_bounds.py"
spec = importlib.util.spec_from_file_location("sku_prepare_boundary_fix", source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ref = here / "official_csv_generator.py.reference"
assert (
    m.sha256(ref) == "44284158a427a46eb26be6a05454af105c1ac95cb3f7a39374897386528c3b08"
)
tree = ast.parse(ref.read_text())
selected = ast.Module(
    body=[
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("_parse", "_read_annotations")
    ],
    type_ignores=[],
)
ns = {"os": os}
exec(compile(selected, str(ref), "exec"), ns)  # noqa: S102 -- pinned official reference AST, functions only
official = ns["_read_annotations"]
tick = time.monotonic()
splits = {}
for split, expected in m.SKU_COUNTS.items():
    p = a.raw / "annotations" / f"annotations_{split}.csv"
    images = set()
    rows = adjusted = oob = 0
    with p.open(newline="") as stream:
        for index, row in enumerate(csv.reader(stream), 1):
            filename = row[0]
            images.add(filename)
            w, h = map(int, row[6:8])
            raw = list(map(float, row[1:5]))
            assert all(math.isfinite(v) for v in raw)
            got = m.official_sku_boundary_box(raw, w, h)
            parsed = official(
                [row],
                {"object": 0},
                "/official",
                {os.path.join("/official", filename): {}},
            )
            item = parsed[os.path.join("/official", filename)][0]
            want = [item[k] for k in ("x1", "y1", "x2", "y2")]
            assert got == want, (split, index, got, want)
            rows += 1
            adjusted += got != raw
            oob += not (0 <= raw[0] < raw[2] <= w and 0 <= raw[1] < raw[3] <= h)
    assert len(images) == expected
    splits[split] = {
        "images": len(images),
        "objects": rows,
        "official_boundary_adjustments": adjusted,
        "raw_out_of_bounds": oob,
        "dropped_objects": 0,
        "source_csv_sha256": m.sha256(p),
    }
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "annotations.csv"
    p.write_text(
        "train_0.jpg,2850,2290,3027,2732,object,3024,3024\ntrain_0.jpg,0,0,3024,3024,object,3024,3024\n"
    )
    parsed = m.sku_rows(p)["train_0.jpg"]["objects"]
    assert len(parsed) == 2
    assert parsed[0]["bbox_sku_csv_original_xyxy"] == [2850.0, 2290.0, 3027.0, 2732.0]
    assert parsed[0]["bbox_xyxy0_half_open"] == [2850.0, 2290.0, 3023.0, 2732.0]
    assert parsed[1]["bbox_xyxy0_half_open"] == [0.0, 0.0, 3023.0, 3023.0]
    assert parsed[0]["source_csv_line"] == 1
    for invalid in [
        "bad.jpg,0,0,0,1,object,5,5",
        "bad.jpg,-1,0,2,2,object,5,5",
        "bad.jpg,nan,0,2,2,object,5,5",
        "bad.jpg,0,5,2,6,object,5,5",
    ]:
        p.write_text(invalid + "\n")
        try:
            m.sku_rows(p)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid annotation failure")
report = {
    "status": "PASS",
    "scope": "All official 1,730,996 rows compared to actual author _read_annotations via AST; CPU only",
    "source_sha256": m.sha256(source),
    "original_source_sha256": "4bb13602c3c369554a334a09a9b5e4fcd424bfa39f01c90665455c03f379379b",
    "official_reference_url": m.SKU_BOUNDARY_SOURCE,
    "official_reference_sha256": m.sha256(ref),
    "policy": m.SKU_BOUNDARY_POLICY,
    "splits": splits,
    "preservation_contracts": "Original CSV bbox and line retained; every object kept; nonfinite/invalid/negative/degenerate still fail",
    "seconds": time.monotonic() - tick,
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
