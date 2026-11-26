"""Build the optional "-dur" Piper voice variants used by the SID voice for phone timings.

A "-dur" model is the same Piper .onnx voice with one extra graph output: the duration predictor's
rounded per-phoneme durations (the VITS "w_ceil" tensor, in frames of hop 256 at 22,050 Hz). No
weights change; the file is ~24 bytes larger. server/voice/sidVoice/piper.js reads that output as
"/Ceil_output_0" or "w_ceil" and falls back to the plain model (no consonant-timing model) without it.

Requires the "onnx" Python package (pip install onnx). It is not installed by this project.

Usage, from the repo root:
    python server/voice/tools/make-dur-models.py                 # every models/piper/*.onnx without a -dur sibling
    python server/voice/tools/make-dur-models.py path/to/voice.onnx [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "piper"
PREFERRED = "/Ceil_output_0"


def dur_path(src: Path) -> Path:
    return src.with_name(src.stem + "-dur.onnx")


def make(src: Path, dst: Path | None = None) -> Path:
    import onnx
    from onnx import helper

    dst = dst or dur_path(src)
    model = onnx.load(str(src))
    graph = model.graph
    outputs = {o.name for o in graph.output}
    if PREFERRED in outputs or "w_ceil" in outputs:
        raise SystemExit(f"{src.name} already exposes durations")
    produced = {out for node in graph.node for out in node.output}
    if PREFERRED in produced:
        name = PREFERRED
    else:
        ceils = [n for n in graph.node if n.op_type == "Ceil"]
        if len(ceils) != 1:
            raise SystemExit(f"{src.name}: expected one Ceil node (duration predictor), found {len(ceils)}")
        graph.node.append(helper.make_node("Identity", [ceils[0].output[0]], ["w_ceil"], name="expose_w_ceil"))
        name = "w_ceil"
    source = PREFERRED if name == PREFERRED else ceils[0].output[0]
    out = helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
    try:  # copy the inferred type/shape of the duration tensor when shape inference can find it
        inferred = onnx.shape_inference.infer_shapes(model)
        for vi in inferred.graph.value_info:
            if vi.name == source:
                out.type.CopyFrom(vi.type)
                break
    except Exception:  # noqa: BLE001 - an untyped output still loads in onnxruntime
        pass
    graph.output.append(out)
    onnx.save(model, str(dst))
    return dst


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv] or sorted(p for p in MODEL_DIR.glob("*.onnx") if not p.stem.endswith("-dur"))
    if not targets:
        print(f"No Piper .onnx voices found in {MODEL_DIR}")
        return 1
    for src in targets:
        dst = dur_path(src)
        if dst.exists():
            print(f"skip {src.name}: {dst.name} exists")
            continue
        print(f"wrote {make(src).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
