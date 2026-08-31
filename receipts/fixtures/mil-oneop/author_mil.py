"""Author one explicit fp16 elementwise-add MIL program and serialize it to MIL text.

The program is hand-authored through coremltools' MIL Builder with explicit
program syntax (no Keras/torch/numpy conversion). coremltools is used only as
the MIL text serializer.
"""

import sys

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types

SHAPE = (1, 64, 1, 1)


def build(opset):
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=SHAPE, dtype=types.fp16),
            mb.TensorSpec(shape=SHAPE, dtype=types.fp16),
        ],
        opset_version=opset,
    )
    def add_one_op(a, b):
        return mb.add(x=a, y=b)

    return add_one_op


def main():
    text = None
    used = None
    for name in ("iOS18", "iOS17", "iOS16"):
        opset = getattr(ct.target, name)
        prog = build(opset)
        rendered = str(prog)
        header = rendered.split("{", 1)[0].strip()
        print(f"# probe {name}: header={header!r}")
        if header == "program(1.3)":
            text, used = rendered, name
            break
    if text is None:
        print("# no opset produced program(1.3)")
        return 1
    with open("model.mil", "w") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
    print(f"# selected opset={used}; wrote model.mil ({len(text)} bytes)")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
