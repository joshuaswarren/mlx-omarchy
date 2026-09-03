# Bool Any/All boundary table. Prints got/expected per check instead of
# asserting, so wrong values read as table rows. Run with the mlx package
# on the path of the build under test. No override flag on Apple
# hardware; MLX_OMARCHY_ALLOW_NON_APPLE=1 on a development machine.

import sys

import mlx.core as mx

SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 31, 32, 33, 34, 63, 64, 65,
         127, 128, 129, 255, 256, 257, 511, 512, 1000, 4095, 4096, 4097]

total = 0
failed = 0


def host_all(values):
    result = True
    for value in values:
        result = result and value
    return result


def host_any(values):
    result = False
    for value in values:
        result = result or value
    return result


def report(label, got, expected):
    got = [got] if not isinstance(got, (list, tuple)) else list(got)
    expected = list(expected)
    global total, failed
    total += 1
    ok = list(got) == list(expected)
    if not ok:
        failed += 1
    got_text = "".join(str(int(bool(v))) for v in got)
    exp_text = "".join(str(int(bool(v))) for v in expected)
    print(f"{label:<34} got={got_text:<10} exp={exp_text:<10}"
          f" {'ok' if ok else 'FAIL'}")


def probe_size(n):
    ones = mx.array([True] * n)
    zeros = mx.array([False] * n)
    false_in_word0 = [True] * n
    if n >= 2:
        false_in_word0[1] = False
    false_outside = [True] * n
    if n >= 5:
        false_outside[4] = False
    if n >= 9:
        false_outside[8] = False
    true_outside = [False] * n
    if n >= 5:
        true_outside[4] = True

    for name, values in [("ones", ones), ("zeros", zeros),
                         ("false@1", mx.array(false_in_word0)),
                         ("false@4,8", mx.array(false_outside)),
                         ("true@4only", mx.array(true_outside))]:
        report(f"n={n} all({name})",
               mx.all(values).tolist(), [host_all(values.tolist())])
        report(f"n={n} any({name})",
               mx.any(values).tolist(), [host_any(values.tolist())])

    # Axis reduce over a (2, n) grid: row 1 starts mid-word for odd n.
    grid = mx.array([[True] * n, false_outside])
    report(f"n={n} axis1 all", mx.all(grid, axis=1).tolist(),
           [True, host_all(false_outside)])
    report(f"n={n} axis1 any", mx.any(grid, axis=1).tolist(),
           [True, host_any(false_outside)])

    # Typed control: int32 AnyAll carries one element per word.
    if n in (33, 257, 4097):
        ints = [7] * n
        ints[0] = 0
        arr = mx.array(ints)
        report(f"n={n} i32 all", mx.all(arr).tolist(), [False])
        report(f"n={n} i32 any", mx.any(arr).tolist(), [True])

    # Per-position sweep at one representative odd size past word 0:
    # All(F@k) -> True means position k is read truthy though it holds
    # false; Any(T@k) -> False means position k is read falsy though it
    # holds true. Together they map exactly which packed-bool reads the
    # kernel gets wrong.
    if n in (9, 33):
        misread_true = []
        misread_false = []
        for k in range(n):
            f_at_k = [True] * n
            f_at_k[k] = False
            t_at_k = [False] * n
            t_at_k[k] = True
            arr_f = mx.array(f_at_k)
            arr_t = mx.array(t_at_k)
            if bool(mx.all(arr_f).item()):
                misread_true.append(k)
            if not bool(mx.any(arr_t).item()):
                misread_false.append(k)
        print(f"n={n} All(F@k) read-true at k={misread_true}")
        print(f"n={n} Any(T@k) read-false at k={misread_false}")

    # Cross-check through compare_bool.comp: LogicalAnd reads the same
    # word-packed bool inputs through the BYTE_AT constant-shift chain
    # that 959c7a0 proved on this driver. A green LogicalAnd beside a
    # red All pins the defect to the shift-then-mask construct in
    # reduce_general.comp, not to the input layout.
    if n in (33, 65):
        a = mx.array([True] * n)
        land = mx.logical_and(a, a).tolist()
        lor_bad = [i for i, v in enumerate(land) if not v]
        print(f"n={n} logical_and(ones,ones) wrong at {lor_bad}")


def main():
    print("== bool Any/All boundary probe ==")
    for n in SIZES:
        probe_size(n)
    print(f"== {total} checks, {failed} failed ==")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
