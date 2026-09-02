#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Generate the omarchy primitive compatibility matrix.

Reads the backend sources and the test suite, derives one row per
primitive entry in overlay/mlx/backend/omarchy/primitives.cpp, and
writes a markdown document to stdout. Redirect into
docs/compatibility-matrix.md. `--json-out PATH` also writes the
coverage badge data in shields.io endpoint format (docs/coverage.json).

Sources (read-only):
  overlay/mlx/backend/omarchy/primitives.cpp  primitive entries and gates
  overlay/mlx/backend/omarchy/compute.h       ComputeKernel variants
  overlay/mlx/backend/omarchy/copy.cpp        engine kernels behind copies
  overlay/tests/omarchy/**/*.cpp              TEST_CASE bodies (anchor evidence)
  .work/mlx/mlx/primitives.h                  upstream primitive classes
  .work/mlx/mlx/fast_primitives.h             upstream fast:: primitives
  .work/mlx/mlx/{,io/,distributed/}*.cpp      op -> primitive construction map
  .work/mlx/mlx/distributed/primitives.h      upstream distributed:: primitives
  .work/mlx/mlx/backend/gpu/primitives.cpp    shared GPU eval_gpu definitions
"""

import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PRIMITIVES = ROOT / "overlay/mlx/backend/omarchy/primitives.cpp"
COMPUTE_H = ROOT / "overlay/mlx/backend/omarchy/compute.h"
COPY_CPP = ROOT / "overlay/mlx/backend/omarchy/copy.cpp"
TEST_GLOB = "overlay/tests/omarchy/**/*.cpp"

UPSTREAM_H = ROOT / ".work/mlx/mlx/primitives.h"
UPSTREAM_FAST_H = ROOT / ".work/mlx/mlx/fast_primitives.h"
UPSTREAM_DIST_H = ROOT / ".work/mlx/mlx/distributed/primitives.h"

# Classes in the upstream headers that are dispatch bases, not
# primitives themselves.
UPSTREAM_BASES = {"Primitive", "UnaryPrimitive", "Custom", "DistPrimitive"}

UPSTREAM_CLASS = re.compile(
    r"^class (?:MLX_API\s+)?(\w+) : public (\w+) \{", re.M)

# Helpers whose fragments are generic engine limits, not per-primitive
# constraints. They appear once in the footnote instead of every row.
GENERIC_HELPERS = {"checked_u32", "checked_item_offset"}

NAME_VARS = {"name", "tag", "operation_name", "error_name"}

DTYPE_LABELS = {
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
    "int32": "i32",
    "uint32": "u32",
    "int64": "i64",
    "bool_": "bool",
}

VARIANT_TOKENS = {
    "F32": "f32",
    "F16": "f16",
    "BF16": "bf16",
    "I32": "i32",
    "U32": "u32",
    "Bool": "bool",
}

# Upstream sources whose function bodies construct primitives. Anchor
# resolution maps every op call in a TEST_CASE body through these
# bodies (see parse_op_constructions), so op-to-primitive resolution
# is derived from the sources instead of a hand-maintained table.
CONSTRUCTION_GLOBS = [
    ".work/mlx/mlx/*.cpp",
    ".work/mlx/mlx/io/*.cpp",
    ".work/mlx/mlx/distributed/*.cpp",
]

DTYPE_ORDER = ["f32", "f16", "bf16", "i32", "u32", "i64", "bool"]

UPSTREAM_GPU_PRIMITIVES = ROOT / ".work/mlx/mlx/backend/gpu/primitives.cpp"


def parse_shared_gpu_primitives():
    """Names upstream's shared backend/gpu layer gives an eval_gpu to.

    A primitive with no omarchy entry resolves through this shared layer,
    which drives the omarchy copy engine via copy_gpu / fill_gpu /
    reshape_gpu / concatenate_gpu. If the layer is missing an eval_gpu the
    omarchy build would not link, so presence here is the honest signal
    that upstream handles the primitive generically on this backend.
    """
    text = strip_comments(UPSTREAM_GPU_PRIMITIVES.read_text())
    return set(re.findall(r"void (\w+)::eval_gpu\(", text))




def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def balanced(text, open_index):
    """Return the text inside the parenthesis that opens at open_index."""
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:index]
    return text[open_index + 1:]


def function_body(lines, start):
    """Return the brace-balanced body text of the function at lines[start]."""
    joined = "\n".join(lines[start:start + 200])
    open_index = joined.find("{")
    depth = 0
    for index in range(open_index, len(joined)):
        char = joined[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return joined[open_index + 1:index]
    return joined[open_index + 1:]


def git(*args):
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def mask_noncode(text):
    """Blank comments and string/char literal contents, keeping length.

    Offsets in the result address the original text, so structural
    parsing never trips over braces or parens inside literals, and
    line numbers survive. Quote characters themselves stay.
    """
    out = list(text)
    length = len(text)
    index = 0
    while index < length:
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = length if end == -1 else end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = length if end == -1 else end + 2
        elif text[index] in "\"'":
            quote = text[index]
            end = index + 1
            while end < length and text[end] != quote:
                end += 2 if text[end] == "\\" else 1
            end = min(end + 1, length)
            for offset in range(index + 1, end - 1):
                if out[offset] != "\n":
                    out[offset] = " "
            index = end
            continue
        else:
            index += 1
            continue
        for offset in range(index, end):
            if out[offset] != "\n":
                out[offset] = " "
        index = end
    return "".join(out)


def block_end(masked, open_index):
    """Index of the brace closing the block opened at open_index."""
    depth = 0
    for index in range(open_index, len(masked)):
        char = masked[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(masked) - 1


NAMESPACE_TAIL = re.compile(r"\bnamespace\s+([A-Za-z_][\w:]*)\s*$")
NAMESPACE_ANON = re.compile(r"\bnamespace\s*$")
SCOPE_KEYWORD = re.compile(r"\b(struct|class|enum|union)\b")
CALL_SITE = re.compile(r"([A-Za-z_]\w*)\s*\(")
MAKE_SHARED = re.compile(r"make_shared<([A-Za-z_][\w:]*)>\s*\(")


def function_name(header):
    """Name before the balanced trailing parameter list of a definition.

    The last ')' in a definition header closes the parameter list, and
    its matching '(' is found by scanning back, so parens inside return
    types and parameters (std::function<...>) cannot hijack the name.
    Member methods keep their class prefix (Add::vjp), so upstream rule
    methods never merge with the free op functions the tests call.
    """
    close = header.rfind(")")
    if close == -1:
        return None
    depth = 0
    for index in range(close, -1, -1):
        char = header[index]
        if char == ")":
            depth += 1
        elif char == "(":
            depth -= 1
            if depth == 0:
                member = re.search(
                    r"([A-Za-z_]\w*)\s*::\s*([A-Za-z_]\w*)\s*$",
                    header[:index])
                if member:
                    return "::".join(member.groups())
                match = re.search(r"([A-Za-z_]\w*)\s*$", header[:index])
                return match.group(1) if match else None
    return None


MAKE_SHARED = re.compile(r"make_shared<([A-Za-z_][\w:]*)>\s*\(")

# Primitives live in mlx::core (no namespace in the matrix rows),
# mlx::core::fast, and mlx::core::distributed; the namespace walk only
# sees the tail, so the tail maps to the matrix namespace.
NAMESPACE_ALIAS = {"core": ""}


def next_block(masked, start, end):
    """First '{' outside any parenthesis in masked[start:end].

    A brace inside a parameter list (a default argument such as
    StreamOrDevice s = {}) is not the function body opener.
    """
    depth = 0
    for index in range(start, end):
        char = masked[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "{" and depth == 0:
            return index
    return -1


def attribute_functions(masked):
    """(namespace, function name, body) for every function-definition block."""
    out = []

    def walk(start, end, ns):
        index = start
        while index < end:
            brace = next_block(masked, index, end)
            if brace == -1:
                return
            header = masked[index:brace].rstrip()
            close = block_end(masked, brace)
            tail = NAMESPACE_TAIL.search(header)
            if tail:
                tail_ns = tail.group(1).rpartition("::")[2]
                walk(brace + 1, close, NAMESPACE_ALIAS.get(tail_ns, tail_ns))
            else:
                name = function_name(header)
                if name:
                    out.append((ns, name, masked[brace + 1:close]))
            index = close + 1

    walk(0, len(masked), "")
    return out


def parse_op_constructions(upstream_keys):
    """Map every upstream op function name to the primitives it builds.

    Derived from the upstream sources in two passes. Pass one
    attributes each function body its own std::make_shared<Primitive>
    types and the functions it calls; an unqualified primitive name
    built inside an inner namespace (random::bits building RandomBits)
    falls back to its unique upstream namespace. Pass two walks the
    name-level call graph two hops out, so ops that build primitives
    through one helper (bitwise_and -> in_binary, compile ->
    compile_fuse, the linalg wrappers) still resolve, with no curated
    list anywhere.
    """
    direct = {}
    calls = {}
    for pattern in CONSTRUCTION_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            masked = mask_noncode(path.read_text())
            for ns, op, body in attribute_functions(masked):
                key = (ns, op)
                prims = direct.setdefault(key, set())
                edges = calls.setdefault(key, set())
                for match in MAKE_SHARED.finditer(body):
                    prim_ns, _, prim = match.group(1).rpartition("::")
                    if prim_ns and prim_ns not in ("fast", "distributed"):
                        continue
                    constructed = (prim_ns or ns, prim)
                    if constructed not in upstream_keys and not prim_ns:
                        owners = {
                            owner for owner, up in upstream_keys
                            if up == prim}
                        if len(owners) == 1:
                            constructed = (owners.pop(), prim)
                    if constructed in upstream_keys:
                        prims.add(constructed)
                for call in CALL_SITE.finditer(body):
                    qualifier = body[max(0, call.start() - 6):call.start()]
                    if (qualifier.endswith("std::")
                            or qualifier.endswith(".")
                            or qualifier.endswith("->")):
                        continue
                    edges.add(call.group(1))
    by_name = {}
    for key in direct:
        by_name.setdefault(key[1], []).append(key)
    op_map = {}
    for name in by_name:
        built = set()
        seen = set()
        frontier = {name}
        for _ in range(3):
            following = set()
            for entry in sorted(frontier):
                if entry in seen:
                    continue
                seen.add(entry)
                for key in by_name.get(entry, ()):
                    built |= direct[key]
                    following |= calls[key]
            frontier = following - seen
        for constructed in built:
            op_map.setdefault(name, set()).add(constructed)
    return op_map


def parse_test_cases(op_map, upstream_keys, upstream_names):
    """TEST_CASEs with the primitives each body provably exercises.

    Evidence is content-derived from the case body:
    - ops: primitives reached by a direct op call. A call is the
      maximal identifier before '('; std::-qualified host references
      and method calls do not count, other qualifiers such as fast::
      do.
    - exclusive: the subset of ops reached by calls whose whole
      construction map is one primitive, plus primitives the body
      builds itself with std::make_shared (the contract tests build
      primitives no public op constructs).
    - mentions: upstream class names inside body string literals, the
      named-error qualifiers the case expects.
    """
    cases = []
    order = {}
    for path in sorted(ROOT.glob(TEST_GLOB)):
        raw = path.read_text()
        masked = mask_noncode(raw)
        rel = path.relative_to(ROOT)
        if rel not in order:
            order[rel] = len(order)
        for match in re.finditer(r"TEST_CASE\s*\(", masked):
            line = masked[:match.start()].count("\n") + 1
            title = re.match(
                r'TEST_CASE\s*\(\s*"((?:[^"\\]|\\.)*)"',
                raw[match.start():]).group(1)
            args = balanced(masked, match.end() - 1)
            brace = masked.find("{", match.end() + len(args) + 1)
            close = block_end(masked, brace)
            body = masked[brace + 1:close]
            raw_body = raw[brace + 1:close]
            ops = set()
            exclusive = set()
            for call in CALL_SITE.finditer(body):
                qualifier = body[max(0, call.start() - 6):call.start()]
                if (qualifier.endswith("std::")
                        or qualifier.endswith(".")
                        or qualifier.endswith("->")):
                    continue
                mapped = op_map.get(call.group(1), set())
                ops |= mapped
                if len(mapped) == 1:
                    exclusive |= mapped
            for built in MAKE_SHARED.finditer(raw_body):
                prim = built.group(1)
                matches = {
                    up_ns for up_ns, up_prim in upstream_keys
                    if up_prim == prim}
                if len(matches) == 1:
                    constructed = (next(iter(matches)), prim)
                    ops.add(constructed)
                    exclusive.add(constructed)
            literals = " ".join(
                re.findall(r'"((?:[^"\\]|\\.)*)"', raw_body))
            mentions = {
                name for name in upstream_names
                if re.search(rf"\b{re.escape(name)}\b", literals)}
            cases.append({
                "order": order[rel],
                "rel": str(rel),
                "line": line,
                "name": title,
                "ops": ops,
                "exclusive": exclusive,
                "mentions": mentions,
            })
    return cases



def anchor_for(display, cases):
    """Anchor link for one primitive, resolved from case bodies.

    Evidence tiers, strongest first: a case calling an op whose
    construction map is exactly this primitive, then any case calling
    an op that constructs it (composite paths such as dequantize count,
    but yield to dedicated tests), then a case naming the primitive in
    an expected-error string. Within a tier the earliest case in file
    order wins.
    """
    ns, _, name = display.rpartition("::")
    key = (ns, name)
    for tier in ("exclusive", "ops", "mentions"):
        best = None
        for case in cases:
            if tier == "mentions":
                hit = name in case["mentions"]
            else:
                hit = key in case[tier]
            if not hit:
                continue
            if best is None or (case["order"], case["line"]) < (
                    best["order"], best["line"]):
                best = case
        if best is not None:
            return f"[`{best['name']}`](../{best['rel']}#L{best['line']})"
    return "—"


def parse_kernel_families():
    """Group compute.h ComputeKernel variants into families."""
    text = COMPUTE_H.read_text()
    body = re.search(
        r"enum class ComputeKernel[^{]*\{(.*?)\};", text, re.S).group(1)
    variants = re.findall(r"^\s*(\w+),\s*$", body, re.M)
    families = {}
    variant_map = {}
    for variant in variants:
        if variant == "Count":
            continue
        if variant.startswith("Cast"):
            base = "Cast"
        else:
            base = variant
            for suffix in ("BF16", "F32", "F16", "I32", "U32", "Bool"):
                if variant.endswith(suffix):
                    base = variant[: -len(suffix)]
                    break
        tokens = re.findall(r"BF16|F32|F16|I32|U32|Bool", variant)
        dtypes = {VARIANT_TOKENS[token] for token in tokens}
        entry = families.setdefault(
            base, {"variants": [], "dtypes": set()})
        entry["variants"].append(variant)
        entry["dtypes"] |= dtypes
        variant_map[variant] = dtypes
    return families, variant_map


def parse_capability_gates(primitives_text):
    """Extract the float16/bfloat16 gates from require_float_dtype."""
    text = strip_comments(primitives_text)
    match = re.search(
        r"void require_float_dtype\(.*?\{(.*?)\n\}", text, re.S)
    if not match:
        return []
    body = match.group(1)
    gates = []
    for dtype, first, second in re.findall(
            r"==\s*(float16|bfloat16)\s*&&\s*"
            r"\(!capabilities\.(\w+)\s*\|\|\s*!capabilities\.(\w+)\)", body):
        gates.append((dtype, first, second))
    return gates


def parse_helpers(primitives_text, variant_dtypes):
    """Map helper name to (clean body, unsupported fragments, dtypes)."""
    text = strip_comments(primitives_text)
    lines = primitives_text.splitlines()
    clean_lines = text.splitlines()
    helpers = {}
    pattern = re.compile(
        r"^(?:void|omarchy::ComputeKernel|uint32_t|bool|array)\s+"
        r"(dispatch_\w+|fill_broadcast_transport|require_float_dtype|"
        r"require_index_source_dtype|select_float_kernel|checked_u32|"
        r"checked_item_offset|materialize_batched_matrix)\(")
    for index, line in enumerate(clean_lines):
        match = pattern.match(line)
        if not match:
            continue
        body = strip_comments(function_body(lines, index))
        helpers[match.group(1)] = body
    fragments = {
        name: extract_fragments(body)
        for name, body in helpers.items()
    }
    called = {
        name: set(re.findall(
            r"\b(dispatch_\w+|fill_broadcast_transport|require_\w+|"
            r"select_float_kernel|checked_\w+|materialize_\w+)\(", body))
        for name, body in helpers.items()
    }
    dtypes = {}
    for name, body in helpers.items():
        found = set()
        for token, label in DTYPE_LABELS.items():
            if re.search(r"\b" + token + r"\b", body):
                found.add(label)
        for token in re.findall(r"omarchy::ComputeKernel::(\w+)", body):
            found |= variant_dtypes.get(token, set())
        dtypes[name] = found
    return helpers, fragments, called, dtypes


def render_expr(expr):
    """Render an unsupported() argument as a qualifier fragment."""
    literals = re.findall(r'"((?:[^"\\]|\\.)*)"', expr)
    identifiers = [
        token for token in re.findall(r"[A-Za-z_]\w*", expr)
        if token in NAME_VARS
    ]
    if literals and identifiers:
        text = "".join(literals).strip()
        if expr.lstrip().startswith('"'):
            return f"{text} {{name}}"
        return f"{{name}} {text}".strip()
    if literals:
        return "".join(literals).strip()
    return "{name}"


def extract_fragments(body):
    fragments = []
    for match in re.finditer(r"omarchy::unsupported\s*\(", body):
        args = balanced(body, match.end() - 1)
        depth = 0
        split = len(args)
        for index, char in enumerate(args):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                split = index
                break
        fragment = render_expr(args[:split])
        if fragment not in fragments:
            fragments.append(fragment)
    return fragments


def helper_fragments(helper, fragments, called, seen=None):
    """Transitive named-error fragments of a dispatch helper."""
    if seen is None:
        seen = set()
    if helper in seen or helper in GENERIC_HELPERS:
        return []
    seen.add(helper)
    out = list(fragments.get(helper, []))
    for child in sorted(called.get(helper, ())):
        out.extend(helper_fragments(child, fragments, called, seen))
    return out


def helper_dtypes(helper, dtypes, called, seen=None):
    if seen is None:
        seen = set()
    if helper in seen:
        return set()
    seen.add(helper)
    out = set(dtypes.get(helper, ()))
    for child in called.get(helper, ()):
        out |= helper_dtypes(child, dtypes, called, seen)
    return out


def parse_primitives():
    """Parse primitive entries in source order."""
    text = PRIMITIVES.read_text()
    lines = text.splitlines()
    entries = []
    index_of = {}
    namespace = ""
    for number, line in enumerate(lines):
        if re.match(r"^namespace (fast|distributed) \{", line):
            namespace = re.match(r"^namespace (\w+) \{", line).group(1)
        elif re.match(r"^\} // namespace", line):
            namespace = ""
        macro = re.match(
            r"^OMARCHY_(UNSUPPORTED|UNSUPPORTED_MULTI|USE_FALLBACK|BINARY|"
            r"UNARY)\((\w+)", line)
        impl = re.match(r"^void (\w+)::eval_gpu\(", line)
        fallback = re.match(r"^bool (\w+)::use_fallback\(", line)
        if macro:
            kind, name = macro.group(1), macro.group(2)
            key = (namespace, name)
            entry = index_of.setdefault(
                key, {"ns": namespace, "name": name, "kinds": [],
                      "order": len(index_of), "line": number + 1})
            entry["kinds"].append(kind)
        elif impl:
            name = impl.group(1)
            key = (namespace, name)
            entry = index_of.setdefault(
                key, {"ns": namespace, "name": name, "kinds": [],
                      "order": len(index_of), "line": number + 1})
            entry["impl_body"] = function_body(lines, number)
            entry["impl_line"] = number + 1
        elif fallback:
            name = fallback.group(1)
            key = (namespace, name)
            entry = index_of.setdefault(
                key, {"ns": namespace, "name": name, "kinds": [],
                      "order": len(index_of), "line": number + 1})
            entry["fallback_true"] = "return true" in function_body(
                lines, number)
    entries = sorted(index_of.values(), key=lambda entry: entry["order"])
    return text, entries


def parse_upstream_primitives():
    """Concrete primitive classes upstream MLX defines, as (ns, name).

    The coverage denominator is upstream's total primitive list, not
    the entries the omarchy backend happens to enumerate.
    """
    out = []
    for path, ns in [
        (UPSTREAM_H, ""),
        (UPSTREAM_FAST_H, "fast"),
        (UPSTREAM_DIST_H, "distributed"),
    ]:
        try:
            text = path.read_text()
        except OSError as exc:
            raise SystemExit(
                f"cannot read upstream header {path}: {exc}\n"
                "The coverage denominator needs the upstream MLX sources "
                "under .work/mlx; sync them and rerun.")
        for match in UPSTREAM_CLASS.finditer(text):
            name, base = match.groups()
            if name in UPSTREAM_BASES:
                continue
            if base not in UPSTREAM_BASES:
                raise SystemExit(
                    f"unrecognized base class {base!r} for upstream "
                    f"primitive {name!r} in {path}; extend UPSTREAM_BASES "
                    "before trusting the denominator.")
            out.append((ns, name))
    return out


HELPER_FAMILY = {
    "dispatch_elementwise": ["Elementwise"],
    "dispatch_int_elementwise": ["Elementwise"],
    "dispatch_comparison": ["Compare"],
    "dispatch_logical_or": ["LogicalOr"],
    "dispatch_matmul": ["Matmul"],
    "dispatch_softmax": ["Softmax"],
}

COPY_EVIDENCE = re.compile(
    r"\b(copy_gpu|copy_gpu_inplace|contiguous_copy_gpu|reshape_in_eval|"
    r"swapaxes_in_eval)\b")

DELEGATE_EVIDENCE = re.compile(r"\bomarchy::(eval_\w+)\(")
DELEGATE_FILES = {"eval_compiled_tape": "compiled.cpp"}


def parse_delegate_info(entries):
    """Mine fragments and tape op names from delegate target files."""
    info = {}
    for entry in entries:
        body = strip_comments(entry.get("impl_body", ""))
        match = DELEGATE_EVIDENCE.search(body)
        if not match:
            continue
        target = DELEGATE_FILES.get(match.group(1))
        fragments, tape_ops = [], []
        if target:
            path = ROOT / "overlay/mlx/backend/omarchy" / target
            text = strip_comments(path.read_text())
            for throw_match in re.finditer(
                    r"throw std::(?:runtime_error|invalid_argument)\s*\(",
                    text):
                args = balanced(text, throw_match.end() - 1)
                rendered = ""
                for literal, name_call in re.findall(
                        r'"((?:[^"\\]|\\.)*)"|(\w+\.name\(\))', args):
                    rendered += "{name}" if name_call else literal
                if rendered not in fragments:
                    fragments.append(rendered.strip())
            tape_ops = re.findall(
                r"typeid\(\w+\) == typeid\((\w+)\)", text)
        info[(entry["ns"], entry["name"])] = {
            "fragments": fragments, "tape_ops": tape_ops}
    return info


def build_row(entry, families, variant_dtypes, fragments, called, dtypes,
              bool_out_helpers, cases, delegate=None):
    name = entry["name"]
    display = f"{entry['ns']}::{name}" if entry["ns"] else name
    kinds = set(entry["kinds"])
    impl_body = entry.get("impl_body", "")

    if not impl_body and (
            "USE_FALLBACK" in kinds or entry.get("fallback_true")):
        status = "composed (fallback)"
        constraint_frags = [
            "no omarchy kernel; mlx composes core ops on the device"
        ]
        kernel_names = []
        row_dtypes = set()
    elif not impl_body and ("BINARY" in kinds or "UNARY" in kinds):
        status = "native"
        kernel_names = ["Elementwise"]
        row_dtypes = set(helper_dtypes(
            "dispatch_elementwise", dtypes, called))
        constraint_frags = helper_fragments(
            "dispatch_elementwise", fragments, called)
    elif not impl_body:
        status = "named-error"
        constraint_frags = [f"unsupported({display})"]
        kernel_names = []
        row_dtypes = set()
    else:
        body = strip_comments(impl_body)
        tokens = re.findall(r"omarchy::ComputeKernel::(\w+)", body)
        helpers_called = re.findall(
            r"\b(dispatch_\w+|fill_broadcast_transport)\(", body)
        kernel_names = []
        for token in tokens:
            for base in families:
                if token == base or token.startswith(base):
                    if base not in kernel_names:
                        kernel_names.append(base)
                    break
        for helper in helpers_called:
            for base in HELPER_FAMILY.get(helper, []):
                if base not in kernel_names:
                    kernel_names.append(base)
            if helper == "dispatch_sort":
                args = re.search(r"dispatch_sort\(([^)]*)\)", body)
                base = "ArgSort" if args and re.search(
                    r"\btrue\b", args.group(1)) else "Sort"
                if base not in kernel_names:
                    kernel_names.append(base)

        row_dtypes = set()
        for token in tokens:
            row_dtypes |= variant_dtypes.get(token, set())
        for helper in helpers_called:
            row_dtypes |= helper_dtypes(helper, dtypes, called)
            if helper in bool_out_helpers:
                row_dtypes.add("bool")
        # A dtype guard in the body narrows the family coverage, as in
        # GreaterEqual accepting int32 only.
        guard = {
            DTYPE_LABELS[token]
            for token in re.findall(
                r"dtype\(\)\s*!=\s*"
                r"(float32|float16|bfloat16|int32|uint32|int64)\b", body)
        }
        if guard and row_dtypes & guard:
            row_dtypes = (row_dtypes & guard) | (row_dtypes & {"bool"})

        delegate_dtypes = set()
        delegate_kernels = []
        if delegate:
            delegate_dtypes = set(delegate.get("dtypes", ()))
            for base in delegate.get("kernels", ()):
                if base not in delegate_kernels:
                    delegate_kernels.append(base)
        row_dtypes |= delegate_dtypes
        for base in delegate_kernels:
            if base not in kernel_names:
                kernel_names.append(base)
        copy_evidence = bool(COPY_EVIDENCE.search(body))
        if tokens:
            status = "native"
        elif delegate or len(set(kernel_names)) > 1 or copy_evidence:
            status = "composed"
        else:
            status = "native"
        constraint_frags = (
            list(delegate["fragments"]) if delegate
            else extract_fragments(body))
        seen = set(constraint_frags)
        for helper in helpers_called:
            for fragment in helper_fragments(helper, fragments, called):
                if fragment not in seen:
                    seen.add(fragment)
                    constraint_frags.append(fragment)

    ordered = [
        dtype + ("*" if dtype in ("f16", "bf16") else "")
        for dtype in DTYPE_ORDER
        if dtype in row_dtypes
    ]
    return {
        "display": display,
        "status": status,
        "kernels": kernel_names,
        "dtypes": ordered,
        "constraints": constraint_frags,
        "anchor": anchor_for(display, cases),
        "line": entry.get("impl_line", entry["line"]),
    }


def format_families(families, copy_families, gates):
    lines = [
        "## Kernel families",
        "",
        "Variants come from `ComputeKernel` in "
        "`overlay/mlx/backend/omarchy/compute.h`. "
        "One variant exists per dtype the family serves.",
        "",
        "| Family | Variants | Dtypes | Dispatched from |",
        "|---|---|---|---|",
    ]
    for base, entry in families.items():
        dtypes = ", ".join(
            dtype + ("*" if dtype in ("f16", "bf16") else "")
            for dtype in DTYPE_ORDER if dtype in entry["dtypes"])
        source = "copy.cpp" if base in copy_families else "primitives.cpp"
        lines.append(
            f"| {base} | {len(entry['variants'])} | {dtypes} | {source} |")
    lines.append("")
    if gates:
        lines.append("Capability gates behind the `*` markers:")
        lines.append("")
        for dtype, first, second in gates:
            label = "float16" if dtype == "float16" else "bfloat16"
            lines.append(
                f"- {label} needs `{first}` and `{second}`.")
        lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=datetime.date.today().isoformat(),
        help="generation date recorded in the header (YYYY-MM-DD)")
    parser.add_argument(
        "--json-out",
        default=None,
        metavar="PATH",
        help="also write the coverage badge in shields.io endpoint "
             "format (docs/coverage.json)")
    args = parser.parse_args()

    primitives_text, entries = parse_primitives()
    families, variant_dtypes = parse_kernel_families()
    gates = parse_capability_gates(primitives_text)
    helpers, fragments, called, helper_dtype_map = parse_helpers(
        primitives_text, variant_dtypes)
    upstream = parse_upstream_primitives()
    op_map = parse_op_constructions(set(upstream))
    upstream_names = sorted({name for _, name in upstream})
    cases = parse_test_cases(op_map, set(upstream), upstream_names)
    bool_out_helpers = {
        helper
        for helper, body in helpers.items()
        if re.search(r"out\.dtype\(\)\s*!=\s*bool_", body)
    }
    copy_text = strip_comments(COPY_CPP.read_text())
    copy_families = set()
    for token in re.findall(r"omarchy::ComputeKernel::(\w+)", copy_text):
        for base in families:
            if token == base or token.startswith(base):
                copy_families.add(base)
                break

    delegate_info = parse_delegate_info(entries)

    def build(entry, delegate=None):
        return build_row(entry, families, variant_dtypes, fragments,
                         called, helper_dtype_map, bool_out_helpers,
                         cases, delegate)

    core_rows = {}
    rows_by_key = {}
    for entry in entries:
        key = (entry["ns"], entry["name"])
        if key in delegate_info:
            continue
        rows_by_key[key] = build(entry)
    for key, info in delegate_info.items():
        tape_dtypes, tape_kernels = set(), []
        for op_name in info["tape_ops"]:
            for (ns, name), row in rows_by_key.items():
                if ns == "" and name == op_name:
                    tape_dtypes |= {
                        dtype.rstrip("*") for dtype in row["dtypes"]}
                    for base in row["kernels"]:
                        if base not in tape_kernels:
                            tape_kernels.append(base)
                    break
        info["dtypes"] = tape_dtypes
        info["kernels"] = tape_kernels
        entry = next(entry for entry in entries
                     if (entry["ns"], entry["name"]) == key)
        rows_by_key[key] = build(entry, delegate=info)
    rows = [rows_by_key[(entry["ns"], entry["name"])]
            for entry in entries]

    backend_keys = {(entry["ns"], entry["name"]) for entry in entries}
    phantom = backend_keys - set(upstream)
    if phantom:
        raise SystemExit(
            "backend entries absent from upstream MLX: "
            + ", ".join(
                f"{ns}::{name}" if ns else name
                for ns, name in sorted(phantom)))
    no_entry = sorted(set(upstream) - backend_keys)
    shared_gpu_names = parse_shared_gpu_primitives()
    shared_gpu = [key for key in no_entry if key[1] in shared_gpu_names]
    nothing_runs = [key for key in no_entry if key[1] not in shared_gpu_names]
    total = len(upstream)

    def is_anchored(row):
        return row["anchor"] != "—"

    def share(count):
        return f"{count / total * 100:.1f}%"

    native_hit = sum(
        1 for row in rows
        if row["status"] == "native" and is_anchored(row))
    native_miss = sum(
        1 for row in rows
        if row["status"] == "native" and not is_anchored(row))
    composed_hit = sum(
        1 for row in rows
        if row["status"].startswith("composed") and is_anchored(row))
    composed_miss = sum(
        1 for row in rows
        if row["status"].startswith("composed") and not is_anchored(row))
    named_error = sum(
        1 for row in rows if row["status"] == "named-error")
    named_error_anchored = sum(
        1 for row in rows
        if row["status"] == "named-error" and is_anchored(row))
    shared_hit = sum(
        1 for key in shared_gpu
        if anchor_for(f"{key[0]}::{key[1]}" if key[0] else key[1], cases)
        != "—")
    shared_miss = len(shared_gpu) - shared_hit
    buckets = [
        ("native, test-anchored", native_hit),
        ("native, untested", native_miss),
        ("composed, test-anchored", composed_hit),
        ("composed, untested", composed_miss),
        ("shared-gpu, test-anchored", shared_hit),
        ("shared-gpu, untested", shared_miss),
        ("named-error (backend entry raises unsupported)", named_error),
        ("not implemented (nothing runs)", len(nothing_runs)),
    ]
    covered = native_hit + composed_hit + shared_hit
    implemented = (
        covered + native_miss + composed_miss + shared_miss)
    coverage_pct = covered / total * 100

    def badge_color(pct):
        if pct < 34:
            return "red"
        if pct < 67:
            return "orange"
        if pct < 90:
            return "yellow"
        return "green"

    if args.json_out:
        payload = {
            "schemaVersion": 1,
            "label": "MLX coverage",
            "message":
                f"{coverage_pct:.1f}% ({covered}/{total} primitives)",
            "color": badge_color(coverage_pct),
        }
        pathlib.Path(args.json_out).write_text(
            json.dumps(payload) + "\n")

    out = []
    out.append("# Compatibility matrix")
    out.append("")
    sha = git("rev-parse", "--short", "HEAD")
    branch = git("branch", "--show-current")
    # The header records the canonical invocation so the document is
    # byte-identical no matter where --json-out pointed.
    command = (
        "python3 tools/gen-compat-matrix.py"
        " --json-out docs/coverage.json > docs/compatibility-matrix.md")
    out.append(
        f"Generated by `{command}` on {args.date} "
        f"from the working tree at commit `{sha}` on branch `{branch}`.")
    out.append(
        "Regenerate after backend changes; the generator is the source "
        "of truth and the same tree produces the same bytes.")
    out.append("")
    out.append("## Coverage")
    out.append("")
    out.append(
        f"**MLX primitive coverage: {coverage_pct:.1f}% — {covered} of "
        f"{total} upstream primitives.**")
    out.append("")
    out.append(
        "Coverage counts a primitive only when the omarchy backend "
        "dispatches or composes it (native or composed) and a TEST_CASE "
        "in `overlay/tests/omarchy/` anchors it: the case body calls an "
        "op that constructs the primitive, constructs the primitive "
        "itself, or names the primitive in an expected-error string. "
        "Native-but-untested primitives keep "
        "their own buckets below and are not counted as covered.")
    out.append("")
    out.append(
        "The denominator is every concrete primitive class upstream "
        "MLX defines — parsed from `.work/mlx/mlx/primitives.h`, "
        "`.work/mlx/mlx/fast_primitives.h`, and "
        "`.work/mlx/mlx/distributed/primitives.h` — not only the "
        f"{len(rows)} entries the omarchy backend enumerates.")
    out.append("")
    out.append("| Bucket | Count | Share of upstream |")
    out.append("|---|---|---|")
    for label, count in buckets:
        out.append(f"| {label} | {count} | {share(count)} |")
    out.append(f"| upstream total | {total} | 100.0% |")
    out.append("")
    out.append(
        "Implementation-path coverage without the test requirement: "
        f"{implemented}/{total} = {implemented / total * 100:.1f}%. "
        f"{named_error_anchored} named-error primitives pin their "
        "unsupported error with a test; a pinned error is not coverage.")
    out.append("")
    out.append("## Status terms")
    out.append("")
    out.append(
        "- native: one omarchy kernel or engine path dispatches the "
        "operation.")
    out.append(
        "- composed: the operation assembles on the device from several "
        "dispatched kernels, or mlx composes core ops because "
        "use_fallback returns true.")
    out.append(
        "- named-error: eval_gpu raises `omarchy::unsupported` and the "
        "run stops with that qualifier.")
    out.append(
        "- shared-gpu: the primitive has no omarchy entry, and upstream's "
        "shared `backend/gpu/primitives.cpp` resolves it generically — "
        "zero-copy buffer views or the omarchy copy engine. It counts as "
        "covered only when a TEST_CASE anchors it.")
    out.append(
        "- not implemented (nothing runs): upstream MLX defines the "
        "primitive, the omarchy backend has no entry for it, and the "
        "shared GPU layer has no eval_gpu for it either.")
    out.append(
        "- Dtypes marked `*` pass the capability gates listed under the "
        "kernel families.")
    out.append(
        "- `{name}` in a constraint stands for the dispatched "
        "operation name.")
    out.append("")
    out.extend(format_families(families, copy_families, gates))
    out.append("## Primitives")
    out.append("")
    out.append(
        "One row per primitive entry in "
        "`overlay/mlx/backend/omarchy/primitives.cpp`, in source order.")
    out.append("")
    out.append(
        "| Primitive | Status | Kernels | Dtypes | Named-error "
        "constraints | Test anchor |")
    out.append("|---|---|---|---|---|---|")
    for row in rows:
        kernels = (
            ", ".join(row["kernels"]) if row["kernels"] else "—")
        dtypes = ", ".join(row["dtypes"]) if row["dtypes"] else "—"
        constraints = (
            "; ".join(f"`{fragment}`" for fragment in row["constraints"])
            if row["constraints"] else "—")
        out.append(
            f"| {row['display']} | {row['status']} | {kernels} | "
            f"{dtypes} | {constraints} | {row['anchor']} |")
    out.append("")
    out.append(
        "Every dispatch path also bounds element counts, item offsets, "
        "and index spans to `uint32` and raises a named error beyond. "
        "No primitive falls back to CPU tensors.")
    out.append("")
    out.append(
        f"## No omarchy entry — {len(no_entry)} primitives")
    out.append("")
    out.append(
        "Upstream MLX defines these primitives, but "
        "`overlay/mlx/backend/omarchy/primitives.cpp` has no entry for "
        "them. They count in the coverage denominator.")
    out.append("")
    out.append("| Primitive | Status | Test anchor |")
    out.append("|---|---|---|")
    for ns, name in no_entry:
        display = f"{ns}::{name}" if ns else name
        if (ns, name) in shared_gpu:
            status = "shared-gpu"
            anchor = anchor_for(display, cases)
        else:
            status = "not implemented (nothing runs)"
            anchor = "—"
        out.append(f"| {display} | {status} | {anchor} |")
    out.append("")
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
