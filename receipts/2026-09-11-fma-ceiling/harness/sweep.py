#!/usr/bin/env python3
"""FMA ceiling sweep for honeykrisp on Apple M1 (jwm1-linux).

Generates GLSL compute shaders sweeping accumulator count, unroll factor,
register pressure, workgroup size and data type; compiles each with
glslangValidator; times with fma_bench; aggregates JSON.

Run under: flock /tmp/m1-gpu.lock
Usage:
  sweep.py run <outdir> [stage...]    # stages: cal, screen, wg, pressure, f16, f16p, i32, coop
  sweep.py dump <outdir> arm [arm...] # AGX_DEBUG=shaders asm dump for arms
"""
import json, os, subprocess, sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fma_bench")
T = 16384           # total FMAs per thread per dispatch
TOTAL_INV = 262144  # total invocations
COOP_EXT = "#extension GL_KHR_memory_scope_semantic : require\n"
GFLOP_FIXED = 2.0 * TOTAL_INV * T / 1e9  # for width-1 ops

COOP_SHADER = """#version 460
#extension GL_KHR_memory_scope_semantic : require
#extension GL_KHR_shader_subgroup_basic : require
layout(local_size_x = 128) in;
layout(set = 0, binding = 0) buffer Out { uint out_buf[]; };
void main() {
    coopmat<float, gl_ScopeSubgroup, 8, 8, gl_MatrixUseA> ma = coopmat<float, gl_ScopeSubgroup, 8, 8, gl_MatrixUseA>(1.0000001);
    coopmat<float, gl_ScopeSubgroup, 8, 8, gl_MatrixUseB> mb = coopmat<float, gl_ScopeSubgroup, 8, 8, gl_MatrixUseB>(1e-7);
    coopmat<float, gl_ScopeSubgroup, 8, 8, gl_MatrixUseAccumulator> mc = coopmat<float, gl_ScopeSubgroup, 8, 8, gl_MatrixUseAccumulator>(float(gl_SubgroupInvocationID));
    for (uint i = 0u; i < iters; ++i) {
        mc = coopMatMulAdd(ma, mb, mc);
    }
    coopMatStore(mc, sink_tile, 0, 8, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    memoryBarrierShared();
    out_buf[gl_GlobalInvocationID.x] = floatBitsToUint(sink_tile[gl_LocalInvocationID.x]);
}
"""

def gen(wgx, typ, nacc, unroll, press):
    trip = T // (nacc * unroll)
    assert trip * nacc * unroll == T, "T not divisible"
    width = 2 if typ == "f16p" else 1
    if typ == "f32":
        t, ext = "float", ""
        cs = [f"{1.0 + 1e-7*(1 + (j % 7)):.9f}" for j in range(16)]
        ds = [f"{1e-7*(1 + (j % 5)):.9e}" for j in range(16)]
    elif typ in ("f16", "f16p"):
        vec = "f16vec2" if typ == "f16p" else "float16_t"
        t = vec
        ext = "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
        cs = [f"{vec}({1.0 + 0.002*(1 + (j % 7)):.5f})" for j in range(16)]
        ds = [f"{vec}({0.001*(1 + (j % 5)):.5f})" for j in range(16)]
    else:
        t, ext = "int", ""
        cs = [str(1103515245 + j) for j in range(16)]
        ds = [str(12345 + 7919 * j) for j in range(16)]

    decls = [f"{t} a{j} = {t}(float(g) * 1e-6 + {j}) + {t}({ds[j % 16]});" for j in range(nacc)]
    pdecl = [f"{t} p{k} = {t}(float(g) * 1e-5 + {k}) * {t}({cs[k % 16]});" for k in range(press)]
    body = []
    for k in range(unroll):
        for j in range(nacc):
            ci, di = (j + k) % 16, (j + 2 * k + 1) % 16
            if typ == "i32":
                body.append(f"    a{j} = a{j} * {cs[ci]} + {ds[di]};")
            else:
                body.append(f"    a{j} = fma(a{j}, {cs[ci]}, {ds[di]});")
    sink_acc = "+".join([f"a{j}" for j in range(nacc)] or ["0"])
    sink_p = "+".join([f"p{k}" for k in range(press)] or ["0"])
    if typ == "i32":
        sink = f"uint({sink_acc}) ^ uint({sink_p})"
    else:
        sink = f"floatBitsToUint(float({sink_acc})) ^ floatBitsToUint(float({sink_p}))"

    return f"""#version 460
{ext}layout(local_size_x = {wgx}) in;
layout(set = 0, binding = 0) buffer Out {{ uint out_buf[]; }};
layout(push_constant) uniform PC {{ uint iters; }};
void main() {{
    uint g = gl_GlobalInvocationID.x;
{chr(10).join('    ' + d for d in decls + pdecl)}
    for (uint i = 0u; i < iters; ++i) {{
{chr(10).join(body)}
    }}
    out_buf[g] = {sink};
}}
"""

def assemble_coopmin(outdir):
    asm = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coop_min.spvasm")
    spv = os.path.join(outdir, "coop_min.spv")
    r = subprocess.run(["spirv-as", "--target-env", "spv1.3", "-o", spv, asm],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stdout + r.stderr
    return spv, None
def compile_text(outdir, name, text):
    src = os.path.join(outdir, f"{name}.comp")
    spv = os.path.join(outdir, f"{name}.spv")
    with open(src, "w") as f:
        f.write(text)
    r = subprocess.run(["glslangValidator", "-V", "--target-env", "vulkan1.3", "-o", spv, src],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stdout + r.stderr
    return spv, None

def run_one(outdir, name, wgx, typ, nacc, unroll, press, iters_override=None, gflop=None):
    if typ in ("coop", "coopmin"):
        spv, err = assemble_coopmin(outdir) if typ == "coopmin" else (None, "glslang coopmat unsupported")
    else:
        spv, err = compile_text(outdir, name, gen(wgx, typ, nacc, unroll, press))
    if not spv:
        return {"name": name, "compile_error": err.strip()[:400]}
    groups = TOTAL_INV // wgx
    width = 2 if typ == "f16p" else (16 if typ in ("coop", "coopmin") else 1)
    if iters_override is not None:
        iters = iters_override
    else:
        iters = 2048 if typ in ("coop", "coopmin") else T // (nacc * unroll)
    if typ == "coopmin" and gflop is None:
        gflop = 2.0 * (TOTAL_INV // 32) * iters * 512 / 1e9
    if gflop is None:
        gflop = 2.0 * TOTAL_INV * iters * nacc * unroll * width / 1e9
    r = subprocess.run([HARNESS, spv, str(TOTAL_INV), str(iters), f"{gflop:.6f}", str(groups)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"name": name, "run_error": (r.stderr or r.stdout).strip()[:300]}
    j = json.loads(r.stdout.strip().splitlines()[-1])
    j.update(name=name, typ=typ, nacc=nacc, unroll=unroll, wgx=wgx, press=press,
             iters=iters, gflop_total=gflop)
    return j

def stage_arms(stage):
    if stage == "cal":
        return [("f32", 8, 4, 128, 0)]
    if stage == "screen":
        return [("f32", na, u, 128, 0) for na in (1, 2, 4, 8, 16) for u in (1, 2, 4, 8)]
    if stage == "wg":
        return [("f32", 8, 4, w, 0) for w in (32, 64, 128, 256, 512)]
    if stage == "pressure":
        return [("f32", 8, 4, 128, p) for p in (4, 8, 16, 24, 32, 40)]
    if stage == "f16":
        return [("f16", na, u, 128, 0) for na in (4, 8, 16) for u in (2, 4, 8)]
    if stage == "f16p":
        return [("f16p", na, u, 128, 0) for na in (4, 8, 16) for u in (2, 4, 8)]
    if stage == "i32":
        return [("i32", na, u, 128, 0) for na in (4, 8, 16) for u in (2, 4)]
    if stage == "coop":
        return [("coop", 8, 8, 128, 0), ("coop", 8, 16, 128, 0)]
    if stage == "coopmin":
        return [("coopmin", 8, 8, 128, 0)]
    raise SystemExit(f"unknown stage {stage}")

def main():
    mode, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    if mode == "run":
        stages = sys.argv[3:] or ["cal", "screen", "wg", "pressure", "f16", "f16p", "i32", "coopmin"]
        results = []
        for st in stages:
            for (typ, na, u, w, p) in stage_arms(st):
                name = f"{typ}_a{na}_u{u}_w{w}_p{p}"
                if st == "cal":
                    j = run_one(outdir, name, w, typ, na, u, p, iters_override=0, gflop=1e-9)
                else:
                    j = run_one(outdir, name, w, typ, na, u, p)
                print(json.dumps(j), flush=True)
                results.append(j)
        with open(os.path.join(outdir, "results.json"), "a") as f:
            for j in results:
                f.write(json.dumps(j) + "\n")
    elif mode == "dump":
        with open(os.path.join(outdir, "results.json")) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        by = {r["name"]: r for r in rows if r.get("name") and r.get("typ")}
        for name in sys.argv[3:]:
            r = by[name]
            if r["typ"] == "coopmin":
                spv, err = assemble_coopmin(outdir)
            elif r["typ"] == "coop":
                spv, err = None, "glslang coopmat unsupported"
            else:
                spv, err = compile_text(outdir, name, gen(r["wgx"], r["typ"], r["nacc"], r["unroll"], r["press"]))
            if not spv:
                print(f"{name}: no spv ({err[:120]})"); continue
            env = {**os.environ, "AGX_MESA_DEBUG": "shaders", "MESA_SHADER_CACHE_DISABLE": "true"}
            rr = subprocess.run([HARNESS, spv, str(TOTAL_INV), str(r["iters"]), "-1",
                                 str(TOTAL_INV // r["wgx"])], capture_output=True, text=True, env=env)
            with open(os.path.join(outdir, f"{name}.agx"), "w") as f:
                f.write(rr.stdout + "\n===== stderr =====\n" + rr.stderr)

if __name__ == "__main__":
    main()
