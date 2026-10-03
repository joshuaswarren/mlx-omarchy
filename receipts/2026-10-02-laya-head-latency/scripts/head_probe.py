#!/usr/bin/env python3
"""Laya head latency probe for the 250 ms routing-gate work.

Subcommands (all GPU parts run on the M2 inside a gpu-turn ticket):

  phases   in-process LayaEngine, per-call phase timers (tokenize, collate,
           array build, forward record, mx.eval, post-processing)
  serve    spawn mlx_omarchy_laya.server, time N warm HTTP round trips,
           capture the engine's own prompt_ms/predicted_ms per call
  ab       alternating same-boot A/B arms (env levers), one server per arm
           per cycle, n calls per arm-cycle
  subs     run the phases driver under MLX_OMARCHY_GPU_PROFILE and report
           submissions + ms/submission (reuses subcap_calibrate parsing)
  answers  full-suite head answers (for decision-agreement comparisons)
  compare  two answer files -> exact decision equality + policy-3 route
           equality (pure python, dev-box safe)

Timing-validity context (loadavg, PSI, other python/mlx processes) is
recorded before/after every measured batch.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

QUESTION_TEXT = (
    "Classify this user turn as exactly one of: conversation (the "
    "user wants an explanation or chat, no bounded choice), "
    "structured_decision (the user supplies explicit alternatives to "
    "pick from, even when phrased as a question), or clarify (the "
    "user's request is missing options, criteria, or scope and the "
    "assistant needs to ask before deciding)."
)
ROUTES = ("conversation", "structured_decision", "clarify")


def payload_for(text):
    return {"state": text, "questions": {"route": {"type": "choice", "instructions": QUESTION_TEXT, "criteria": {label: None for label in ROUTES}}}}


def suite_texts(path):
    return [c["text"] for c in json.loads(Path(path).read_text())["cases"]]


def summarize(ms_list):
    # Same conventions as the gate's routing_latency.py: p50 = ok[n//2],
    # p95 = ok[int(n*0.95)-1], so numbers stay comparable to the receipt.
    ok = sorted(ms_list)
    if not ok:
        return {"n": 0, "p50": None, "p95": None, "min": None, "max": None}
    return {"n": len(ok), "p50": ok[len(ok) // 2], "p95": ok[max(0, int(len(ok) * 0.95) - 1)], "min": ok[0], "max": ok[-1]}


def host_context():
    loadavg = Path("/proc/loadavg").read_text().strip()
    psi_path = Path("/proc/pressure/cpu")
    psi_raw = psi_path.read_text() if psi_path.exists() else ""
    psi_avg10 = None
    for line in psi_raw.splitlines():
        if line.startswith("some "):
            psi_avg10 = float(next((v.split("=", 1)[1] for v in line.split() if v.startswith("avg10=")), "nan"))
            break
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    load1 = float(loadavg.split()[0])
    processes = others()
    chat_workers = [p for p in processes
                    if "_mlxlm_server" in p["command"] or "mlx_lm" in p["command"]]
    return {"loadavg": loadavg, "load1": load1,
            "psi_cpu": psi_raw.strip().replace("\n", "; "),
            "psi_cpu_some_avg10": psi_avg10,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "uptime_s": uptime,
            "quiet_eligible": uptime >= 360 and load1 < 0.5 and psi_avg10 == 0.0,
            "other_python_mlx": processes, "chat_workers": chat_workers}


def require_quiet(ctx):
    if not ctx["quiet_eligible"]:
        raise RuntimeError("timing gate failed (need uptime >= 360s, load1 < 0.5, CPU PSI some avg10 = 0): %s" % ctx)


def require_chat_worker(ctx, required):
    if required and not ctx["chat_workers"]:
        raise RuntimeError("co-serving run requires an already-resident chat worker; none was found")


def others():
    out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
    me = {str(os.getpid()), str(os.getppid())}
    result = []
    for line in out.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        pid, command = fields
        if (("python" in command or "mlx" in command) and pid not in me
                and "mlx_omarchy_laya.server" not in command
                and "head_probe" not in command):
            result.append({"pid": int(pid), "command": command})
    return result


def call_decision(url, payload, timeout_seconds):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST", headers={"Content-Type": "application/json"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds + 5.0) as resp:
            raw = resp.read()
        return json.loads(raw), (time.monotonic() - start) * 1000.0, False
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        return {"error": str(exc), "timed_out": True}, (time.monotonic() - start) * 1000.0, True


def write_out(path, doc):
    tmp = Path(path + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)
    os.sync()


def spawn_server(python, ckpt, port, dtype, env, log_path):
    env = dict(env or {})
    env["PYTHONPATH"] = str(REPO / "serve") + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    log = open(log_path, "w")
    proc = subprocess.Popen([python, "-m", "mlx_omarchy_laya.server", "--model", ckpt, "--host", "127.0.0.1", "--port", str(port), "--dtype", dtype, "--max-questions", "8"], env=env, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("laya server exited early; log %s" % log_path)
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as r:
                if r.status == 200:
                    return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("laya server not healthy in 120s; log %s" % log_path)


def timed_batch(port, texts, warmups, n, timeout_s=5.0):
    url = "http://127.0.0.1:%d/v1/decisions" % port

    def measure(i):
        raw, ms, failed = call_decision(
            url, payload_for(texts[i % len(texts)]), timeout_s)
        timings = raw.get("timings") or {}
        prompt_ms = timings.get("prompt_ms")
        predicted_ms = timings.get("predicted_ms")
        accounted_ms = (prompt_ms + predicted_ms
                        if prompt_ms is not None and predicted_ms is not None else None)
        return {"ms": ms, "failed": failed,
                "prompt_ms": prompt_ms, "predicted_ms": predicted_ms,
                "http_and_post_residual_ms": (ms - accounted_ms
                                               if accounted_ms is not None else None),
                "input_tokens": (raw.get("usage") or {}).get("input_tokens")}

    warmup_rows = [measure(i) for i in range(warmups)]
    rows = [measure(i) for i in range(n)]
    return {"warmups": warmup_rows, "rows": rows}


def cmd_phases(args):
    sys.path.insert(0, str(REPO / "serve"))
    from mlx_omarchy_laya import api as laya_api
    texts = suite_texts(args.suite)
    before = host_context(); require_quiet(before); require_chat_worker(before,args.require_chat_worker)
    engine = laya_api.LayaEngine(args.ckpt)
    from mlx_omarchy_laya.sequence import collate, encode_questions
    from mlx_omarchy_laya import model as laya_model
    import mlx.core as mx
    import numpy as np
    for i in range(5):
        text = texts[i % len(texts)]
        engine.system_one(text, payload_for(text)["questions"])
    rows = []
    for i in range(args.n):
        text = texts[i % len(texts)]; questions = payload_for(text)["questions"]
        t0 = time.perf_counter()
        _, items = encode_questions(engine.tok, text, questions, engine.max_len, engine.head_max_len)
        t1 = time.perf_counter()
        b = collate(items, engine.tok.pad_token_id); t2 = time.perf_counter()
        arrays = (mx.array(b["input_ids"]), mx.array(b["attention_mask"]), mx.array(b["marker_pos"]), mx.array(b["marker_mask"]), mx.array(b["qtype"])); t3 = time.perf_counter()
        logits, act = laya_model.forward(engine.weights, engine.enc_cfg, *arrays, engine.head_layers); t4 = time.perf_counter()
        mx.eval(logits, act); t5 = time.perf_counter()
        _ = np.array(logits.astype(mx.float32)); _ = np.array(mx.softmax(act.astype(mx.float32), axis=-1)); t6 = time.perf_counter()
        rows.append({"wall_ms": (t6-t0)*1000, "tokenize_ms": (t1-t0)*1000, "collate_ms": (t2-t1)*1000, "array_ms": (t3-t2)*1000, "fwd_record_ms": (t4-t3)*1000, "eval_ms": (t5-t4)*1000, "post_ms": (t6-t5)*1000, "n_tokens": b["n_tokens"]})
    api_rows = []
    for i in range(args.n):
        text = texts[i % len(texts)]
        started = time.perf_counter()
        result = engine.system_one(text, payload_for(text)["questions"])
        wall_ms = (time.perf_counter() - started) * 1000.0
        prompt_ms = result["timings"]["prompt_ms"]
        predicted_ms = result["timings"]["predicted_ms"]
        api_rows.append({"wall_ms": wall_ms, "prompt_ms": prompt_ms,
                         "predicted_ms": predicted_ms,
                         "post_and_python_residual_ms": wall_ms - prompt_ms - predicted_ms,
                         "input_tokens": result["usage"]["input_tokens"]})
    after = host_context()
    summary = {k: summarize([r[k] for r in rows]) for k in ("wall_ms", "tokenize_ms", "collate_ms", "array_ms", "fwd_record_ms", "eval_ms", "post_ms")}
    api_summary = {k: summarize([r[k] for r in api_rows]) for k in
                   ("wall_ms", "prompt_ms", "predicted_ms", "post_and_python_residual_ms")}
    model_info = {"model_id": engine.model_id, "dtype": str(engine.dtype),
                  "max_len": engine.max_len, "head_max_len": engine.head_max_len,
                  "head_layers": engine.head_layers,
                  "weight_storage": "fp16 safetensors; no quantization",
                  "encoder": vars(engine.enc_cfg),
                  "weight_elements": sum(w.size for w in engine.weights.values())}
    write_out(args.out, {"mode":"phases", "n":args.n, "model":model_info,
                         "before":before, "after":after,
                         "quiet_valid":before["quiet_eligible"] and after["quiet_eligible"],
                         "co_served":bool(before["chat_workers"]) and bool(after["chat_workers"]),
                         "summary":summary, "api_summary":api_summary, "api_rows":api_rows,
                         "rows":rows})
    print(json.dumps(summary, indent=1))


def run_serve_batch(python, ckpt, port, dtype, env, n, texts, log_path, warmups=5):
    proc = spawn_server(python, ckpt, port, dtype, env, log_path)
    try:
        return timed_batch(port, texts, warmups, n)
    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill()


def cmd_serve(args):
    texts = suite_texts(args.suite); before = host_context(); require_quiet(before); require_chat_worker(before,args.require_chat_worker)
    batch = run_serve_batch(sys.executable,args.ckpt,args.port,args.dtype,dict(os.environ),args.n,texts,args.log)
    after = host_context(); summary = summarize([r["ms"] for r in batch["rows"] if not r["failed"]])
    write_out(args.out,{"mode":"serve","n":args.n,"dtype":args.dtype,"before":before,"after":after,"quiet_valid":before["quiet_eligible"] and after["quiet_eligible"],"co_served":bool(before["chat_workers"]) and bool(after["chat_workers"]),"summary":summary,"warmups":batch["warmups"],"rows":batch["rows"]}); print(json.dumps(summary,indent=1))


def parse_arms(spec):
    arms=[]
    for chunk in spec.split(";"):
        name,_,vars_str=chunk.partition(":"); env={}
        for pair in vars_str.split(","):
            if pair.strip():
                k,_,v=pair.partition("="); env[k.strip()]=v.strip()
        arms.append((name,env))
    names = [name for name, _ in arms]
    if len(arms) < 2 or len(set(names)) != len(names):
        raise ValueError("A/B requires at least two uniquely named arms")
    return arms


def cmd_ab(args):
    arms=parse_arms(args.arms); texts=suite_texts(args.suite)
    out_path=Path(args.out); doc=json.loads(out_path.read_text()) if out_path.exists() else {}
    if doc and doc.get("mode")!="ab":
        raise ValueError("existing output is not an ab result")
    if doc.get("arms") and doc.get("arms")!=dict(arms):
        raise ValueError("existing output arms spec differs from requested")
    if doc.get("cycles_requested") not in (None,args.cycles) or doc.get("n_per_cycle") not in (None,args.n):
        raise ValueError("existing output was produced with different cycles/n")
    cycles=doc.get("cycles",[]); completed={(c["cycle"],c["arm"]) for c in cycles}
    if len(completed)!=len(cycles):
        raise ValueError("duplicate cycle/arm records in existing output")
    if not completed<={(cyc,name) for cyc in range(args.cycles) for name,_ in arms}:
        raise ValueError("existing output does not match requested arms/cycle count")
    for cycle in range(args.cycles):
        cycle_arms = arms if cycle % 2 == 0 else list(reversed(arms))
        for name,env in cycle_arms:
            if (cycle,name) in completed:
                continue
            full_env=dict(os.environ); full_env.update(env); before=host_context(); require_quiet(before); require_chat_worker(before,args.require_chat_worker)
            rows=run_serve_batch(sys.executable,args.ckpt,args.port,args.dtype,full_env,args.n,texts,"%s.%s.cycle%d.log"%(args.log,name,cycle))
            after=host_context(); cycles.append({"cycle":cycle,"arm":name,"env":env,"before":before,"after":after,"quiet_valid":before["quiet_eligible"] and after["quiet_eligible"],"co_served":bool(before["chat_workers"]) and bool(after["chat_workers"]),"warmups":rows["warmups"],"rows":rows["rows"]})
            write_out(args.out,{"mode":"ab","cycles_requested":args.cycles,"n_per_cycle":args.n,"dtype":args.dtype,"arms":dict(arms),"cycles":cycles}); completed.add((cycle,name))
            summary=summarize([r["ms"] for r in rows["rows"] if not r["failed"]])
            print("cycle %d arm %s p50=%.1f p95=%.1f n=%d failed=%d"%(cycle,name,summary["p50"] or -1,summary["p95"] or -1,summary["n"],sum(r["failed"] for r in rows["rows"])),flush=True)
    per_arm={}
    for name,_ in arms:
        ms=[r["ms"] for c in cycles if c["arm"]==name for r in c["rows"] if not r["failed"]]
        per_arm[name]=summarize(ms); per_arm[name]["env"]=dict(arms)[name]
    write_out(args.out,{"mode":"ab","cycles_requested":args.cycles,"n_per_cycle":args.n,"dtype":args.dtype,"arms":dict(arms),"per_arm":per_arm,"cycles":cycles}); print(json.dumps(per_arm,indent=1))


def cmd_subs(args):
    driver=Path(args.log+".subs-driver.py")
    extra = ", '--require-chat-worker'" if args.require_chat_worker else ""
    driver.write_text("import sys\nsys.path.insert(0, %r)\nsys.argv=['head_probe','phases','--ckpt',%r,'--suite',%r,'--n',%d,'--out',%r%s]\nexec(open(%r).read())\n"%(str(REPO/"serve"),args.ckpt,args.suite,args.n,args.out,extra,str(Path(__file__).resolve())))
    env=dict(os.environ); env["MLX_OMARCHY_GPU_PROFILE"]=args.profile; env["MLX_OMARCHY_GPU_PROFILE_LABEL"]=args.label
    rc=subprocess.run([sys.executable,str(driver)],env=env,capture_output=True,text=True,timeout=1200); print(rc.stdout[-2000:])
    if rc.returncode: print(rc.stderr[-2000:],file=sys.stderr); sys.exit(rc.returncode)
    import importlib.util
    spec=importlib.util.spec_from_file_location("subcap_calibrate",REPO/"scripts"/"subcap_calibrate.py"); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    subs_ms=mod.ms_per_submit(args.profile); tail=subs_ms[-20:]
    report={"n_submissions_total":len(subs_ms),"tail20":{"n":len(tail),"p50":mod.percentile(tail,50),"p95":mod.percentile(tail,95),"max":max(tail) if tail else None}}
    write_out(args.out+".subs.json",report); print(json.dumps(report,indent=1))


def cmd_answers(args):
    cases=json.loads(Path(args.suite).read_text())["cases"]; texts=[c["text"] for c in cases]; before=host_context(); require_quiet(before); require_chat_worker(before,args.require_chat_worker)
    proc=spawn_server(sys.executable,args.ckpt,args.port,args.dtype,dict(os.environ),args.log)
    try:
        url="http://127.0.0.1:%d/v1/decisions"%args.port
        for i in range(5): call_decision(url,payload_for(texts[i%len(texts)]),5.0)
        records=[]
        for case in cases:
            raw,ms,failed=call_decision(url,payload_for(case["text"]),5.0); answer=(raw.get("answers") or {}).get("route")
            records.append({"id":case["id"],"category":case.get("category"),"expected":case.get("expected"),"latency_ms":ms,"failed":failed,"answer":answer})
    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill()
    after=host_context(); write_out(args.out,{"mode":"answers","suite":str(args.suite),"dtype":args.dtype,"before":before,"after":after,"quiet_valid":before["quiet_eligible"] and after["quiet_eligible"],"co_served":bool(before["chat_workers"]) and bool(after["chat_workers"]),"records":records})
    print(json.dumps(summarize([r["latency_ms"] for r in records if not r["failed"]]),indent=1))


def policy3_route(answer):
    sys.path.insert(0, str(REPO / "serve"))
    from mlx_omarchy_assistant.routing import ROUTING_POLICY, decide_route
    return decide_route(answer, ROUTING_POLICY)


def cmd_compare(args):
    ref=json.loads(Path(args.ref).read_text())["records"]
    cand=json.loads(Path(args.cand).read_text())["records"]
    ref_ids=[r["id"] for r in ref]
    cand_ids=[r["id"] for r in cand]
    if len(set(ref_ids)) != len(ref_ids) or len(set(cand_ids)) != len(cand_ids):
        raise ValueError("duplicate case ids in reference or candidate")
    if set(ref_ids) != set(cand_ids):
        raise ValueError("case-id sets differ")
    if any(r.get("failed") or not r.get("answer") for r in ref):
        raise ValueError("reference contains missing or failed answers")
    by_id={r["id"]:r for r in cand}; mismatches=[]; route_mismatches=[]; deltas=[]
    for r in ref:
        c=by_id.get(r["id"])
        if c is None or c.get("failed") or not c.get("answer"):
            mismatches.append({"id":r["id"],"why":"candidate missing/failed"}); continue
        ra,ca=r["answer"],c["answer"]
        if ra["choice"]!=ca["choice"]: mismatches.append({"id":r["id"],"ref":ra["choice"],"cand":ca["choice"]})
        rr,cr=policy3_route(ra),policy3_route(ca)
        if rr!=cr: route_mismatches.append({"id":r["id"],"ref":rr,"cand":cr})
        for k in ra["probabilities"]: deltas.append(abs(ra["probabilities"][k]-ca["probabilities"][k]))
        deltas.append(abs(ra["rl_agent"]["act_probability"]-ca["rl_agent"]["act_probability"]))
    out={"n_ref":len(ref),"choice_mismatches":mismatches,"route_mismatches":route_mismatches,"decisions_equal":not mismatches and not route_mismatches,"max_prob_delta":max(deltas) if deltas else None,"mean_prob_delta":statistics.fmean(deltas) if deltas else None}
    write_out(args.out,out); print(json.dumps(out,indent=1))


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    def common(p):
        p.add_argument("--ckpt",required=True); p.add_argument("--suite",required=True); p.add_argument("--out",required=True); p.add_argument("--log",default="/tmp/head_probe-server.log")
        p.add_argument("--require-chat-worker",action="store_true")
    p=sub.add_parser("phases"); common(p); p.add_argument("--n",type=int,default=40); p.set_defaults(fn=cmd_phases)
    p=sub.add_parser("serve"); common(p); p.add_argument("--n",type=int,default=100); p.add_argument("--dtype",default="float16"); p.add_argument("--port",type=int,default=50501); p.set_defaults(fn=cmd_serve)
    p=sub.add_parser("ab"); common(p); p.add_argument("--arms",required=True); p.add_argument("--cycles",type=int,default=3); p.add_argument("--n",type=int,default=30); p.add_argument("--dtype",default="float16"); p.add_argument("--port",type=int,default=50501); p.set_defaults(fn=cmd_ab)
    p=sub.add_parser("subs"); common(p); p.add_argument("--n",type=int,default=12); p.add_argument("--profile",default="/tmp/head-probe-profile"); p.add_argument("--label",default="head"); p.set_defaults(fn=cmd_subs)
    p=sub.add_parser("answers"); common(p); p.add_argument("--dtype",default="float16"); p.add_argument("--port",type=int,default=50501); p.set_defaults(fn=cmd_answers)
    p=sub.add_parser("compare"); p.add_argument("--ref",required=True); p.add_argument("--cand",required=True); p.add_argument("--out",required=True); p.set_defaults(fn=cmd_compare)
    args=ap.parse_args(); args.fn(args)

if __name__=="__main__": main()
