"""
Gate: the JavaScript actually shipped in docs/demo.html must reproduce
scikit-learn's predict_proba.

This does not test a Python transcription of the browser code. It reads
docs/demo.html, cuts the scoring functions out of the shipped <script> block by
brace matching, and executes those exact characters under Node against every
applicant in the test split. If someone edits the demo's arithmetic, this fails.

Checks:
  1. interp/score reproduce CalibratedClassifierCV.predict_proba (both models)
  2. the local attribution decomposition is exact:
     intercept + sum(coef*z_mean) + sum(phi) == decision_function margin
  3. docs/model.json is in sync with a fresh retrain at the same seed

Run:  python verify_web_demo.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(BASE, "src"))

from generate_data import GOVERNED_FEATURES, NAIVE_FEATURES          # noqa: E402
from train import load, split                                        # noqa: E402
from export_web_model import fit_logistic, extract                   # noqa: E402

DEMO = os.path.join(BASE, "docs", "demo.html")
MODEL_JSON = os.path.join(BASE, "docs", "model.json")
TOL = 1e-12


def extract_function(src, name):
    """Cut `function <name>(...) { ... }` out of the page by brace matching."""
    start = src.index("function " + name + "(")
    i = src.index("{", start)
    depth, j = 0, i
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[start:j + 1]
        j += 1
    raise ValueError("unbalanced braces extracting " + name)


def main():
    src = open(DEMO, encoding="utf-8").read()
    js_funcs = "\n".join(extract_function(src, n) for n in ("interp", "score", "contributions"))
    print(f"extracted {len(js_funcs):,} chars of scoring JS from docs/demo.html")

    shipped = json.load(open(MODEL_JSON, encoding="utf-8"))

    df = load()
    train, val, test = split(df)

    failures = 0
    for name, feats in (("naive", NAIVE_FEATURES), ("governed", GOVERNED_FEATURES)):
        Xtr, ytr = train[feats], train["advanced_to_onsite"]
        pipe, cal = fit_logistic(Xtr, ytr, val[feats], val["advanced_to_onsite"])
        fresh = extract(pipe, cal, feats, Xtr)

        # ---- check 3: shipped model.json matches a fresh retrain ----
        ship = shipped["models"][name]
        for key in ("coef", "mean", "scale", "iso_x", "iso_y", "z_mean"):
            drift = float(np.max(np.abs(np.array(ship[key]) - np.array(fresh[key]))))
            if drift > 1e-12:
                print(f"  FAIL {name}.{key} drifted from retrain by {drift:.3e}")
                failures += 1
        if abs(ship["intercept"] - fresh["intercept"]) > 1e-12:
            print(f"  FAIL {name}.intercept drifted")
            failures += 1

        # ---- reference probabilities from scikit-learn ----
        Xte = test[feats].to_numpy(dtype=float)
        ref = cal.predict_proba(test[feats])[:, 1]
        margin_ref = pipe.decision_function(test[feats])

        with tempfile.TemporaryDirectory() as td:
            vec_path = os.path.join(td, "vec.json")
            out_path = os.path.join(td, "out.json")
            js_path = os.path.join(td, "run.js")
            with open(vec_path, "w") as fh:
                json.dump([dict(zip(feats, row)) for row in Xte.tolist()], fh)

            runner = f"""
{js_funcs}
const fs = require("fs");
const M = JSON.parse(fs.readFileSync({json.dumps(MODEL_JSON)}, "utf8")).models[{json.dumps(name)}];
const rows = JSON.parse(fs.readFileSync({json.dumps(vec_path)}, "utf8"));
const p = [], margin = [], phisum = [];
for (const r of rows) {{
  const s = score(M, r);
  p.push(s.p);
  margin.push(s.margin);
  let t = 0;
  for (const c of contributions(M, s.z)) t += c.phi;
  phisum.push(t);
}}
fs.writeFileSync({json.dumps(out_path)}, JSON.stringify({{p, margin, phisum}}));
"""
            with open(js_path, "w") as fh:
                fh.write(runner)

            res = subprocess.run(["node", js_path], capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  FAIL node exited {res.returncode}: {res.stderr.strip()[:400]}")
                failures += 1
                continue
            got = json.load(open(out_path, encoding="utf-8"))

        # ---- check 1: probability parity ----
        err_p = float(np.max(np.abs(np.array(got["p"]) - ref)))
        err_m = float(np.max(np.abs(np.array(got["margin"]) - margin_ref)))

        # ---- check 2: attribution decomposition is exact ----
        base = fresh["intercept"] + float(np.dot(fresh["coef"], fresh["z_mean"]))
        recon = base + np.array(got["phisum"])
        err_a = float(np.max(np.abs(recon - margin_ref)))

        ok = err_p <= TOL and err_m <= TOL and err_a <= 1e-10
        failures += 0 if ok else 1
        print(f"  {name:<9} n={len(ref):,}  "
              f"max|p_js - p_sklearn|={err_p:.2e}  "
              f"max|margin diff|={err_m:.2e}  "
              f"max|attribution recon|={err_a:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("All checks passed: the shipped browser JS is numerically identical to scikit-learn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
