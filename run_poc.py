"""
OpenQA POC orchestrator.
Flow per use case:  YAML (English) -> Planner LLM -> Playwright Executor
                    -> Validator LLM -> Markdown + JSON report

Usage:
  export NVIDIA_API_KEY=nvapi-xxxx          # from build.nvidia.com
  python run_poc.py                          # all 3 use cases, headless
  python run_poc.py --headed                 # watch the browser
  python run_poc.py --only UC1               # run one use case
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from core.nim_client import NimClient
from core.planner import build_plan
from core.executor import Executor
from core.validator import validate

REPORT_DIR = Path("reports")


def render_markdown(results: list) -> str:
    lines = [f"# OpenQA POC Report - {datetime.now():%Y-%m-%d %H:%M}",
             "", "Engine: NVIDIA Build (NIM) | Runner: Playwright | Pattern: Plan -> Execute -> Validate", ""]
    for r in results:
        v = r["validation"]
        lines.append(f"## {r['use_case']} — **{v.get('verdict','?')}**")
        lines.append(f"- Execution: {r['execution']['execution_status']}")
        if r["execution"]["heals"]:
            for h in r["execution"]["heals"]:
                lines.append(f"- Self-healed selector: `{h['old']}` -> `{h['new']}` ({h['reason']})")
        for rule in v.get("rules", []):
            lines.append(f"- [{rule['result']}] {rule['rule']} — _{rule.get('evidence','')}_")
        lines.append(f"\n> {v.get('summary','')}\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headed", action="store_true", help="run with visible browser")
    ap.add_argument("--only", help="run a single use case by name prefix, e.g. UC2")
    args = ap.parse_args()

    nim = NimClient()
    use_cases = yaml.safe_load(Path("usecases/amazon_usecases.yaml").read_text())
    if args.only:
        use_cases = [u for u in use_cases if u["name"].startswith(args.only)]
        if not use_cases:
            sys.exit(f"No use case matching '{args.only}'")

    REPORT_DIR.mkdir(exist_ok=True)
    results = []

    for uc in use_cases:
        print(f"\n=== {uc['name']} ===")
        print("  [1/3] Planning with NIM ...")
        plan = build_plan(nim, uc)
        (REPORT_DIR / f"{uc['name']}_plan.json").write_text(
            json.dumps(plan, indent=2), encoding="utf-8"
        )
        print(f"        {len(plan['steps'])} steps generated")

        print("  [2/3] Executing with Playwright ...")
        import os
        env_headless = os.getenv("HEADLESS", "true").lower() != "false"
        slow_mo = int(os.getenv("SLOW_MO", "150"))
        execu = Executor(nim, headless=(not args.headed) and env_headless, slow_mo=slow_mo)
        exec_result = execu.run(plan)

        print("  [3/3] Validating with NIM ...")
        verdict = validate(nim, uc, exec_result)
        print(f"        VERDICT: {verdict.get('verdict')}  - {verdict.get('summary','')}")

        results.append({"use_case": uc["name"], "execution": exec_result, "validation": verdict})
        time.sleep(3)  # be polite to the free-tier rate limit between cases

    (REPORT_DIR / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = render_markdown(results)
    (REPORT_DIR / "report.md").write_text(md, encoding="utf-8")
    print(f"\nDone. Reports in {REPORT_DIR}/ (report.md, results.json, screenshots)")
    failed = any(r["validation"].get("verdict") != "PASS" for r in results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
