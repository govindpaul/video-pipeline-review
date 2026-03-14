#!/usr/bin/env python3
"""
Validator benchmark runner.

Tests scene and prompt validators against labeled fixture cases.
Measures: parse success rate, PASS/FAIL agreement, false FAIL/PASS rates.

Usage:
  python tests/benchmarks/validators/run_benchmark.py [--max-tokens 8192]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pipeline.llm import local as llm
from pipeline.rubrics import load_rubric, format_rubric_text
from pipeline.validators.scene import _parse_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_scene_benchmark(max_tokens: int = 8192):
    """Run scene validator benchmark against labeled cases."""
    benchmark_file = Path(__file__).parent / "scene_benchmark.json"
    with open(benchmark_file) as f:
        benchmark = json.load(f)

    rubric = load_rubric("scene")
    rubric_text = format_rubric_text(rubric)
    model = "qwen3.5:35b-a3b"

    llm.ensure_running()

    results = []
    total = len(benchmark["cases"])

    print(f"\n{'='*60}")
    print(f"Scene Validator Benchmark — {total} cases, max_tokens={max_tokens}")
    print(f"{'='*60}\n")

    for i, case in enumerate(benchmark["cases"]):
        case_id = case["id"]
        expected = case["label"]  # PASS, FAIL, or AMBIGUOUS

        scenes_text = ""
        for s in case["scenes"]:
            scenes_text += f"Scene {s['num']}: {s['title']} ({s['duration']}s)\n"

        estimated_runtime = sum(s["duration"] for s in case["scenes"])

        prompt = rubric["prompt_template"].format(
            title=case["title"],
            summary=case["summary"],
            scene_count=len(case["scenes"]),
            estimated_runtime=estimated_runtime,
            scenes_text=scenes_text,
            rubric_text=rubric_text,
        )

        start = time.time()
        try:
            response = llm.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                format_json=False,
                max_tokens=max_tokens,
            )

            data = _parse_json(response)
            elapsed = time.time() - start

            if data is None:
                actual = "PARSE_FAIL"
                score = None
                confidence = None
            else:
                actual = data.get("state", "UNKNOWN").upper()
                confidence = data.get("confidence", 0)

            result = {
                "id": case_id,
                "expected": expected,
                "actual": actual,
                "confidence": confidence,
                "parse_ok": data is not None,
                "elapsed": round(elapsed, 1),
                "response_length": len(response),
                "raw_response": response[:300],
            }

            # Agreement check
            if expected == "AMBIGUOUS":
                match = "N/A"
            elif actual == "PARSE_FAIL":
                match = "PARSE_FAIL"
            elif actual == expected:
                match = "CORRECT"
            else:
                match = "WRONG"

            result["match"] = match
            results.append(result)

            symbol = {"CORRECT": "+", "WRONG": "X", "N/A": "~", "PARSE_FAIL": "!"}[match]
            print(
                f"  [{symbol}] {case_id:12s} expected={expected:9s} actual={actual:10s} "
                f"conf={confidence or 0:.2f} {elapsed:.1f}s"
            )

        except Exception as e:
            elapsed = time.time() - start
            results.append({
                "id": case_id,
                "expected": expected,
                "actual": "ERROR",
                "parse_ok": False,
                "elapsed": round(elapsed, 1),
                "match": "ERROR",
                "error": str(e),
            })
            print(f"  [E] {case_id}: ERROR {e}")

    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    parse_ok = sum(1 for r in results if r["parse_ok"])
    parse_fail = sum(1 for r in results if not r["parse_ok"])
    print(f"Parse success rate: {parse_ok}/{total} ({parse_ok/total*100:.0f}%)")

    # Agreement on labeled cases (excluding AMBIGUOUS)
    labeled = [r for r in results if r["expected"] != "AMBIGUOUS"]
    correct = sum(1 for r in labeled if r["match"] == "CORRECT")
    wrong = sum(1 for r in labeled if r["match"] == "WRONG")
    parse_fails = sum(1 for r in labeled if r["match"] == "PARSE_FAIL")

    print(f"Agreement (PASS/FAIL): {correct}/{len(labeled)} ({correct/len(labeled)*100:.0f}%)")
    print(f"  Correct: {correct}")
    print(f"  Wrong:   {wrong}")
    print(f"  Parse failures: {parse_fails}")

    # False rates
    good_cases = [r for r in results if r["expected"] == "PASS"]
    bad_cases = [r for r in results if r["expected"] == "FAIL"]

    false_fail = sum(1 for r in good_cases if r["actual"] == "FAIL")
    false_pass = sum(1 for r in bad_cases if r["actual"] == "PASS")

    print(f"\nFalse FAIL rate (good labeled PASS, validator said FAIL): {false_fail}/{len(good_cases)}")
    print(f"False PASS rate (bad labeled FAIL, validator said PASS):  {false_pass}/{len(bad_cases)}")

    # Ambiguous cases
    ambig = [r for r in results if r["expected"] == "AMBIGUOUS"]
    if ambig:
        ambig_pass = sum(1 for r in ambig if r["actual"] == "PASS")
        ambig_fail = sum(1 for r in ambig if r["actual"] == "FAIL")
        print(f"\nAmbiguous cases: {ambig_pass} PASS, {ambig_fail} FAIL (no right answer)")

    # Save full results
    output_file = Path(__file__).parent / f"scene_benchmark_results_t{max_tokens}.json"
    with open(output_file, "w") as f:
        json.dump({
            "max_tokens": max_tokens,
            "total": total,
            "parse_success_rate": parse_ok / total,
            "agreement_rate": correct / len(labeled) if labeled else 0,
            "false_fail_rate": false_fail / len(good_cases) if good_cases else 0,
            "false_pass_rate": false_pass / len(bad_cases) if bad_cases else 0,
            "results": results,
        }, f, indent=2)
    print(f"\nFull results saved to: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    run_scene_benchmark(max_tokens=args.max_tokens)
