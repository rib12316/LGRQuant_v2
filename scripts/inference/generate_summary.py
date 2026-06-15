#!/usr/bin/env python3
"""
Parse LGRQuant inference log(s) and print a results summary table.

Usage:
    python generate_summary.py <log_file> <model_name> <quant_type>
        Parse a single log and print FP16 baseline + this quantized config.

    python generate_summary.py --all [logs_dir]
        Scan all logs in the directory, merge partial results, and print a
        combined comparison table for every model / quantization combo.

Table columns
    Model | Quant | Memory(MB) | PIQA | BoolQ | HellaSwag | WinoGrande | Speed(tok/s)

    Quantized rows show accuracy diffs from FP16 in parentheses,
    e.g. ``78.78(-3.27)`` means 3.27 pp drop from the FP16 baseline.
    Speed shows the multiplier, e.g. ``36.6(1.61x)``.
"""
import re
import sys
import os
import glob
import json


# ═══════════════════════════════════════════════════════════════════════════
# FP16 Baseline Data
# ═══════════════════════════════════════════════════════════════════════════
# These are model-level properties that do not change between runs.
# Override or extend by creating ``fp16_baselines.json`` next to this script.
_FP16_BASELINES = {
    "Qwen2.5-7B-Instruct": {
        "peak_mem_mb": 15104,
        "piqa": 80.20,
        "boolq": 85.72,
        "hellaswag": 79.57,
        "winogrande": 69.93,
        "speed_tps": 43.0,
    },
    "Qwen2.5-14B": {
        "peak_mem_mb": 31529,
        "piqa": 82.05,
        "boolq": 85.26,
        "hellaswag": 82.91,
        "winogrande": 75.30,
        "speed_tps": 22.7,
    },
}

# Expected model -> quantization patterns for ``--all`` mode
_EXPECTED_CONFIGS = [
    ("Qwen2.5-7B-Instruct", [
        ("q2.5-7b-ins-2bit-stage2_*.log", "W2A16"),
        ("q2.5-7b-ins_*.log",             "W4A16"),
    ]),
    ("Qwen2.5-14B", [
        ("q2.5-14b-2bit-stage2_*.log", "W2A16"),
        ("q2.5-14b-4bit_*.log",        "W4A16"),
    ]),
]


def _load_baselines():
    """Load FP16 baselines, optionally overridden by a JSON sidecar file."""
    baselines = {k: dict(v) for k, v in _FP16_BASELINES.items()}
    cfg = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fp16_baselines.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            for model, data in json.load(f).items():
                baselines.setdefault(model, {}).update(data)
    return baselines


# ═══════════════════════════════════════════════════════════════════════════
# Log Parsing
# ═══════════════════════════════════════════════════════════════════════════
def parse_log(log_file):
    """Extract all metrics from an inference log file."""
    with open(log_file) as f:
        content = f.read()

    r = {}

    # FP16: per-token time + peak memory  (from Speedup Summary section)
    m = re.search(
        r'^\s+FP\s+:.*?per-token\s+([0-9.]+)\s*ms.*?'
        r'peak_mem\s+([0-9.]+)\s*GB',
        content, re.MULTILINE)
    if m:
        r['fp_per_token_ms'] = float(m.group(1))
        r['fp_peak_mem_gb']  = float(m.group(2))
        r['fp_peak_mem_mb']  = round(r['fp_peak_mem_gb'] * 1024)
        r['fp_speed_tps']    = 1000.0 / r['fp_per_token_ms']

    # Quantized: kernel label + per-token time + peak memory
    m = re.search(
        r'^\s+(W\d+A\d+)\s+:.*?per-token\s+([0-9.]+)\s*ms.*?'
        r'peak_mem\s+([0-9.]+)\s*GB',
        content, re.MULTILINE)
    if m:
        r['quant_label']    = m.group(1)
        r['q_per_token_ms'] = float(m.group(2))
        r['q_peak_mem_gb']  = float(m.group(3))
        r['q_peak_mem_mb']  = round(r['q_peak_mem_gb'] * 1024)
        r['q_speed_tps']    = 1000.0 / r['q_per_token_ms']

    # Speedup (token) from log
    m = re.search(r'Speedup \(token\) = ([0-9.]+)x', content)
    r['speedup_token'] = float(m.group(1)) if m else None

    # Downstream task accuracies
    for task in ('piqa', 'hellaswag', 'winogrande', 'boolq'):
        m = re.search(rf'{task} acc = ([0-9.]+)%', content)
        r[f'{task}_acc'] = float(m.group(1)) if m else None

    return r


def _merge_all_logs(logs_dir, pattern):
    """Merge metrics from *all* complete logs matching *pattern*.

    Different runs may produce partial results.  We keep the latest
    non-``None`` value for every key so the table is as complete as possible.
    """
    files = sorted(glob.glob(os.path.join(logs_dir, pattern)))
    merged = {}
    for f in files:
        try:
            res = parse_log(f)
            if not any(res.get(k) is not None
                       for k in ('speedup_token', 'piqa_acc', 'q_peak_mem_gb')):
                continue
            for key, val in res.items():
                if val is not None:
                    merged[key] = val          # later file wins
        except Exception:
            continue
    return merged or None


# ═══════════════════════════════════════════════════════════════════════════
# Normalized Row Builders
# ═══════════════════════════════════════════════════════════════════════════
def _fp16_row(model, baseline, fp_log=None):
    """Build the FP16 baseline row for *model*."""
    b = baseline
    mem = (fp_log or {}).get('fp_peak_mem_mb') or b.get('peak_mem_mb')
    spd = (fp_log or {}).get('fp_speed_tps')   or b.get('speed_tps')
    return {
        'model': model, 'quant': 'FP16',
        'mem_mb': mem,
        'piqa': b.get('piqa'), 'boolq': b.get('boolq'),
        'hellaswag': b.get('hellaswag'), 'winogrande': b.get('winogrande'),
        'speed_tps': spd, 'speedup': None,
        'fp_peak_gb': None, 'q_peak_gb': None,
    }


def _quant_row(model, quant, log_data):
    """Build a quantized-result row from parsed log data."""
    return {
        'model': model, 'quant': quant,
        'mem_mb': log_data.get('q_peak_mem_mb'),
        'piqa': log_data.get('piqa_acc'),
        'boolq': log_data.get('boolq_acc'),
        'hellaswag': log_data.get('hellaswag_acc'),
        'winogrande': log_data.get('winogrande_acc'),
        'speed_tps': log_data.get('q_speed_tps'),
        'speedup': log_data.get('speedup_token'),
        'fp_peak_gb': log_data.get('fp_peak_mem_gb'),
        'q_peak_gb':  log_data.get('q_peak_mem_gb'),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Formatting Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _v(val, spec=".0f"):
    """Format a numeric value; return 'N/A' when None."""
    if val is None:
        return "N/A"
    return format(val, spec)


def _fmt_acc(val):
    """Plain accuracy, e.g. ``82.05``."""
    if val is None:
        return "N/A"
    return f"{val:.2f}"


def _fmt_acc_diff(q_val, fp_val):
    """Quantized accuracy with diff from FP16, e.g. ``78.78(-3.27)``."""
    if q_val is None:
        return "N/A"
    s = f"{q_val:.2f}"
    if fp_val is not None:
        diff = round(q_val - fp_val, 2)
        if abs(diff) >= 0.005:
            sign = "+" if diff > 0 else ""
            s += f"({sign}{diff:.2f})"
    return s


def _fmt_mem_diff(q_mem_mb, fp_mem_mb):
    """Quantized memory with reduction rate, e.g. ``10035(-68.17%)``."""
    if q_mem_mb is None:
        return "N/A"
    s = f"{q_mem_mb:.0f}"
    if fp_mem_mb:
        reduction = (fp_mem_mb - q_mem_mb) / fp_mem_mb * 100.0
        s += f"(-{reduction:.2f}%)"
    return s


def _fmt_speed(speed, speedup):
    """Speed with optional multiplier, e.g. ``36.6(1.61x)``."""
    if speed is None:
        return "N/A"
    s = f"{speed:.1f}"
    if speedup is not None:
        s += f"({speedup:.2f}x)"
    return s


# ═══════════════════════════════════════════════════════════════════════════
# Table Printer
# ═══════════════════════════════════════════════════════════════════════════
_COL_W = [22, 8, 18, 14, 14, 14, 14, 16]
_HEADERS = ["Model", "Quant", "Memory(MB)", "PIQA", "BoolQ",
            "HellaSwag", "WinoGrande", "Speed(tok/s)"]


def _hline(fill="-"):
    return "+" + "+".join(fill * (w + 2) for w in _COL_W) + "+"


def _row(cells):
    return "|" + "|".join(f" {c:<{w}} " for c, w in zip(cells, _COL_W)) + "|"


def print_table(model_groups, baselines):
    """Print the grouped summary table.

    *model_groups*: list of ``(model_name, [row_dict, ...])``.
    The first row in each group is always FP16.
    """
    total_w = sum(_COL_W) + 3 * len(_COL_W) + 1

    print()
    print("=" * total_w)
    print(f"{'Inference Results Summary Table':^{total_w}}")
    print("=" * total_w)
    print(_hline("="))
    print(_row(_HEADERS))
    print(_hline("="))

    for model_name, rows in model_groups:
        first = True
        bl = baselines.get(model_name, {})

        for r in rows:
            model_cell = model_name if first else ""
            q = r['quant']

            if q == "FP16":
                cells = [
                    model_cell, q,
                    _v(r['mem_mb']),
                    _fmt_acc(r['piqa']),
                    _fmt_acc(r['boolq']),
                    _fmt_acc(r['hellaswag']),
                    _fmt_acc(r['winogrande']),
                    _v(r['speed_tps'], ".1f"),
                ]
            else:
                cells = [
                    model_cell, q,
                    _fmt_mem_diff(r['mem_mb'], bl.get('peak_mem_mb')),
                    _fmt_acc_diff(r['piqa'],      bl.get('piqa')),
                    _fmt_acc_diff(r['boolq'],     bl.get('boolq')),
                    _fmt_acc_diff(r['hellaswag'], bl.get('hellaswag')),
                    _fmt_acc_diff(r['winogrande'], bl.get('winogrande')),
                    _fmt_speed(r['speed_tps'], r.get('speedup')),
                ]
            print(_row(cells))
            first = False

        print(_hline("-"))

    # ── Calculation Details ──────────────────────────────────────────────
    print("\n  Calculation Details:")
    for model_name, rows in model_groups:
        for r in rows:
            if r['quant'] == "FP16":
                continue
            fp_gb = r.get('fp_peak_gb')
            q_gb  = r.get('q_peak_gb')
            sp    = r.get('speedup')
            if fp_gb and q_gb:
                red = (fp_gb - q_gb) / fp_gb * 100.0
                print(f"    [{model_name} {r['quant']}] "
                      f"Mem Saved = ({fp_gb:.2f} - {q_gb:.2f}) / "
                      f"{fp_gb:.2f} * 100% = {red:.2f}%")
            if sp:
                print(f"    [{model_name} {r['quant']}] "
                      f"Speedup(token) = {sp:.2f}x")


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    baselines = _load_baselines()

    if len(sys.argv) >= 4 and sys.argv[1] != "--all":
        # ── Single-log mode ──────────────────────────────────────────────
        log_file   = sys.argv[1]
        model_name = sys.argv[2]
        quant_type = sys.argv[3]

        if not os.path.exists(log_file):
            print(f"Error: log not found: {log_file}", file=sys.stderr)
            sys.exit(1)

        log_data = parse_log(log_file)
        bl = baselines.get(model_name, {})
        fp16  = _fp16_row(model_name, bl, log_data)
        quant = _quant_row(model_name, quant_type, log_data)
        print_table([(model_name, [fp16, quant])], baselines)

    elif len(sys.argv) >= 2 and sys.argv[1] == "--all":
        # ── All-logs mode ────────────────────────────────────────────────
        logs_dir = (
            sys.argv[2] if len(sys.argv) >= 3 and not sys.argv[2].startswith("-")
            else os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'logs')))

        model_groups = []
        for model_name, patterns in _EXPECTED_CONFIGS:
            bl = baselines.get(model_name, {})

            # Collect FP16 benchmark data from whichever quant logs have it
            all_fp16_log = {}
            quant_rows = []
            for pattern, quant in patterns:
                merged = _merge_all_logs(logs_dir, pattern)
                if merged:
                    for k in ('fp_peak_mem_mb', 'fp_speed_tps',
                              'fp_per_token_ms', 'fp_peak_mem_gb'):
                        if merged.get(k) is not None:
                            all_fp16_log[k] = merged[k]
                    quant_rows.append(_quant_row(model_name, quant, merged))
                else:
                    quant_rows.append(_quant_row(model_name, quant, {}))

            fp16 = _fp16_row(model_name, bl, all_fp16_log)
            model_groups.append((model_name, [fp16] + quant_rows))

        print_table(model_groups, baselines)

    else:
        prog = os.path.basename(sys.argv[0])
        print("Usage:")
        print(f"  python {prog} <log_file> <model_name> <quant_type>")
        print(f"  python {prog} --all [logs_dir]")
        sys.exit(1)


if __name__ == "__main__":
    main()
