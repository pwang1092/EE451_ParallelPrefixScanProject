#!/usr/bin/env python3

import argparse
import csv
import os
import re
from collections import defaultdict


KERNEL_ORDER = {
    "warp_shuffle": 0,
    "blelloch": 1,
    "hillis_steele": 2,
}


def parse_float(text):
    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return None
    lower = value.lower()
    if lower in {"nan", "n/a", "na", "none", "-", "--"}:
        return None
    value = value.replace(",", "").replace("%", "")
    try:
        return float(value)
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None


def fmt(value, decimals=4):
    if value is None:
        return "NA"
    return f"{value:.{decimals}f}"


def safe_div(num, den):
    if num is None or den is None or den == 0:
        return None
    return num / den


def chunk_size_for_d(d):
    if d <= 16:
        return 512
    if d <= 64:
        return 128
    if d <= 256:
        return 32
    return 16


def parse_timing_csv(path):
    timing = {}
    if not os.path.exists(path):
        return timing
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                kernel = row["kernel"].strip()
                d = int(row["D"])
                l = int(row["L"])
            except (KeyError, ValueError):
                continue
            key = (kernel, d, l)
            timing[key] = {
                "time_ms": parse_float(row.get("time_ms")),
                "correct": int(parse_float(row.get("correct")) or 0),
                "throughput_GB_s": parse_float(row.get("throughput_GB_s")),
            }
    return timing


def parse_ncu_filename(path):
    name = os.path.basename(path)
    match = re.match(r"^(?P<kernel>.+)_D(?P<D>\d+)_L(?P<L>\d+)\.csv$", name)
    if not match:
        return None
    return match.group("kernel"), int(match.group("D")), int(match.group("L"))


def parse_ncu_raw_csv(path):
    launches = {}
    header = None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            cells = [c.strip() for c in row]
            if "Metric Name" in cells and "Metric Value" in cells:
                header = cells
                continue

            if header is None:
                continue
            if len(cells) < len(header):
                continue

            record = {header[i]: cells[i] for i in range(len(header))}
            lower = {k.lower(): v for k, v in record.items()}

            metric_name = lower.get("metric name") or lower.get("metric name / label")
            metric_value_text = lower.get("metric value") or lower.get("value")
            if not metric_name or metric_value_text is None:
                continue

            metric_value = parse_float(metric_value_text)
            if metric_value is None:
                continue

            launch_id = lower.get("id") or lower.get("kernel id") or "unknown"
            kernel_name = lower.get("kernel name") or lower.get("kernel name / demangled") or "unknown"
            launch_key = f"{launch_id}|{kernel_name}"

            if launch_key not in launches:
                launches[launch_key] = {
                    "__launch_id__": launch_id,
                    "__kernel_name__": kernel_name,
                }
            launches[launch_key][metric_name] = metric_value

    return list(launches.values())


def find_first_exact(metric_names, candidates):
    for name in candidates:
        if name in metric_names:
            return name
    return None


def find_first_contains(metric_names, required_tokens):
    for name in sorted(metric_names):
        if all(token in name for token in required_tokens):
            return name
    return None


def aggregate_launches(launches):
    if not launches:
        return {}

    metric_names = set()
    for launch in launches:
        metric_names.update(k for k in launch.keys() if not k.startswith("__"))

    duration_metric = find_first_exact(
        metric_names,
        [
            "gpu__time_duration.sum",
            "gpu__time_duration.avg",
        ],
    )
    if duration_metric is None:
        duration_metric = find_first_contains(metric_names, ["time_duration", "gpu__"])

    def values(name):
        if name is None:
            return [], []
        vals = []
        weights = []
        for launch in launches:
            if name not in launch:
                continue
            vals.append(launch[name])
            if duration_metric and duration_metric in launch and launch[duration_metric] > 0:
                weights.append(launch[duration_metric])
            else:
                weights.append(1.0)
        return vals, weights

    def weighted_avg(name):
        vals, weights = values(name)
        if not vals:
            return None
        wsum = sum(weights)
        if wsum > 0:
            return sum(v * w for v, w in zip(vals, weights)) / wsum
        return sum(vals) / len(vals)

    def metric_sum(name):
        vals, _ = values(name)
        if not vals:
            return None
        return sum(vals)

    def metric_max(name):
        vals, _ = values(name)
        if not vals:
            return None
        return max(vals)

    sm_util_name = find_first_exact(
        metric_names,
        ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
    ) or find_first_contains(metric_names, ["sm__throughput", "pct_of_peak"])

    dram_util_name = find_first_exact(
        metric_names,
        ["dram__throughput.avg.pct_of_peak_sustained_elapsed"],
    ) or find_first_contains(metric_names, ["dram__throughput", "pct_of_peak"])

    occ_name = find_first_exact(
        metric_names,
        ["sm__warps_active.avg.pct_of_peak_sustained_active"],
    ) or find_first_contains(metric_names, ["sm__warps_active", "pct_of_peak"])

    warp_ratio_name = find_first_exact(
        metric_names,
        ["smsp__thread_inst_executed_per_inst_executed.ratio"],
    ) or find_first_contains(metric_names, ["thread_inst_executed_per_inst_executed", "ratio"])

    dram_bytes_name = find_first_exact(metric_names, ["dram__bytes.sum"])
    dram_read_bytes_name = find_first_exact(metric_names, ["dram__bytes_read.sum"])
    dram_write_bytes_name = find_first_exact(metric_names, ["dram__bytes_write.sum"])

    fadd_name = find_first_exact(metric_names, ["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"]) \
        or find_first_contains(metric_names, ["op_fadd", "pred_on", "sum"])
    fmul_name = find_first_exact(metric_names, ["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"]) \
        or find_first_contains(metric_names, ["op_fmul", "pred_on", "sum"])
    ffma_name = find_first_exact(metric_names, ["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"]) \
        or find_first_contains(metric_names, ["op_ffma", "pred_on", "sum"])

    regs_name = find_first_exact(metric_names, ["launch__registers_per_thread"]) \
        or find_first_contains(metric_names, ["launch__registers_per_thread"])
    shmem_name = find_first_exact(metric_names, ["launch__shared_mem_per_block_allocated"]) \
        or find_first_contains(metric_names, ["launch__shared_mem_per_block"])
    block_size_name = find_first_exact(metric_names, ["launch__block_size"]) \
        or find_first_contains(metric_names, ["launch__block_size"])

    l1_hit_name = find_first_exact(metric_names, ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"]) \
        or find_first_contains(metric_names, ["l1tex__", "op_ld", "lookup_hit"])
    l1_miss_name = find_first_exact(metric_names, ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"]) \
        or find_first_contains(metric_names, ["l1tex__", "op_ld", "lookup_miss"])

    l2_hit_name = find_first_exact(metric_names, ["lts__t_sectors_srcunit_lsu_op_read_lookup_hit.sum"]) \
        or find_first_contains(metric_names, ["lts__", "op_read", "lookup_hit"])
    l2_miss_name = find_first_exact(metric_names, ["lts__t_sectors_srcunit_lsu_op_read_lookup_miss.sum"]) \
        or find_first_contains(metric_names, ["lts__", "op_read", "lookup_miss"])

    dram_bytes = metric_sum(dram_bytes_name)
    if dram_bytes is None:
        read_bytes = metric_sum(dram_read_bytes_name)
        write_bytes = metric_sum(dram_write_bytes_name)
        if read_bytes is not None or write_bytes is not None:
            dram_bytes = (read_bytes or 0.0) + (write_bytes or 0.0)

    fadd = metric_sum(fadd_name) or 0.0
    fmul = metric_sum(fmul_name) or 0.0
    ffma = metric_sum(ffma_name) or 0.0
    flops = fadd + fmul + 2.0 * ffma
    if fadd == 0.0 and fmul == 0.0 and ffma == 0.0:
        flops = None

    l1_hits = metric_sum(l1_hit_name)
    l1_misses = metric_sum(l1_miss_name)
    l2_hits = metric_sum(l2_hit_name)
    l2_misses = metric_sum(l2_miss_name)

    l1_hit_rate = None
    if l1_hits is not None and l1_misses is not None and (l1_hits + l1_misses) > 0:
        l1_hit_rate = 100.0 * l1_hits / (l1_hits + l1_misses)

    l2_hit_rate = None
    if l2_hits is not None and l2_misses is not None and (l2_hits + l2_misses) > 0:
        l2_hit_rate = 100.0 * l2_hits / (l2_hits + l2_misses)

    warp_threads_per_inst = weighted_avg(warp_ratio_name)
    warp_efficiency = None
    if warp_threads_per_inst is not None:
        warp_efficiency = max(0.0, min(100.0, 100.0 * warp_threads_per_inst / 32.0))

    return {
        "sm_util_pct": weighted_avg(sm_util_name),
        "dram_util_pct": weighted_avg(dram_util_name),
        "achieved_occupancy_pct": weighted_avg(occ_name),
        "warp_threads_per_inst": warp_threads_per_inst,
        "warp_efficiency_pct": warp_efficiency,
        "dram_bytes": dram_bytes,
        "flops": flops,
        "registers_per_thread_peak": metric_max(regs_name),
        "shared_mem_per_block_peak_bytes": metric_max(shmem_name),
        "block_size_peak": metric_max(block_size_name),
        "l1_hit_rate_pct": l1_hit_rate,
        "l2_hit_rate_pct": l2_hit_rate,
    }


def group_mean(rows, key_fields, value_fields):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        groups[key].append(row)

    out = []
    for key, members in sorted(groups.items(), key=lambda x: (x[0][1], KERNEL_ORDER.get(x[0][0], 999))):
        item = {k: v for k, v in zip(key_fields, key)}
        for field in value_fields:
            vals = [m[field] for m in members if m.get(field) is not None]
            item[field] = (sum(vals) / len(vals)) if vals else None
        out.append(item)
    return out


def sort_key(row):
    return (
        row["D"],
        row["L"],
        KERNEL_ORDER.get(row["kernel"], 999),
        row["kernel"],
    )


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Analyze timing + Nsight Compute CSV outputs")
    parser.add_argument("--timing_csv", required=True)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--peak_bw_gbs", type=float, default=1555.0,
                        help="Peak DRAM bandwidth in GB/s (default: 1555)")
    parser.add_argument("--peak_fp32_gflops", type=float, default=19500.0,
                        help="Peak FP32 throughput in GFLOP/s (default: 19500)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    timing = parse_timing_csv(args.timing_csv)

    ncu_data = {}
    if os.path.isdir(args.raw_dir):
        for name in os.listdir(args.raw_dir):
            if not name.endswith(".csv"):
                continue
            full_path = os.path.join(args.raw_dir, name)
            parsed = parse_ncu_filename(full_path)
            if not parsed:
                continue
            kernel, d, l = parsed
            launches = parse_ncu_raw_csv(full_path)
            ncu_data[(kernel, d, l)] = aggregate_launches(launches)

    all_keys = set(timing.keys()) | set(ncu_data.keys())
    rows = []
    ridge_ai = safe_div(args.peak_fp32_gflops, args.peak_bw_gbs)

    for key in sorted(all_keys, key=lambda k: (k[1], k[2], KERNEL_ORDER.get(k[0], 999), k[0])):
        kernel, d, l = key
        time_rec = timing.get(key, {})
        ncu_rec = ncu_data.get(key, {})

        row = {
            "kernel": kernel,
            "D": d,
            "L": l,
            "chunk_size": chunk_size_for_d(d),
            "time_ms": time_rec.get("time_ms"),
            "correct": time_rec.get("correct"),
            "throughput_GB_s_timing": time_rec.get("throughput_GB_s"),
            "sm_util_pct": ncu_rec.get("sm_util_pct"),
            "dram_util_pct": ncu_rec.get("dram_util_pct"),
            "achieved_occupancy_pct": ncu_rec.get("achieved_occupancy_pct"),
            "warp_efficiency_pct": ncu_rec.get("warp_efficiency_pct"),
            "registers_per_thread_peak": ncu_rec.get("registers_per_thread_peak"),
            "shared_mem_per_block_peak_bytes": ncu_rec.get("shared_mem_per_block_peak_bytes"),
            "block_size_peak": ncu_rec.get("block_size_peak"),
            "dram_bytes": ncu_rec.get("dram_bytes"),
            "flops": ncu_rec.get("flops"),
            "l1_hit_rate_pct": ncu_rec.get("l1_hit_rate_pct"),
            "l2_hit_rate_pct": ncu_rec.get("l2_hit_rate_pct"),
        }

        row["resident_warps_est"] = None
        if row["achieved_occupancy_pct"] is not None:
            row["resident_warps_est"] = 64.0 * row["achieved_occupancy_pct"] / 100.0

        row["ai_flop_per_byte"] = safe_div(row["flops"], row["dram_bytes"])

        time_s = None
        if row["time_ms"] is not None:
            time_s = row["time_ms"] * 1e-3

        row["gflops"] = None
        if row["flops"] is not None and time_s and time_s > 0:
            row["gflops"] = row["flops"] / time_s / 1e9

        row["dram_bw_gb_s"] = None
        if row["dram_bytes"] is not None and time_s and time_s > 0:
            row["dram_bw_gb_s"] = row["dram_bytes"] / time_s / 1e9

        row["bandwidth_util_pct"] = row["dram_util_pct"]
        if row["bandwidth_util_pct"] is None and row["dram_bw_gb_s"] is not None:
            row["bandwidth_util_pct"] = 100.0 * row["dram_bw_gb_s"] / args.peak_bw_gbs

        row["roofline_bound"] = None
        if row["ai_flop_per_byte"] is not None and ridge_ai is not None:
            row["roofline_bound"] = "memory-bound" if row["ai_flop_per_byte"] < ridge_ai else "compute-bound"

        row["roofline_limit_gflops"] = None
        if row["ai_flop_per_byte"] is not None:
            row["roofline_limit_gflops"] = min(args.peak_fp32_gflops,
                                               row["ai_flop_per_byte"] * args.peak_bw_gbs)

        row["roofline_efficiency_pct"] = safe_div(100.0 * (row["gflops"] or 0.0),
                                                    row["roofline_limit_gflops"]) \
            if row["gflops"] is not None and row["roofline_limit_gflops"] not in (None, 0.0) else None

        row["utilization_bound"] = None
        if row["sm_util_pct"] is not None and row["bandwidth_util_pct"] is not None:
            if row["bandwidth_util_pct"] > row["sm_util_pct"] + 5.0:
                row["utilization_bound"] = "memory-bound"
            elif row["sm_util_pct"] > row["bandwidth_util_pct"] + 5.0:
                row["utilization_bound"] = "compute-bound"
            else:
                row["utilization_bound"] = "mixed"

        rows.append(row)

    rows.sort(key=sort_key)

    merged_csv = os.path.join(args.out_dir, "merged_metrics.csv")
    merged_fields = [
        "kernel", "D", "L", "chunk_size", "time_ms", "correct", "throughput_GB_s_timing",
        "sm_util_pct", "dram_util_pct", "achieved_occupancy_pct", "resident_warps_est",
        "warp_efficiency_pct", "registers_per_thread_peak", "shared_mem_per_block_peak_bytes",
        "block_size_peak", "dram_bytes", "flops", "ai_flop_per_byte", "gflops", "dram_bw_gb_s",
        "bandwidth_util_pct", "l1_hit_rate_pct", "l2_hit_rate_pct", "roofline_bound",
        "roofline_limit_gflops", "roofline_efficiency_pct", "utilization_bound",
    ]
    write_csv(merged_csv, rows, merged_fields)

    occ_summary = group_mean(
        rows,
        key_fields=["kernel", "D"],
        value_fields=[
            "achieved_occupancy_pct",
            "resident_warps_est",
            "warp_efficiency_pct",
            "registers_per_thread_peak",
            "shared_mem_per_block_peak_bytes",
            "block_size_peak",
            "sm_util_pct",
            "dram_util_pct",
            "bandwidth_util_pct",
        ],
    )
    occ_summary_csv = os.path.join(args.out_dir, "occupancy_summary.csv")
    write_csv(
        occ_summary_csv,
        occ_summary,
        [
            "kernel", "D", "achieved_occupancy_pct", "resident_warps_est", "warp_efficiency_pct",
            "registers_per_thread_peak", "shared_mem_per_block_peak_bytes", "block_size_peak",
            "sm_util_pct", "dram_util_pct", "bandwidth_util_pct",
        ],
    )

    crossover_rows = []
    ds = sorted({row["D"] for row in rows})
    ls = sorted({row["L"] for row in rows})
    time_lookup = {(row["kernel"], row["D"], row["L"]): row["time_ms"] for row in rows}

    for d in ds:
        ratio_series = []
        for l in ls:
            hillis = time_lookup.get(("hillis_steele", d, l))
            blelloch = time_lookup.get(("blelloch", d, l))
            if hillis is None or blelloch in (None, 0.0):
                continue
            ratio_series.append((l, hillis / blelloch))

        crossing = None
        for i in range(1, len(ratio_series)):
            prev_l, prev_r = ratio_series[i - 1]
            cur_l, cur_r = ratio_series[i]
            if (prev_r - 1.0) == 0:
                crossing = (prev_l, prev_l)
                break
            if (prev_r - 1.0) * (cur_r - 1.0) < 0:
                crossing = (prev_l, cur_l)
                break

        if crossing:
            note = f"crosses between L={crossing[0]} and L={crossing[1]}"
        elif ratio_series and all(r > 1.0 for _, r in ratio_series):
            note = "hillis_steele slower than blelloch for all tested L"
        elif ratio_series and all(r < 1.0 for _, r in ratio_series):
            note = "hillis_steele faster than blelloch for all tested L"
        else:
            note = "no crossover detected in tested range"

        crossover_rows.append({
            "D": d,
            "note": note,
            "ratio_series": "; ".join(f"L={l}: {r:.4f}" for l, r in ratio_series),
        })

    crossover_csv = os.path.join(args.out_dir, "crossover_summary.csv")
    write_csv(crossover_csv, crossover_rows, ["D", "note", "ratio_series"])

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit("=== SSM PREFIX SCAN METRICS REPORT ===")
    emit(f"timing_csv: {args.timing_csv}")
    emit(f"raw_dir:    {args.raw_dir}")
    emit(f"out_dir:    {args.out_dir}")
    emit(f"peak_bw_gbs={args.peak_bw_gbs:.3f} peak_fp32_gflops={args.peak_fp32_gflops:.3f}")
    emit("")

    emit("=== PERFORMANCE TABLES (time_ms) ===")
    for d in ds:
        emit(f"D={d}")
        emit("L,warp_shuffle,blelloch,hillis_steele")
        for l in ls:
            ws = time_lookup.get(("warp_shuffle", d, l))
            bl = time_lookup.get(("blelloch", d, l))
            hs = time_lookup.get(("hillis_steele", d, l))
            emit(f"{l},{fmt(ws, 4)},{fmt(bl, 4)},{fmt(hs, 4)}")
        emit("")

    emit("=== ROOFLINE POINTS (all runs) ===")
    emit("kernel,D,L,AI_flop_per_byte,GFLOP_s,DRAM_BW_GB_s,roofline_bound,roofline_eff_pct")
    for row in rows:
        emit(
            f"{row['kernel']},{row['D']},{row['L']},"
            f"{fmt(row['ai_flop_per_byte'], 6)},"
            f"{fmt(row['gflops'], 3)},"
            f"{fmt(row['dram_bw_gb_s'], 3)},"
            f"{row['roofline_bound'] or 'NA'},"
            f"{fmt(row['roofline_efficiency_pct'], 2)}"
        )
    emit("")

    emit("=== FULL PER-RUN UTILIZATION TABLE ===")
    emit("kernel,D,L,occ_pct,warp_eff_pct,sm_util_pct,dram_util_pct,bw_util_pct,resident_warps,regs_per_thread,block_size,shared_mem_B,l1_hit_pct,l2_hit_pct,utilization_bound")
    for row in rows:
        emit(
            f"{row['kernel']},{row['D']},{row['L']},"
            f"{fmt(row['achieved_occupancy_pct'], 2)},"
            f"{fmt(row['warp_efficiency_pct'], 2)},"
            f"{fmt(row['sm_util_pct'], 2)},"
            f"{fmt(row['dram_util_pct'], 2)},"
            f"{fmt(row['bandwidth_util_pct'], 2)},"
            f"{fmt(row['resident_warps_est'], 2)},"
            f"{fmt(row['registers_per_thread_peak'], 2)},"
            f"{fmt(row['block_size_peak'], 2)},"
            f"{fmt(row['shared_mem_per_block_peak_bytes'], 2)},"
            f"{fmt(row['l1_hit_rate_pct'], 2)},"
            f"{fmt(row['l2_hit_rate_pct'], 2)},"
            f"{row['utilization_bound'] or 'NA'}"
        )
    emit("")

    emit("=== OCCUPANCY/REGISTER SUMMARY (avg over L) ===")
    emit("kernel,D,occ_pct_avg,resident_warps_avg,warp_eff_avg,regs_peak_avg,block_size_avg,shared_mem_B_avg,sm_util_avg,dram_util_avg,bw_util_avg")
    for item in occ_summary:
        emit(
            f"{item['kernel']},{item['D']},"
            f"{fmt(item['achieved_occupancy_pct'], 2)},"
            f"{fmt(item['resident_warps_est'], 2)},"
            f"{fmt(item['warp_efficiency_pct'], 2)},"
            f"{fmt(item['registers_per_thread_peak'], 2)},"
            f"{fmt(item['block_size_peak'], 2)},"
            f"{fmt(item['shared_mem_per_block_peak_bytes'], 2)},"
            f"{fmt(item['sm_util_pct'], 2)},"
            f"{fmt(item['dram_util_pct'], 2)},"
            f"{fmt(item['bandwidth_util_pct'], 2)}"
        )
    emit("")

    emit("=== CROSSOVER ANALYSIS (hillis_steele / blelloch) ===")
    for row in crossover_rows:
        emit(f"D={row['D']}: {row['note']}")
        emit(f"  {row['ratio_series']}")
    emit("")

    emit("=== OUTPUT FILES ===")
    emit(f"merged_metrics_csv: {merged_csv}")
    emit(f"occupancy_summary_csv: {occ_summary_csv}")
    emit(f"crossover_summary_csv: {crossover_csv}")

    report_path = os.path.join(args.out_dir, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
