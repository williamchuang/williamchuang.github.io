import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def loadj(path):
    with open(path, "r") as f:
        return json.load(f)

def first_key(d, names):
    for n in names:
        if n in d: return d[n]
    r = d.get("roofline")
    if isinstance(r, dict):
        for n in names:
            if n in r: return r[n]
    return None

def extract_ai_ach(d):
    # Try common names first
    ai = first_key(d, [
        "arithmetic_intensity_ops_per_byte",  # correct for your JSON
        "arithmetic_intensity",
        "ai",
        "AI"
    ])
    ach = first_key(d, [
        "achieved_gops",
        "achieved_ops_per_s_giga",
        "throughput_gops",
        "gops"
    ])
    # Fallback: compute AI from totals if missing
    if ai is None:
        tot_ops = first_key(d, ["total_ops", "ops_total"])
        bytes_mb = first_key(d, ["bytes_moved_MB", "bytes_MB"])
        bytes_b  = first_key(d, ["bytes_moved_B", "bytes_B"])
        if tot_ops is not None and (bytes_mb is not None or bytes_b is not None):
            bytes_float = (bytes_b if bytes_b is not None else float(bytes_mb)*1e6)
            ai = float(tot_ops) / float(bytes_float)
    if ai is None or ach is None:
        print("DEBUG top-level keys:", sorted(d.keys()))
        if isinstance(d.get("roofline"), dict):
            print("DEBUG roofline keys:", sorted(d["roofline"].keys()))
        raise KeyError("Could not find arithmetic intensity and/or achieved Gops in summary JSON.")
    return float(ai), float(ach)

def main():
    ap = argparse.ArgumentParser(description="Overlay calibrated roofline with measured points.")
    ap.add_argument("--dot", required=True, help="DOT kernel *_summary.json (calibration)")
    ap.add_argument("--other", required=True, help="Second point (e.g., L1) *_summary.json")
    ap.add_argument("--peak-gops", type=float, default=None, help="Override compute peak (Gops/s)")
    ap.add_argument("--bw-gbs", type=float, default=None, help="Override memory bandwidth (GB/s)")
    ap.add_argument("--label-other", default="Other kernel")
    ap.add_argument("--out", default="roofline_overlay.png")
    args = ap.parse_args()

    jd = loadj(args.dot)
    jo = loadj(args.other)

    # Peak/bandwidth: prefer CLI; else pull from JSON; else sensible defaults
    PEAK = args.peak_gops if args.peak_gops is not None else first_key(jd, [
        "compute_cap_gops", "peak_gops", "roofline_compute_cap_gops"
    ])
    if PEAK is None:
        # last resort: use measured DOT achieved as compute ceiling
        PEAK = first_key(jd, ["achieved_gops"]) or 1.0

    BW = args.bw_gbs if args.bw_gbs is not None else first_key(jd, [
        "bw_gbs", "bandwidth_gbs", "roofline_bw_gbs", "mem_bw_gbs"
    ])
    if BW is None:
        BW = 35.0  # reasonable default for your Mac

    ai_dot,   ach_dot   = extract_ai_ach(jd)
    ai_other, ach_other = extract_ai_ach(jo)

    # Build roofline curve
    ais  = [10**(x/10) for x in range(-10, 41)]  # ~1e-1 .. 1e4
    roof = [min(float(PEAK), float(BW)*ai) for ai in ais]

    plt.figure(figsize=(7.2,5.2))
    plt.loglog(ais, roof, label=f"Roofline (peak={float(PEAK):.1f} Gops/s, BW={float(BW):.0f} GB/s)")
    plt.scatter([ai_dot], [ach_dot], s=60, marker="o",
                label=f"DOT fp32 (AI={ai_dot:.0f}, {ach_dot:.2f} Gops/s)")
    plt.scatter([ai_other], [ach_other], s=60, marker="^",
                label=f"{args.label_other} (AI={ai_other:.0f}, {ach_other:.2f} Gops/s)")

    plt.xlabel("Arithmetic Intensity (ops/byte)")
    plt.ylabel("Achieved Throughput (Gops/s)")
    plt.title("Calibrated Roofline with Measured Points")
    plt.grid(True, which='both', ls='--', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")
    print(f"[debug] Using PEAK={float(PEAK):.2f} Gops/s, BW={float(BW):.2f} GB/s")
    print(f"[debug] DOT:   AI={ai_dot:.2f}, Ach={ach_dot:.2f} Gops/s")
    print(f"[debug] OTHER: AI={ai_other:.2f}, Ach={ach_other:.2f} Gops/s")

if __name__ == "__main__":
    main()

