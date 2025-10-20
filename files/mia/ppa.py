#!/usr/bin/env python3
# ppa_roofline.py — adds --impl {naive,numpy,numba} and safe tiling for L1
import argparse, time, math, json, csv, os, sys
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

# ----------------------------
# Kernels (dot, l1) + backends
# ----------------------------

def dot_numpy(A, B):
    # A: (M,K), B: (N,K)  ->  C: (M,N) = A @ B^T
    return A @ B.T

def dot_numba(A, B):
    try:
        import numba as nb
    except Exception:
        return dot_numpy(A, B)
    # JIT a simple tiled dot (A @ B^T)
    @nb.njit(fastmath=True, parallel=False, cache=True)
    def _dot(A, B):
        M, K = A.shape
        N = B.shape[0]
        C = np.zeros((M, N), dtype=A.dtype)
        for i in range(M):
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += A[i, k] * B[j, k]
                C[i, j] = s
        return C
    return _dot(A, B)

def l1_naive(A, B):
    # A:(M,K), B:(N,K) -> C:(M,N) with sum_k |A[i,k]-B[j,k]|
    M, K = A.shape; N = B.shape[0]
    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += abs(A[i, k] - B[j, k])
            C[i, j] = s
    return C

def l1_numpy_tiled(A, B, Ti=64, Tj=64, Tk=128):
    # Tiled to keep the (Ti x Tj x Tk) working set bounded
    M, K = A.shape; N = B.shape[0]
    C = np.zeros((M, N), dtype=A.dtype)
    for i0 in range(0, M, Ti):
        i1 = min(i0 + Ti, M)
        Ai = A[i0:i1, :]  # (bi, K)
        for j0 in range(0, N, Tj):
            j1 = min(j0 + Tj, N)
            Bj = B[j0:j1, :]  # (bj, K)
            # accumulate over K in chunks
            block = np.zeros((i1 - i0, j1 - j0), dtype=A.dtype)
            for k0 in range(0, K, Tk):
                k1 = min(k0 + Tk, K)
                Aik = Ai[:, k0:k1]              # (bi, tk)
                Bjk = Bj[:, k0:k1]              # (bj, tk)
                # (bi,1,tk) - (1,bj,tk) -> (bi,bj,tk)
                diff = np.abs(Aik[:, None, :] - Bjk[None, :, :])
                block += diff.sum(axis=2)
            C[i0:i1, j0:j1] = block
    return C

def l1_numba(A, B, Ti=64, Tj=64, Tk=64):
    try:
        import numba as nb
    except Exception:
        return l1_numpy_tiled(A, B, Ti, Tj, Tk)
    # Simple numba triple-loop; good for moderate sizes
    @nb.njit(fastmath=True, parallel=False, cache=True)
    def _l1(A, B):
        M, K = A.shape
        N = B.shape[0]
        C = np.zeros((M, N), dtype=A.dtype)
        for i in range(M):
            for j in range(N):
                s = 0.0
                for k in range(K):
                    d = A[i, k] - B[j, k]
                    if d < 0: d = -d
                    s += d
                C[i, j] = s
        return C
    return _l1(A, B)

# ----------------------------
# Utilities
# ----------------------------

def dtype_from_str(s):
    s = s.lower()
    if s in ("fp16", "float16", "half"):   return np.float16
    if s in ("fp32", "float32", "single"): return np.float32
    raise ValueError(f"Unsupported dtype {s}")

def gops(ops, seconds):
    return (ops / seconds) / 1e9

def roofline(ai_ops_per_byte, peak_gops, bw_gbs):
    # compute roofline limit = min(peak, AI * BW)
    return min(peak_gops, ai_ops_per_byte * bw_gbs * 1e9 / 1e9)

def bytes_moved_est(kernel, M, N, K, dtype):
    # heuristic: read A (M*K), read B (N*K), write C (M*N)
    s = np.dtype(dtype).itemsize
    return s * (M*K + N*K + M*N)

def ops_count(kernel, M, N, K):
    if kernel == "dot":
        # multiply + add per inner step
        return 2 * M * N * K
    elif kernel == "l1":
        # sub + abs + add per inner step
        return 3 * M * N * K
    else:
        raise ValueError(kernel)

def run_kernel(kernel, impl, M, N, K, dtype, repeat, tiles):
    rng = np.random.default_rng(42)
    A = rng.standard_normal((M, K), dtype=dtype)
    B = rng.standard_normal((N, K), dtype=dtype)

    # warmup
    if kernel == "dot":
        if impl == "numba":
            dot_numba(A, B)
        elif impl == "numpy":
            dot_numpy(A, B)
        else:
            # naive dot is too slow; use numpy instead
            dot_numpy(A, B)
    else:
        if impl == "numba":
            l1_numba(A, B, tiles[0], tiles[1], tiles[2])
        elif impl == "numpy":
            l1_numpy_tiled(A, B, tiles[0], tiles[1], tiles[2])
        else:
            l1_naive(A, B)

    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        if kernel == "dot":
            C = dot_numba(A, B) if impl == "numba" else dot_numpy(A, B)
        else:
            if impl == "numba":
                C = l1_numba(A, B, tiles[0], tiles[1], tiles[2])
            elif impl == "numpy":
                C = l1_numpy_tiled(A, B, tiles[0], tiles[1], tiles[2])
            else:
                C = l1_naive(A, B)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best

def maybe_plot(out_base, ai, measured_gops, peak_gops, bw_gbs):
    if not HAVE_PLT: return
    # Simple roofline with one point
    xs = np.logspace(-3, 4, 200)  # AI range
    roof = np.minimum(peak_gops, xs * bw_gbs)
    plt.figure()
    plt.loglog(xs, roof, label="Roofline limit")
    plt.axhline(peak_gops, linestyle="--", label=f"Compute peak ({peak_gops:.1f} Gops/s)")
    plt.axline((1e-3, 1e-3*bw_gbs), slope=1, linestyle="--", label=f"BW line ({bw_gbs} GB/s)")
    plt.scatter([ai], [measured_gops], marker="o", color="black", label="Measured")
    plt.xlabel("Arithmetic Intensity (ops/byte)")
    plt.ylabel("Performance (Gops/s)")
    plt.title("Roofline")
    plt.legend()
    png = f"{out_base}_roofline.png"
    plt.savefig(png, dpi=140, bbox_inches="tight")

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=["dot","l1"], default="l1")
    ap.add_argument("--impl", choices=["naive","numpy","numba"], default="numpy",
                    help="Backend implementation")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--peak-gops", type=float, default=1.0)
    ap.add_argument("--bw-gbs", type=float, default=3.1)
    ap.add_argument("--out", default="ppa_result")
    ap.add_argument("--energy-json", default=None)
    ap.add_argument("--area-json", default=None)
    ap.add_argument("--Ti", type=int, default=64, help="L1 tile I")
    ap.add_argument("--Tj", type=int, default=64, help="L1 tile J")
    ap.add_argument("--Tk", type=int, default=128, help="L1 tile K")
    args = ap.parse_args()

    dtype = dtype_from_str(args.dtype)
    secs = run_kernel(args.kernel, args.impl, args.M, args.N, args.K, dtype, args.repeat, (args.Ti,args.Tj,args.Tk))
    total_ops = ops_count(args.kernel, args.M, args.N, args.K)
    perf = gops(total_ops, secs)
    bytes_mv = bytes_moved_est(args.kernel, args.M, args.N, args.K, dtype)
    ai = total_ops / bytes_mv
    roof_compute = args.peak_gops
    roof_bw = (ai * args.bw_gbs)
    roof_pred = min(roof_compute, roof_bw)

    # crude energy/area proxies if not provided
    energy_mJ = (total_ops/1e9)*0.5 + (bytes_mv/1e9)*50   # arbitrary model; tune if you have a model
    area_proxy = total_ops * (0.35 if args.kernel=="l1" else 1.0)

    # override if models provided
    if args.energy_json and Path(args.energy_json).exists():
        try:
            ej = json.loads(Path(args.energy_json).read_text())
            energy_mJ = ej.get(args.kernel, {}).get(args.dtype, energy_mJ)
        except Exception:
            pass
    if args.area_json and Path(args.area_json).exists():
        try:
            aj = json.loads(Path(args.area_json).read_text())
            area_proxy = aj.get(args.kernel, {}).get(args.dtype, area_proxy)
        except Exception:
            pass

    base = args.out
    js = {
        "kernel": args.kernel,
        "impl": args.impl,
        "dtype": args.dtype,
        "M": args.M, "N": args.N, "K": args.K,
        "repeat": args.repeat,
        "runtime_s": round(secs, 6),
        "achieved_gops": round(perf, 3),
        "arithmetic_intensity_ops_per_byte": round(ai, 3),
        "roofline_compute_cap_gops": round(roof_compute, 1),
        "roofline_bw_cap_gops": round(roof_bw, 1),
        "roofline_predicted_gops": round(roof_pred, 2),
        "bytes_moved_MB": round(bytes_mv/1e6, 2),
        "total_ops": int(total_ops),
        "est_energy_mJ": round(energy_mJ, 3),
        "est_area_proxy": area_proxy
    }
    Path(f"{base}_summary.json").write_text(json.dumps(js, indent=2))

    with open(f"{base}_summary.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(js.keys()); w.writerow(js.values())

    maybe_plot(base, ai, perf, args.peak_gops, args.bw_gbs)

    print("\n=== PPA + Roofline Summary ===")
    print(f"Kernel: {args.kernel} ({args.dtype})  Size: M={args.M},N={args.N},K={args.K}")
    print(f"Runtime: {secs:.4f} s   Achieved: {perf:.2f} Gops/s")
    print(f"Arithmetic Intensity: {ai:.0f} ops/byte")
    print(f"Roofline (compute cap): {args.peak_gops:.1f} Gops/s")
    print(f"Roofline (BW cap @ {args.bw_gbs:.1f} GB/s): {roof_bw:.1f} Gops/s")
    print(f"Roofline predicted: {roof_pred:.2f} Gops/s")
    print(f"Bytes moved: {bytes_mv/1e6:.2f}MB   Total ops: {total_ops/1e6:.1f}M ops")
    print(f"Est. energy: {energy_mJ:.3f} mJ   (model, not measured)")
    print(f"Est. area-units (proxy): {area_proxy:.2e} (comparative only)")
    print(f"Artifacts: {base}_summary.json / .csv / _roofline.png")
    if args.impl == "naive" and args.kernel == "l1":
        print("Note: naive L1 is very slow; prefer --impl numpy or --impl numba.")

if __name__ == "__main__":
    main()

