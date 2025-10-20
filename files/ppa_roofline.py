#!/usr/bin/env python3
import argparse
import time
import math
import json
import csv
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

@dataclass
class EnergyModel:
    e_add: float = 0.4
    e_abs: float = 0.4
    e_mul: float = 2.0
    e_fma: float = 2.2
    e_xor: float = 0.1
    e_popc: float = 0.3
    e_mem_byte: float = 20.0

@dataclass
class AreaModel:
    a_add: float = 1.0
    a_abs: float = 1.0
    a_mul: float = 6.0
    a_fma: float = 7.0
    a_xor: float = 0.3
    a_popc: float = 1.5

@dataclass
class KernelReport:
    kernel: str
    dtype: str
    M: int
    N: int
    K: int
    repeat: int
    bytes_moved: float
    op_counts: Dict[str, float]
    total_ops: float
    runtime_s: float
    achieved_gops: float
    arithmetic_intensity: float
    roofline_compute_cap_gops: float
    roofline_bw_cap_gops: float
    roofline_pred_gops: float
    est_energy_pJ: float
    est_energy_mJ: float
    est_area_units: float

def human(x, unit=""):
    if x < 1e3:
        return f"{x:.3g}{unit}"
    for s, u in [(1e3,"K"),(1e6,"M"),(1e9,"G"),(1e12,"T")]:
        if x < s*1e3:
            return f"{x/s:.3g}{u}{unit}"
    return f"{x/1e12:.3g}T{unit}"

def dtype_bytes(dtype: str) -> int:
    d = dtype.lower()
    if d in ("fp16","float16","half"):
        return 2
    if d in ("fp32","float32","single"):
        return 4
    if d in ("int8","i8","uint8","u8"):
        return 1
    if d in ("int16","i16","uint16","u16"):
        return 2
    raise ValueError(f"Unsupported dtype: {dtype}")

def measure_mem_bandwidth_gbps(size_bytes: int = 256*1024*1024, iters: int = 3) -> float:
    if np is None:
        return 4.0
    n = size_bytes // 8
    a = np.random.rand(n).astype(np.float64)
    b = np.random.rand(n).astype(np.float64)
    c = a + b
    best = float('inf')
    for _ in range(iters):
        t0 = time.perf_counter()
        c = a + b
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    bytes_moved = 24.0 * n
    if best <= 0:
        return 0.0
    return (bytes_moved / best) / 1e9

def run_kernel_dot(M:int,N:int,K:int,dtype:str, repeat:int):
    if np is None:
        import random
        A = [[random.random() for _ in range(K)] for _ in range(M)]
        B = [[random.random() for _ in range(N)] for _ in range(K)]
        C = [[0.0 for _ in range(N)] for _ in range(M)]
        t0 = time.perf_counter()
        for _ in range(repeat):
            for i in range(M):
                for j in range(N):
                    s = 0.0
                    for k in range(K):
                        s += A[i][k]*B[k][j]
                    C[i][j] = s
        t1 = time.perf_counter()
        muls = M*N*K
        adds = M*N*(K-1)
        ops = muls + adds
        bytes_move = (M*K + K*N + M*N) * dtype_bytes(dtype)
        return (t1 - t0), {"mul": muls, "add": adds}, bytes_move
    dt = np.float16 if dtype.lower() in ("fp16","float16","half") else (
         np.float32 if dtype.lower() in ("fp32","float32","single") else np.float32)
    A = np.random.randn(M,K).astype(dt)
    B = np.random.randn(K,N).astype(dt)
    C = A @ B
    tbest = float('inf')
    for _ in range(repeat):
        t0 = time.perf_counter()
        C = A @ B
        t1 = time.perf_counter()
        tbest = min(tbest, t1 - t0)
    muls = M*N*K
    adds = M*N*(K-1)
    bytes_move = (M*K + K*N + M*N) * dtype_bytes(dtype)
    return tbest, {"mul": float(muls), "add": float(adds)}, float(bytes_move)

def run_kernel_l1(M:int,N:int,K:int,dtype:str, repeat:int):
    if np is None:
        import random
        A = [[random.random() for _ in range(K)] for _ in range(M)]
        B = [[random.random() for _ in range(K)] for _ in range(N)]
        D = [[0.0 for _ in range(N)] for _ in range(M)]
        t0 = time.perf_counter()
        for _ in range(repeat):
            for i in range(M):
                for j in range(N):
                    s = 0.0
                    for k in range(K):
                        s += abs(A[i][k] - B[j][k])
                    D[i][j] = s
        t1 = time.perf_counter()
        subs = M*N*K
        abss = M*N*K
        adds = M*N*(K-1)
        bytes_move = (M*K + N*K + M*N) * dtype_bytes(dtype)
        return (t1 - t0), {"sub":subs, "abs":abss, "add":adds}, bytes_move
    dt = np.float16 if dtype.lower() in ("fp16","float16","half") else (
         np.float32 if dtype.lower() in ("fp32","float32","single") else np.float32)
    A = np.random.randn(M,K).astype(dt)
    B = np.random.randn(N,K).astype(dt)
    tbest = float('inf')
    def compute_once(A,B):
        T = max(1, 32768 // max(1, K))
        Mdim, Kdim = A.shape
        Ndim, _ = B.shape
        out = np.empty((Mdim, Ndim), dtype=dt)
        start = 0
        while start < Ndim:
            end = min(Ndim, start + T)
            diff = A[:,None,:] - B[None,start:end,:]
            val = np.abs(diff).sum(axis=2)
            out[:,start:end] = val
            start = end
        return out
    _ = compute_once(A,B)
    for _ in range(repeat):
        t0 = time.perf_counter()
        D = compute_once(A,B)
        t1 = time.perf_counter()
        tbest = min(tbest, t1 - t0)
    subs = float(M)*float(N)*float(K)
    abss = float(M)*float(N)*float(K)
    adds = float(M)*float(N)*(float(K)-1.0)
    bytes_move = (M*K + N*K + M*N) * dtype_bytes(dtype)
    return tbest, {"sub": subs, "abs": abss, "add": adds}, float(bytes_move)

def estimate_energy_pJ(op_counts: Dict[str,float], bytes_moved: float, em: EnergyModel) -> float:
    e = 0.0
    e += em.e_add * op_counts.get("add", 0.0)
    e += em.e_abs * op_counts.get("abs", 0.0)
    e += em.e_mul * op_counts.get("mul", 0.0)
    e += em.e_fma * op_counts.get("fma", 0.0)
    e += em.e_xor * op_counts.get("xor", 0.0)
    e += em.e_popc * op_counts.get("popc", 0.0)
    e += em.e_mem_byte * bytes_moved
    return e

def estimate_area_units(op_counts: Dict[str,float], am: AreaModel) -> float:
    a = 0.0
    a += am.a_add * op_counts.get("add", 0.0)
    a += am.a_abs * op_counts.get("abs", 0.0)
    a += am.a_mul * op_counts.get("mul", 0.0)
    a += am.a_fma * op_counts.get("fma", 0.0)
    a += am.a_xor * op_counts.get("xor", 0.0)
    a += am.a_popc * op_counts.get("popc", 0.0)
    return a

def roofline_prediction(arith_intensity: float, peak_gops: float, bw_gbs: float):
    bw_cap = arith_intensity * bw_gbs
    pred = min(peak_gops, bw_cap)
    return peak_gops, bw_cap, pred

def run_and_report(kernel: str, M:int,N:int,K:int, dtype:str, repeat:int,
                   peak_gops: Optional[float], bw_gbs: Optional[float],
                   energy_model: EnergyModel, area_model: AreaModel,
                   out_prefix: str) -> KernelReport:
    if kernel == "dot":
        rt, ops, bytes_mv = run_kernel_dot(M,N,K,dtype, repeat)
    elif kernel == "l1":
        rt, ops, bytes_mv = run_kernel_l1(M,N,K,dtype, repeat)
    else:
        raise ValueError("kernel must be one of: dot, l1")

    total_ops = sum(ops.values())
    achieved = (total_ops / rt) / 1e9 if rt > 0 else 0.0
    ai = total_ops / bytes_mv if bytes_mv > 0 else 0.0

    if bw_gbs is None:
        bw_gbs = measure_mem_bandwidth_gbps()

    if peak_gops is None:
        peak_gops = max(achieved*1.5, 1.0)

    comp_cap, bw_cap, pred = roofline_prediction(ai, peak_gops, bw_gbs)

    e_pJ = estimate_energy_pJ(ops, bytes_mv, energy_model)
    area_units = estimate_area_units(ops, area_model)

    rep = KernelReport(
        kernel=kernel, dtype=dtype, M=M, N=N, K=K, repeat=repeat,
        bytes_moved=bytes_mv, op_counts=ops, total_ops=total_ops,
        runtime_s=rt, achieved_gops=achieved, arithmetic_intensity=ai,
        roofline_compute_cap_gops=comp_cap, roofline_bw_cap_gops=bw_cap,
        roofline_pred_gops=pred, est_energy_pJ=e_pJ, est_energy_mJ=e_pJ*1e-6,
        est_area_units=area_units
    )

    with open(f"{out_prefix}_summary.json","w") as f:
        json.dump(asdict(rep), f, indent=2)
    with open(f"{out_prefix}_summary.csv","w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "kernel","dtype","M","N","K","repeat","bytes_moved","total_ops",
            "runtime_s","achieved_gops","arithmetic_intensity",
            "roofline_compute_cap_gops","roofline_bw_cap_gops","roofline_pred_gops",
            "est_energy_pJ","est_energy_mJ","est_area_units"
        ])
        w.writerow([
            rep.kernel, rep.dtype, rep.M, rep.N, rep.K, rep.repeat, rep.bytes_moved,
            rep.total_ops, rep.runtime_s, rep.achieved_gops, rep.arithmetic_intensity,
            rep.roofline_compute_cap_gops, rep.roofline_bw_cap_gops, rep.roofline_pred_gops,
            rep.est_energy_pJ, rep.est_energy_mJ, rep.est_area_units
        ])

    if MATPLOTLIB_OK:
        fig = plt.figure(figsize=(7,5))
        # compute ceiling line
        plt.axhline(y=comp_cap, linestyle="--")
        # bandwidth line
        if np is not None:
            xs = np.logspace(-3, 3, 200)
        else:
            xs = [ai*0.1 if ai>0 else 0.01, ai if ai>0 else 0.1, ai*10 if ai>0 else 1.0]
        ys = [bw_gbs * x for x in xs]
        plt.loglog(xs, ys)
        plt.loglog([ai],[achieved], marker="o")
        plt.xlabel("Arithmetic intensity (ops/byte)")
        plt.ylabel("Throughput (Gops/s)")
        plt.title(f"Roofline — {kernel.upper()} ({dtype}), M={M},N={N},K={K}")
        t = (
            f"Measured BW: {bw_gbs:.1f} GB/s\n"
            f"Compute cap: {comp_cap:.1f} Gops/s\n"
            f"AI: {ai:.3g} ops/byte\n"
            f"Achieved: {achieved:.2f} Gops/s\n"
            f"Predicted: {pred:.2f} Gops/s"
        )
        plt.text(0.02, 0.02, t, transform=plt.gca().transAxes, fontsize=9)
        plt.grid(True, which="both", linewidth=0.4, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_roofline.png", dpi=160)
        plt.close(fig)

    return rep

def main():
    ap = argparse.ArgumentParser(description="PPA + Roofline analyzer for baseline vs MIA-style kernels.")
    ap.add_argument("--kernel", choices=["dot","l1"], default="l1")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--peak-gops", type=float, default=None)
    ap.add_argument("--bw-gbs", type=float, default=None)
    ap.add_argument("--out", default="ppa_result")
    ap.add_argument("--energy-json", default=None)
    ap.add_argument("--area-json", default=None)
    args = ap.parse_args()

    em = EnergyModel()
    am = AreaModel()
    if args.energy_json:
        with open(args.energy_json) as f:
            em = EnergyModel(**json.load(f))
    if args.area_json:
        with open(args.area_json) as f:
            am = AreaModel(**json.load(f))

    rep = run_and_report(args.kernel, args.M, args.N, args.K, args.dtype, args.repeat,
                         args.peak_gops, args.bw_gbs, em, am, args.out)

    print("\n=== PPA + Roofline Summary ===")
    print(f"Kernel: {rep.kernel} ({rep.dtype})  Size: M={rep.M},N={rep.N},K={rep.K}")
    print(f"Runtime: {rep.runtime_s:.4f} s   Achieved: {rep.achieved_gops:.2f} Gops/s")
    print(f"Arithmetic Intensity: {rep.arithmetic_intensity:.3g} ops/byte")
    cap_bw = rep.roofline_bw_cap_gops/rep.arithmetic_intensity if rep.arithmetic_intensity>0 else 0.0
    print(f"Roofline (compute cap): {rep.roofline_compute_cap_gops:.1f} Gops/s")
    print(f"Roofline (BW cap @ {cap_bw:.1f} GB/s): {rep.roofline_bw_cap_gops:.1f} Gops/s")
    print(f"Roofline predicted: {rep.roofline_pred_gops:.2f} Gops/s")
    print(f"Bytes moved: {human(rep.bytes_moved,'B')}   Total ops: {human(rep.total_ops,' ops')}")
    print(f"Est. energy: {rep.est_energy_mJ:.3f} mJ   (model, not measured)")
    print(f"Est. area-units (proxy): {rep.est_area_units:.3g} (comparative only)")
    print(f"Artifacts: {args.out}_summary.json / .csv / _roofline.png")
    if np is None:
        print("\nNote: NumPy not found — ran slow Python loop. Install NumPy for realistic timings.")

if __name__ == "__main__":
    main()
