#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twin invariance demo (MIA) — pure Python, no deps.

- State space: binary codes of fixed bit-width (default 64 bits).
- Invariant primitive: Hamming distance J(q,k) = popcount(q XOR k).
- Program: integer-only classifier that aggregates a monotone function of J.
- Group action g: bit permutation + XOR mask, applied diagonally to
  ALL manifold-resident values (prototypes, queries) -> a "twin" program.

We prove by execution that:
  traces (all distances), class scores, and predictions are IDENTICAL
  before and after the diagonal action. We also print SHA-256 checksums
  of the full distance traces to certify bit-for-bit equality.

Usage:
  python3 mia_twin_demo.py
  python3 mia_twin_demo.py --bits 64 --classes 4 --per-class 32 --queries 300 --seed 123 --twin-seed 42
"""

import argparse, hashlib, random, sys

# --- utils -------------------------------------------------------------------

def popcount(x: int) -> int:
    """Portable popcount (uses int.bit_count() if available)."""
    try:
        return x.bit_count()  # Py3.8+
    except AttributeError:
        return bin(x).count("1")

def rand_code(bits: int, rng: random.Random) -> int:
    """Random integer in [0, 2^bits)."""
    if bits <= 0:
        return 0
    # Use getrandbits for speed and uniformity
    return rng.getrandbits(bits)

def make_permutation(bits: int, rng: random.Random):
    """Return a random permutation pi over {0..bits-1} as a list."""
    pi = list(range(bits))
    rng.shuffle(pi)
    return pi

def apply_perm_mask(x: int, bits: int, perm, mask: int) -> int:
    """
    Apply diagonal group action element:
      - perm: permutation of bit positions (list of length 'bits')
      - mask: XOR mask of same width
    Returns y = P(x) XOR mask, where P permutes bit positions.
    """
    # Permute bits
    y = 0
    for old in range(bits):
        if (x >> old) & 1:
            new = perm[old]
            y |= (1 << new)
    # XOR mask
    return y ^ mask

# --- MIA "program" (integer-only classifier) ---------------------------------

def class_score_kernel(bits: int, ham_dist: int) -> int:
    """
    Monotone integer kernel over Hamming distance.
    Larger similarity -> bigger score contribution.
    Here: score = (bits - ham_dist).  (Any monotone f works.)
    """
    return bits - ham_dist

def build_model(n_classes: int, per_class: int, bits: int, seed: int):
    """
    Construct a toy prototype bank:
      - prototypes: list[int] (codes)
      - labels    : list[int] (class id 0..n_classes-1)
    """
    rng = random.Random(seed)
    prototypes, labels = [], []
    for c in range(n_classes):
        for _ in range(per_class):
            prototypes.append(rand_code(bits, rng))
            labels.append(c)
    return prototypes, labels

def mia_predict_and_trace(query_codes, prototypes, labels, bits: int):
    """
    For each query q:
      - compute distances d_i = popcount(q XOR p_i) for all prototypes
      - compute class scores as sum over kernel(d_i) per class
      - take argmax class
    Returns:
      preds: list[int]
      scores: list[list[int]]  (per-query class scores)
      trace_bytes: bytes (concatenated distances for ALL queries, for hashing)
    """
    n_classes = max(labels) + 1 if labels else 0
    preds, scores = [], []
    trace = bytearray()
    # distances are <= bits; pack each distance in 1 byte if bits<=255
    if bits > 255:
        raise ValueError("bits > 255 not supported in this simple trace packer")

    for q in query_codes:
        # distances to every prototype (integer list)
        dists = [popcount(q ^ p) for p in prototypes]

        # append to trace (exact, order-preserving)
        for d in dists:
            trace.append(d & 0xFF)

        # class scores
        cls = [0] * n_classes
        for d, lab in zip(dists, labels):
            cls[lab] += class_score_kernel(bits, d)

        scores.append(cls)
        # stable argmax (lowest class id on ties)
        best_c, best_v = 0, cls[0] if cls else 0
        for c in range(1, n_classes):
            v = cls[c]
            if v > best_v:
                best_c, best_v = c, v
        preds.append(best_c)
    return preds, scores, bytes(trace)

# --- twin (diagonal) transport ------------------------------------------------

def twin_transport_bank(prototypes, queries, bits: int, twin_seed: int):
    """
    Sample a random permutation and mask; apply to ALL prototypes + queries.
    Return (perm, mask, prototypes', queries').
    """
    rng = random.Random(twin_seed)
    perm = make_permutation(bits, rng)
    mask = rand_code(bits, rng)
    prot2 = [apply_perm_mask(p, bits, perm, mask) for p in prototypes]
    quer2 = [apply_perm_mask(q, bits, perm, mask) for q in queries]
    return perm, mask, prot2, quer2

# --- top-level demo -----------------------------------------------------------

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def run_demo(args):
    # 1) Build model and queries
    prototypes, labels = build_model(args.classes, args.per_class, args.bits, args.seed)
    rng = random.Random(args.seed + 1)
    queries = [rand_code(args.bits, rng) for _ in range(args.queries)]

    # 2) Baseline run
    preds0, scores0, trace0 = mia_predict_and_trace(queries, prototypes, labels, args.bits)

    # 3) Twin (diagonal) transport
    perm, mask, prot2, quer2 = twin_transport_bank(prototypes, queries, args.bits, args.twin_seed)

    # Sanity: show that the internal codes genuinely changed
    print(f"Example prototype before: 0x{prototypes[0]:0{args.bits//4}X}")
    print(f"Example prototype after : 0x{prot2[0]:0{args.bits//4}X}")
    if prototypes[0] == prot2[0]:
        print("WARN: prototype unchanged (unexpected for random twin)")

    # 4) Twin run
    preds1, scores1, trace1 = mia_predict_and_trace(quer2, prot2, labels, args.bits)

    # 5) Compare EVERYTHING (bit-for-bit)
    ok_preds = preds0 == preds1
    ok_scores = scores0 == scores1
    ok_trace = trace0 == trace1

    # 6) Pretty printing + checksums
    h0, h1 = sha256_hex(trace0), sha256_hex(trace1)
    print("\n--- Twin invariance checks ---")
    print(f"Predictions equal : {ok_preds}")
    print(f"Scores equal      : {ok_scores}")
    print(f"Trace equal       : {ok_trace}")
    print(f"Trace SHA-256 (original): {h0}")
    print(f"Trace SHA-256 (twin)    : {h1}")

    if ok_preds and ok_scores and ok_trace:
        print("\n✅ Twin invariance OK — outputs and traces are identical after diagonal transport.")
        return 0
    else:
        print("\n❌ Twin invariance FAILED — investigate invariants or transport.")
        return 1

def parse_args(argv):
    p = argparse.ArgumentParser(description="MIA twin invariance demo (Hamming invariant)")
    p.add_argument("--bits", type=int, default=64, help="bit width of codes (<=255 for this demo)")
    p.add_argument("--classes", type=int, default=3, help="number of classes")
    p.add_argument("--per-class", type=int, default=32, help="prototypes per class")
    p.add_argument("--queries", type=int, default=200, help="number of queries to test")
    p.add_argument("--seed", type=int, default=123, help="dataset RNG seed")
    p.add_argument("--twin-seed", type=int, default=42, help="twin (perm+mask) RNG seed")
    return p.parse_args(argv)

if __name__ == "__main__":
    sys.exit(run_demo(parse_args(sys.argv[1:])))

