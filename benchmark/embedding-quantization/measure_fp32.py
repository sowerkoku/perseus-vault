#!/usr/bin/env python3
"""FP32 vs INT8 embedding quality benchmark (#630 row 2).

Directly measures how much INT8 quantization affects embedding quality:
  - Cosine similarity between INT8 and FP32 embeddings of the same text
  - Nearest-neighbor agreement (do queries find the same top-k results?)
  - Rank correlation (how much does the similarity ordering change?)

This is faster and more direct than paraphrase recall, which requires
realistic bodies that the quantized model can semantically match.

Usage:
    python benchmark/embedding-quantization/measure_fp32.py -n 5000 -q 200
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# ── text generation ─────────────────────────────────────────────────────────
# Generate varied synthetic text sentences. Each sentence is 8-20 words
# drawn from a large vocabulary, producing distinct embedding vectors.

NOUNS = """
cat dog bird fish tree flower cloud river mountain ocean star moon sun wind
rain snow storm fire ice stone rock sand dust grass leaf apple orange banana
grape melon peach plum lemon lime mango berry piano guitar violin drum flute
harp trumpet horn bell whistle chime book page letter word story poem song
verse rhyme riddle door window wall roof floor ceiling stair hall room house
cabin tower bridge road path trail track lane street highway avenue route
bread cheese milk butter honey jam salt sugar spice herb sauce soup stew
coffee tea water wine beer juice lemonade cocoa shirt pants dress coat jacket
hat shoe boot sock glove scarf clock watch phone lamp desk chair table bed
shelf cabinet mirror horse cow sheep goat pig hen duck goose swan eagle hawk
owl crow raven rose lily daisy tulip iris orchid lotus fern moss ivy vine
reed bamboo silver gold copper iron steel bronze brass lead tin zinc nickel
angel devil ghost spirit fairy elf dwarf giant dragon phoenix ocean desert
forest jungle mountain valley canyon glacier cavern reef island anchor sail
mast deck cabin helm compass rudder keel bow stern autumn winter spring
summer dawn dusk noon midnight twilight sunrise sunset
""".split()

ADJECTIVES = """
bright dark warm cold hot cool fresh stale sweet sour bitter salty spicy
mild sharp soft hard smooth rough wet dry clean dirty clear cloudy foggy
misty stormy calm windy sunny rainy snowy icy frosty steamy humid arid
fast slow quick rapid swift steady gradual sudden gentle fierce kind cruel
brave bold timid shy proud humble wise clever foolish witty dull tall
short wide narrow deep shallow thick thin heavy light dense sparse loud
quiet silent noisy harsh gentle melodic rhythmic chaotic ordered neat
old new ancient modern young fresh ripe raw cooked burnt frozen melted
red blue green yellow purple orange pink brown black white gray silver
gold happy sad angry calm excited bored tired sleepy awake alert dreamy
focused strong weak mighty feeble sturdy fragile tough tender resilient
brittle elegant clumsy graceful awkward smooth jagged polished rough
refined coarse
""".split()

VERBS = """
run walk jump swim fly climb dance sing write read draw paint build create
destroy fix break mend weave spin knit sew cook bake roast fry boil steam
grow bloom wither fade shine glow sparkle flicker flash blaze burn smolder
drift float sink rise fall tumble spin whirl twist turn bend stretch fold
speak whisper shout call answer ask tell explain describe remember forget
listen hear watch see observe notice ignore seek find lose hide reveal
push pull lift carry drag drop throw catch bounce roll slide glide slip
greet welcome farewell depart arrive stay leave return wander roam explore
""".split()


def make_sentence(idx):
    """Generate a varied synthetic sentence."""
    rng = random.Random(idx * 0x9E3779B9 + 0x517CC1B7)
    n_words = rng.randint(8, 20)
    words = []
    for _ in range(n_words):
        pool = rng.choice([NOUNS, ADJECTIVES, VERBS])
        words.append(rng.choice(pool))
    # Make it sentence-like: capitalize first, add period
    words[0] = words[0].capitalize()
    return " ".join(words) + f" [{idx}]."


# ── embedding ───────────────────────────────────────────────────────────────

def find_int8_model():
    for build_dir in sorted(Path(REPO, "target", "release", "build").glob("perseus-vault-*")):
        out = build_dir / "out"
        model = out / "model_quantized.onnx"
        tok = out / "tokenizer.json"
        if model.exists() and tok.exists():
            return str(model), str(tok)
    raise FileNotFoundError("INT8 model not found. Build: cargo build --release")


def embed_texts(tokenizer, session, texts, batch_size=64):
    """Batch-embed texts."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encodings = [tokenizer.encode(t) for t in batch]
        max_len = min(max(len(e.ids) for e in encodings), 128)
        input_ids = np.zeros((len(batch), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)
        token_type_ids = np.zeros((len(batch), max_len), dtype=np.int64)

        for j, enc in enumerate(encodings):
            ids = enc.ids[:max_len]
            am = enc.attention_mask[:max_len]
            input_ids[j, :len(ids)] = ids
            attention_mask[j, :len(am)] = am

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        outputs = session.run(None, ort_inputs)
        hidden = outputs[0]

        mask = attention_mask.astype(np.float32)
        active = mask.sum(axis=1, keepdims=True)
        pooled = (hidden * mask[:, :, np.newaxis]).sum(axis=1) / np.maximum(active, 1)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.maximum(norms, 1e-12)
        embeddings.append(pooled)

    return np.concatenate(embeddings, axis=0).astype(np.float32)


# ── metrics ─────────────────────────────────────────────────────────────────

def rank_agreement(sims_a, sims_b, k):
    """Fraction of queries where top-k by A matches top-k by B (order-independent)."""
    top_a = set(np.argsort(-sims_a)[:k])
    top_b = set(np.argsort(-sims_b)[:k])
    return len(top_a & top_b) / k


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5000, help="Number of entities")
    ap.add_argument("-q", type=int, default=200, help="Number of query probes")
    ap.add_argument("--seed", type=int, default=630)
    ap.add_argument("--out", default=str(HERE / "fp32_vs_int8_results.json"))
    args = ap.parse_args()

    random.seed(args.seed)
    n = args.n
    nq = min(args.q, n)

    int8_model, tok_path = find_int8_model()
    fp32_model = "/tmp/all-MiniLM-L6-v2-fp32/onnx/model.onnx"
    tok = Tokenizer.from_file(tok_path)
    sess_int8 = ort.InferenceSession(int8_model, providers=["CPUExecutionProvider"])
    sess_fp32 = ort.InferenceSession(fp32_model, providers=["CPUExecutionProvider"])

    print(f"INT8: {int8_model}")
    print(f"FP32: {fp32_model}")
    print(f"Corpus: {n} entities, {nq} probes")

    # Generate diverse sentences
    texts = [make_sentence(i) for i in range(n)]
    probe_texts = [make_sentence(n + i) for i in range(nq)]  # distinct from corpus

    # Embed with both models
    print(f"\nEmbedding {n} corpus with INT8...", flush=True)
    t0 = time.monotonic()
    emb_int8 = embed_texts(tok, sess_int8, texts)
    t_int8 = time.monotonic() - t0
    print(f"  {t_int8:.1f}s ({n/t_int8:.0f} items/s)", flush=True)

    print(f"Embedding {n} corpus with FP32...", flush=True)
    t0 = time.monotonic()
    emb_fp32 = embed_texts(tok, sess_fp32, texts)
    t_fp32 = time.monotonic() - t0
    print(f"  {t_fp32:.1f}s ({n/t_fp32:.0f} items/s)", flush=True)

    print(f"Embedding {nq} probes with INT8...", flush=True)
    probe_int8 = embed_texts(tok, sess_int8, probe_texts)
    print(f"Embedding {nq} probes with FP32...", flush=True)
    probe_fp32 = embed_texts(tok, sess_fp32, probe_texts)

    # ── Metric 1: Self-consistency (same text, different model) ──
    cos_self = np.sum(probe_int8 * probe_fp32, axis=1)
    print(f"\n── Self-consistency (same text, INT8 vs FP32) ──")
    print(f"  Mean cosine: {cos_self.mean():.4f}")
    print(f"  Min cosine:  {cos_self.min():.4f}")
    print(f"  Median:      {np.median(cos_self):.4f}")

    # ── Metric 2: Nearest-neighbor agreement ──
    print(f"\n── Nearest-neighbor agreement (top-k overlap) ──")
    ks = [1, 5, 10, 20, 50, 100]
    for k in ks:
        if k > n:
            break
        overlaps = []
        for qi in range(nq):
            sims_int8 = np.dot(emb_int8, probe_int8[qi])
            sims_fp32 = np.dot(emb_fp32, probe_fp32[qi])
            overlaps.append(rank_agreement(sims_int8, sims_fp32, k))
        mean_ol = np.mean(overlaps)
        print(f"  top-{k:<4}: {mean_ol*100:.1f}% agreement")

    # ── Metric 3: Pairwise cosine similarity distribution ──
    print(f"\n── Pairwise cosine between INT8 and FP32 for all corpus items ──")
    # Sample 1000 random pairs for speed
    n_sample = min(1000, n)
    indices = random.sample(range(n), n_sample)
    cos_pairs = np.sum(emb_int8[indices] * emb_fp32[indices], axis=1)
    print(f"  n={n_sample}, mean={cos_pairs.mean():.4f}, std={cos_pairs.std():.4f}")
    print(f"  p1={np.percentile(cos_pairs, 1):.4f}, p5={np.percentile(cos_pairs, 5):.4f}")
    print(f"  p95={np.percentile(cos_pairs, 95):.4f}, p99={np.percentile(cos_pairs, 99):.4f}")

    # ── Metric 4: Rank correlation ──
    print(f"\n── Spearman rank correlation (first {min(nq, 10)} probes) ──")
    from scipy.stats import spearmanr
    for qi in range(min(nq, 5)):
        sims_int8 = np.dot(emb_int8, probe_int8[qi])
        sims_fp32 = np.dot(emb_fp32, probe_fp32[qi])
        rho, _ = spearmanr(sims_int8, sims_fp32)
        print(f"  probe {qi}: ρ = {rho:.4f}")

    # Write report
    report = {
        "benchmark": "perseus-vault-fp32-vs-int8",
        "issue": "#630",
        "corpus": {"n_entities": n, "n_probes": nq},
        "models": {
            "int8": "all-MiniLM-L6-v2 qint8 (bundled, 23MB)",
            "fp32": "all-MiniLM-L6-v2 FP32 (HuggingFace, 86MB)",
        },
        "self_consistency": {
            "mean_cosine": float(cos_self.mean()),
            "min_cosine": float(cos_self.min()),
            "median_cosine": float(np.median(cos_self)),
        },
        "nn_agreement": {},
    }
    for k in ks:
        if k > n:
            break
        overlaps = []
        for qi in range(min(nq, 100)):  # 100 probe subset for report
            sims_int8 = np.dot(emb_int8, probe_int8[qi])
            sims_fp32 = np.dot(emb_fp32, probe_fp32[qi])
            overlaps.append(rank_agreement(sims_int8, sims_fp32, k))
        report["nn_agreement"][f"top_{k}"] = round(float(np.mean(overlaps)), 4)

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nReport: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
