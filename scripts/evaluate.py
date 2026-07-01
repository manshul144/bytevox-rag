"""
Evaluation script for Part 1.

Runs a small benchmark of questions against the hybrid retriever + LLM
pipeline, and reports:
  - Retrieval quality: did the expected source file appear in top-k results?
  - Response quality: a quick heuristic + manual-review printout (the
    answer text is printed for human judgment, since "correctness" of a
    generated answer is not reliably automatable with simple heuristics).

This is intentionally lightweight -- the goal (per the assignment) is to
demonstrate engineering rigor in how retrieval/response quality are
*measured*, not to build a full eval harness (e.g. RAGAS) for a take-home.

Usage:
    python scripts/evaluate.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tabulate import tabulate

from app.config import settings
from app.vectorstore import VectorStore
from app.retrieval import HybridRetriever
from app.llm import generate_answer

# Each case: (question, expected_source_substring, retrieval_type_tested)
BENCHMARK = [
    {
        "question": "What pricing plan includes API access?",
        "expected_source": "pricing.md",
        "note": "Exact keyword match ('API access') -- tests BM25 contribution.",
    },
    {
        "question": "Is my data used to train ByteVox's models?",
        "expected_source": "faq.txt",
        "note": "Direct FAQ lookup -- tests basic retrieval precision.",
    },
    {
        "question": "How does ByteVox keep customer information safe and who can see it?",
        "expected_source": "security_and_compliance.pdf",
        "note": "Paraphrase of 'encryption' and 'access controls' -- tests dense/semantic retrieval (no literal keyword overlap with doc headers).",
    },
    {
        "question": "What's the difference between deploying on Managed Cloud versus inside my own AWS account?",
        "expected_source": "product_overview.md",
        "note": "Paraphrase of 'ByteVox VPC' vs 'Managed Cloud' -- tests semantic retrieval across product_overview.md and security_and_compliance.pdf (both mention VPC).",
    },
    {
        "question": "How much does the Enterprise plan cost per month?",
        "expected_source": "pricing.md",
        "note": "Tests whether the model correctly says 'custom pricing' rather than hallucinating a number -- the doc explicitly has no fixed price for Enterprise.",
    },
    {
        "question": "What is ByteVox's refund policy for annual contracts?",
        "expected_source": None,
        "note": "Negative test: this is NOT covered in any document. Checks the system says 'not enough information' instead of hallucinating.",
    },
    {
        "question": "What is the maximum file size I can upload to the Knowledge Engine on the Scale plan?",
        "expected_source": "faq.txt",
        "note": "Tests precise numeric retrieval (100MB on Scale/Enterprise vs 25MB on Starter/Growth).",
    },
]


def run_eval():
    vector_store = VectorStore()
    retriever = HybridRetriever(vector_store)
    if not retriever.load_bm25_index():
        print("BM25 index not found. Run `python scripts/ingest.py` first.")
        return
    if vector_store.count() == 0:
        print("Chroma index is empty. Run `python scripts/ingest.py` first.")
        return

    rows = []
    detailed = []

    for case in BENCHMARK:
        start = time.perf_counter()
        hits = retriever.hybrid_search(case["question"], top_k_final=settings.top_k_final)
        retrieved_sources = [h["source"] for h in hits]

        if case["expected_source"] is None:
            retrieval_hit = len(hits) == 0 or True  # informational only; judged manually below
        else:
            retrieval_hit = any(case["expected_source"] in s for s in retrieved_sources)

        try:
            answer = generate_answer(case["question"], hits) if hits else "(no chunks retrieved)"
        except Exception as e:
            answer = f"[LLM call failed: {e}]"

        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        rows.append(
            [
                case["question"][:55] + ("…" if len(case["question"]) > 55 else ""),
                "✅" if (case["expected_source"] is None or retrieval_hit) else "❌",
                ", ".join(set(retrieved_sources)) or "-",
                f"{latency_ms}ms",
            ]
        )
        detailed.append({**case, "answer": answer, "retrieved_sources": retrieved_sources, "latency_ms": latency_ms})

    print("\n=== Retrieval Quality Summary ===\n")
    print(tabulate(rows, headers=["Question", "Expected Source Hit", "Retrieved Sources", "Latency"], tablefmt="github"))

    print("\n=== Response Quality (manual review) ===\n")
    for d in detailed:
        print(f"Q: {d['question']}")
        print(f"   Note: {d['note']}")
        print(f"   Expected source: {d['expected_source']}")
        print(f"   Retrieved: {d['retrieved_sources']}")
        print(f"   Answer: {d['answer']}")
        print("-" * 100)

    n_scored = sum(1 for d in detailed if d["expected_source"] is not None)
    n_hit = sum(
        1 for d in detailed if d["expected_source"] is not None and any(d["expected_source"] in s for s in d["retrieved_sources"])
    )
    print(f"\nRetrieval hit rate (excluding negative test): {n_hit}/{n_scored} = {n_hit/n_scored:.0%}")


if __name__ == "__main__":
    run_eval()
