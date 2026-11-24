
"""
Retrieval evaluation across multiple similarity thresholds.

Run from project folder:
python eval/run_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

import config
from rag import retrieve


# Load evaluation questions
data = json.load(
    open(
        Path(__file__).parent / "questions.json",
        encoding="utf-8"
    )
)["questions"]


# Thresholds to compare
thresholds = [0.30, 0.35, 0.40, 0.45]

total_ans = sum(1 for q in data if q["answerable"])
total_unans = sum(1 for q in data if not q["answerable"])


print("\n========== RAG THRESHOLD EVALUATION ==========")
print(f"Chunk size: {config.CHUNK_SIZE}")
print(f"Questions: {len(data)}")
print("=" * 65)


# Retrieve results once for each question
results = {}

for q in data:
    results[q["id"]] = retrieve(q["question"], k=3)


# Evaluate each threshold
for threshold in thresholds:

    hits = 0
    refused = 0
    misses = []
    leaks = []

    for q in data:

        res = results[q["id"]]

        if q["answerable"]:

            # Correct source and page must appear
            # within top 3 and pass the threshold.
            ok = any(
                r["source"] == q["source_file"]
                and r["page"] == q["page"]
                and r["score"] >= threshold
                for r in res
            )

            if ok:
                hits += 1
            else:
                misses.append(q)

        else:

            # Unsupported questions should be refused.
            ok = (
                not res
                or res[0]["score"] < threshold
            )

            if ok:
                refused += 1
            else:
                leaks.append(q)

    hit_rate = hits / total_ans if total_ans else 0
    refusal_rate = refused / total_unans if total_unans else 0

    print(f"\nTHRESHOLD: {threshold:.2f}")
    print(
        f"Retrieval hit rate: "
        f"{hits}/{total_ans} = {hit_rate:.0%}"
    )
    print(
        f"Correct refusals: "
        f"{refused}/{total_unans} = {refusal_rate:.0%}"
    )
    print(f"MISS count: {len(misses)}")
    print(f"LEAK count: {len(leaks)}")

    for q in misses:
        print(f"  MISS #{q['id']}: {q['question']}")

    for q in leaks:
        print(f"  LEAK #{q['id']}: {q['question']}")


print("\n========== EVALUATION COMPLETE ==========")
print(
    "Note: This evaluates retrieval and threshold behavior, "
    "not the final quality of Gemini-generated answers."
)
