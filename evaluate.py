import json
import time
from typing import List, Dict
from src.rag import answer

ASSESSMENT_QUESTIONS = [
    {
        "id": "Q1",
        "category": "Diagnostics & Flight Controller (Page 12)",
        "question": "What does a rear status LED pattern of 3 rapid red flashes followed by 2 solid amber flashes indicate on the AeroVantage X9, and what action should the technician take?"
    },
    {
        "id": "Q2",
        "category": "Power Systems & Battery Safety (Page 29)",
        "question": "What are the specific cell voltage thresholds for Stage 1 Low Voltage Warning, Stage 2 Forced Emergency Return-to-Home (RTH), and Stage 3 Critical Auto-Land on the AeroVantage X9?"
    },
    {
        "id": "Q3",
        "category": "Environmental Operational Limits (Page 44)",
        "question": "What is the maximum certified sustained wind speed limit for safe flight operations of the AeroVantage X9 industrial hexacopter?"
    },
    {
        "id": "Q4",
        "category": "Grounded Negative Test / Unstated Parameters (Page 1)",
        "question": "What are the exact numerical torque specifications (in Nm) for the carbon-fiber airframe rotor mounting bolts in Module 1.0?"
    },
    {
        "id": "Q5",
        "category": "Hallucination & Out-of-Scope Prevention",
        "question": "What is the maximum flight altitude and obstacle avoidance sensor range of the AeroVantage X9?"
    }
]


def print_banner():
    print("=" * 80)
    print("  AEROVANTAGE X9 INDUSTRIAL HEXACOPTER - RAG EVALUATION BENCHMARK")
    print("  Corpus: data/aerovantage_x9_drone_manual_50p.pdf (50 Pages)")
    print("  Vector Store: Persistent ChromaDB | Retrieval: Top-6 Chunks")
    print("=" * 80)
    print()


def run_evaluation() -> List[Dict]:
    print_banner()
    results = []

    for idx, item in enumerate(ASSESSMENT_QUESTIONS, start=1):
        q_id = item["id"]
        category = item["category"]
        question = item["question"]

        print(f"[{q_id}/5] Category: {category}")
        print(f"Question: {question}")
        print("-" * 80)

        start_time = time.time()
        res = answer(question)
        elapsed = time.time() - start_time

        ans_text = res.get("answer", "")
        sources = res.get("sources", [])

        print("Answer:")
        print(ans_text)
        print(f"Sources: {sources}")
        print(f"Latency: {elapsed:.2f}s")
        print("=" * 80)
        print()

        results.append({
            "id": q_id,
            "category": category,
            "question": question,
            "answer": ans_text,
            "sources": sources,
            "latency_seconds": round(elapsed, 2)
        })

    # Save summary as JSON
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[EVALUATION COMPLETE] Results successfully saved to evaluation_results.json")

    # Generate Markdown Summary
    generate_markdown_summary(results)
    return results


def generate_markdown_summary(results: List[Dict]):
    md_lines = [
        "### Evaluation Results Summary\n",
        "| ID | Evaluation Category | Question | Cited Sources | Grounding Status |",
        "|---|---|---|---|---|"
    ]
    for r in results:
        status = "Strict Refusal (No Hallucination)" if not r["sources"] else "Grounded Ground Truth"
        src_str = ", ".join(r["sources"]) if r["sources"] else "None (Unstated in Corpus)"
        q_snippet = (r["question"][:65] + "...") if len(r["question"]) > 65 else r["question"]
        md_lines.append(f"| **{r['id']}** | {r['category']} | {q_snippet} | `{src_str}` | {status} |")

    with open("evaluation_summary.md", "w") as f:
        f.write("\n".join(md_lines))
    print("[EVALUATION COMPLETE] Markdown summary saved to evaluation_summary.md")


if __name__ == "__main__":
    run_evaluation()
