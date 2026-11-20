# AeroVantage X9 Drone Manual - Production RAG System

A modular, production-grade Retrieval-Augmented Generation (RAG) system in Python built strictly and exclusively on the 50-page synthetic evaluation corpus: `aerovantage_x9_drone_manual_50p.pdf`.

---

## 1. Executive Summary & Architectural Overview

The **AeroVantage X9 Industrial Hexacopter Technical Maintenance Manual** is a 50-page evaluation corpus designed to rigorously test RAG precision, chunking strategy, and hallucination resistance. 
- **47 Pages** contain repetitive governance boilerplate and overview structures (Chapters 1–5, Modules 1–50).
- **3 Critical Pages** contain embedded, safety-critical quantitative specifications:
  - **Page 12 (Section 3.2):** Flight Controller Diagnostic LED Blink Codes (`ERR_IMU_MAG_07`, 15m relocation, 6-axis compass calibration).
  - **Page 29 (Section 6.4):** 6S LiPo Battery Voltage Cutoffs & Forced RTH (Stage 1 @ 3.60 V/cell, Stage 2 Forced RTH @ 3.45 V/cell, Stage 3 Auto-Land @ 3.30 V/cell).
  - **Page 44 (Section 9.3):** Environmental Operating Envelope & Wind Speed Limits (Certified max sustained wind speed of 38 knots / 70 km/h / 19.5 m/s).

This project employs a **Corpus-Aware Chunking & Retrieval Strategy** with persistent ChromaDB vector storage and strict grounded generation rules to guarantee high recall of isolated technical data without hallucination.

```
                              [ User Query ]
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │                 Retrieval Engine                       │
       │  • Persistent ChromaDB (./chroma_db)                   │
       │  • Embedding: all-MiniLM-L6-v2 (ONNX runtime)           │
       │  • Top-K: 6 Chunks (Avoids Boilerplate Dilution)        │
       └────────────────────────────┬───────────────────────────┘
                                    │ Top-6 Retrieved Chunks + Metadata
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │              Strict Grounding Engine                   │
       │  • Explicit Document Page Context Headers               │
       │  • Strict Refusal on Unstated Data / OOD Queries       │
       │  • Multi-Provider LLM Integration (Groq / Gemini / OAI)│
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
                     { "answer": <str>, "sources": [...] }
```

---

## 2. Directory Structure

```
├── .env.example                               # Environment template for LLM keys
├── .gitignore                                 # Git ignore (ignores venv, secrets, Chroma cache)
├── data/
│   └── aerovantage_x9_drone_manual_50p.pdf    # Assigned 50-page manual
├── chroma_db/                                 # Persistent ChromaDB storage directory
├── evaluate.py                                # Benchmark script for 5 assessment questions
├── evaluation_results.json                    # Serialized benchmark test results
├── evaluation_summary.md                      # Markdown test report
├── requirements.txt                           # Frozen pip dependencies
├── README.md                                  # Complete system documentation
└── src/
    ├── __init__.py
    ├── config.py                              # Central configuration & env loader
    ├── ingest.py                              # Page-by-page parser, chunker, & vector indexer
    └── rag.py                                 # ChromaDB retriever & grounded answer() function
```

---

## 3. Corpus-Aware Retrieval & Chunking Strategy

### 3.1 Page-Aware Recursive Chunking (`src/ingest.py`)
Each page in the 50-page document averages ~900–1,500 characters. Standard naive token chunking across page boundaries runs the risk of splitting critical diagnostic tables or severing section headers from their quantitative parameters.
- **Chunk Size:** `1200` characters.
- **Chunk Overlap:** `200` characters.
- **Splitter:** Recursive character splitting using hierarchical separators (`\n\n`, `\n`, `. `, ` `).
- **Result:** Critical technical pages (Page 12, Page 29, Page 44) fit entirely within a single unified chunk alongside their section headers and compliance references.

### 3.2 Metadata Tracking
Every chunk is indexed into ChromaDB with 1-indexed page metadata:
```python
{
    "page": page_num,                     # e.g., 12
    "chunk_id": f"page_{page_num}_chunk_{idx}", # e.g., "page_12_chunk_0"
    "source": f"Page {page_num}"          # e.g., "Page 12"
}
```

### 3.3 Retrieval Calibration (`top_k = 6`)
Because 47 pages share repetitive governance phrasing, setting `top_k=2` or `3` risks having boilerplate chapters rank adjacent to target technical sections. Setting `top_k=6` ensures the relevant technical chunks are always retrieved, even when surrounding boilerplate shares high syntactic similarity.

### 3.4 Skip-on-Reload Ingestion
`src/ingest.py` checks whether the persistent ChromaDB collection already exists and contains indexed documents. Re-embedding is automatically bypassed on reload unless `--force` is explicitly provided, guaranteeing instant startup times.

---

## 4. Required Function Signature (`src/rag.py`)

The system implements the exact required function signature:

```python
def answer(question: str) -> dict:
    """
    Answers a technical question using persistent ChromaDB retrieval
    and strictly grounded LLM inference.
    
    Returns:
        dict: {"answer": <str>, "sources": [<page_or_chunk_id>, ...]}
    """
```

### Strict Grounding Rules
1. **Context-Only Derivation:** Answers are generated strictly using the retrieved excerpts.
2. **Refusal on Missing Parameters:** If a query asks for a parameter that is mentioned in a title but has no numerical value in the text (e.g. Question 4: torque specs in Module 1.0), the system strictly refuses to fabricate numbers.
3. **Refusal on Out-of-Scope Queries:** If a query asks about features not covered in the manual (e.g. Question 5: flight altitude and obstacle avoidance), the system strictly states that the manual does not contain this information, returning `sources: []`.

---

## 5. Evaluation Benchmark & Results

Running `python evaluate.py` executes all 5 evaluation questions across the assessment suite:

| ID | Category | Question | Cited Sources | Grounding Status |
|:---|:---|:---|:---|:---|
| **Q1** | Diagnostics & Flight Controller (Page 12) | *What does a rear status LED pattern of 3 rapid red flashes followed by 2 solid amber flashes indicate on the AeroVantage X9, and what action should the technician take?* | `Page 12` | **Grounded Ground Truth** |
| **Q2** | Power Systems & Battery Safety (Page 29) | *What are the specific cell voltage thresholds for Stage 1 Low Voltage Warning, Stage 2 Forced Emergency Return-to-Home (RTH), and Stage 3 Critical Auto-Land on the AeroVantage X9?* | `Page 29` | **Grounded Ground Truth** |
| **Q3** | Environmental Operational Limits (Page 44) | *What is the maximum certified sustained wind speed limit for safe flight operations of the AeroVantage X9 industrial hexacopter?* | `Page 44` | **Grounded Ground Truth** |
| **Q4** | Grounded Negative Test / Unstated Parameters (Page 1) | *What are the exact numerical torque specifications (in Nm) for the carbon-fiber airframe rotor mounting bolts in Module 1.0?* | `None (Unstated in Corpus)` | **Strict Refusal (No Hallucination)** |
| **Q5** | Hallucination & Out-of-Scope Prevention | *What is the maximum flight altitude and obstacle avoidance sensor range of the AeroVantage X9?* | `None (Unstated in Corpus)` | **Strict Refusal (No Hallucination)** |

### Verifiable Output Extracts

#### Question 1 (Diagnostics):
> **Answer:** The rear status LED pattern of **3 rapid red flashes followed by 2 solid amber flashes** signals a **Primary IMU Magnetometer Calibration Fault and Compass Interference Lockout (Error Code ERR_IMU_MAG_07)**.  
> **Required technician action:**  
> - **Do not arm the motors.**  
> - **Relocate the aircraft** to a location at least **15 meters away from any steel structures**.  
> - **Run a full 6-axis compass calibration** before attempting to arm again.  
> *(Source: Page 12)*  
> **Sources:** `['Page 12']`

#### Question 2 (Battery Cutoffs):
> **Answer:**  
> - **Stage 1 – Low Voltage Warning:** 3.60 V per cell (amber telemetry alert) – *Page 29*  
> - **Stage 2 – Forced Emergency Return-to-Home (RTH):** 3.45 V per cell – *Page 29*  
> - **Stage 3 – Critical Auto-Land:** 3.30 V per cell – *Page 29*  
> **Sources:** `['Page 29']`

#### Question 3 (Wind Speed Envelope):
> **Answer:** The AeroVantage X9 is certified for safe industrial flight operations in maximum sustained wind speeds of **38 knots** (equivalent to **70 km/h** or **19.5 m/s**) [Page 44].  
> **Sources:** `['Page 44']`

#### Question 4 (Negative Grounding / Torque Specifications):
> **Answer:** The provided AeroVantage X9 technical manual does not contain information to answer this question. (See Page 1, which only outlines procedural guidelines for Module 1.0 and does not list any numerical torque values.)  
> **Sources:** `[]`

#### Question 5 (Out of Scope / Refusal):
> **Answer:** The provided AeroVantage X9 technical manual does not contain information to answer this question.  
> **Sources:** `[]`

---

## 6. Installation & Execution Guide

### 6.1 Prerequisites
- Python 3.9+
- macOS / Linux / Windows

### 6.2 Setup Environment
```bash
# Clone or navigate to the project directory
cd "Gen ai"

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 6.3 Configure Environment Variables
Copy the example configuration:
```bash
cp .env.example .env
```
Set your preferred LLM provider API key in `.env` (Groq, OpenAI, or Google Gemini). The system includes an intelligent deterministic fallback that also allows offline evaluation if API keys are not supplied.

### 6.4 Ingest the PDF Corpus
```bash
# Ingest the 50-page PDF into persistent ChromaDB
python -m src.ingest

# To force a rebuild and overwrite the existing ChromaDB collection:
python -m src.ingest --force
```

### 6.5 Run the Evaluation Suite
```bash
python evaluate.py
```
This executes all 5 assessment questions, outputs formatted terminal results, and updates `evaluation_results.json` and `evaluation_summary.md`.

### 6.6 Interactive Usage via Python
```python
from src.rag import answer

result = answer("What is the maximum certified sustained wind speed for the AeroVantage X9?")
print("Answer:", result["answer"])
print("Sources:", result["sources"])
```
