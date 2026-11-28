import streamlit as st
import json
import time
from pathlib import Path

from src.rag import answer, retrieve
from src.config import TOP_K, COLLECTION_NAME, PDF_PATH

st.set_page_config(
    page_title="AeroVantage X9 RAG Assistant",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #E2E8F0;
        color: #334155;
    }
    .source-tag {
        display: inline-block;
        background-color: #0284C7;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .not-found-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        color: #991B1B;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .grounded-card {
        padding: 1.2rem;
        border-radius: 8px;
        background-color: #F8FAFC;
        border-left: 4px solid #0EA5E9;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/drone.png", width=64)
    st.markdown("### AeroVantage X9 Manual RAG")
    st.caption("Industrial Hexacopter Technical Grounding Engine")
    st.markdown("---")
    
    st.markdown("#### 📂 Corpus Specifications")
    st.markdown(f"• **Source Manual:** `50 Pages`\n• **Boilerplate Ratio:** `94% (47 pages)`\n• **Critical Specs:** `Page 12, 29, 44`\n• **ChromaDB Collection:** `{COLLECTION_NAME}`\n• **Retrieval Depth:** `Top-{TOP_K} Chunks`")
    st.markdown("---")

    st.markdown("#### ⚡ Preset Benchmark Questions")
    sample_queries = {
        "Q1: LED Diagnostics (P.12)": "What does a rear status LED pattern of 3 rapid red flashes followed by 2 solid amber flashes indicate on the AeroVantage X9, and what action should the technician take?",
        "Q2: 6S LiPo Voltage (P.29)": "What are the specific cell voltage thresholds for Stage 1 Low Voltage Warning, Stage 2 Forced Emergency Return-to-Home (RTH), and Stage 3 Critical Auto-Land on the AeroVantage X9?",
        "Q3: Wind Speed Limits (P.44)": "What is the maximum certified sustained wind speed limit for safe flight operations of the AeroVantage X9 industrial hexacopter?",
        "Q4: Torque Specs (Unstated)": "What are the exact numerical torque specifications (in Nm) for the carbon-fiber airframe rotor mounting bolts in Module 1.0?",
        "Q5: Flight Altitude (Out of Scope)": "What is the maximum flight altitude and obstacle avoidance sensor range of the AeroVantage X9?"
    }

    selected_sample = st.selectbox(
        "Choose a test query:",
        options=["(Select a preset question...)"] + list(sample_queries.keys())
    )

# Header
st.markdown('<div class="main-header">🚁 AeroVantage X9 Technical RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-clean Retrieval-Augmented Generation with strict anti-hallucination grounding</div>', unsafe_allow_html=True)

# Navigation tabs
tab_query, tab_eval, tab_corpus = st.tabs(["🔍 Interactive Query", "📊 Assessment Benchmark", "📖 About Corpus & Strategy"])

with tab_query:
    col1, col2 = st.columns([4, 1])
    
    default_text = ""
    if selected_sample != "(Select a preset question...)":
        default_text = sample_queries[selected_sample]

    query_input = st.text_area(
        "Ask a technical question about the AeroVantage X9:",
        value=default_text,
        height=100,
        placeholder="e.g. What does 3 rapid red flashes and 2 solid amber flashes indicate?"
    )

    ask_button = st.button("🚀 Query Technical Manual", type="primary", use_container_width=True)

    if ask_button and query_input.strip():
        with st.spinner("Retrieving top-6 chunks from persistent ChromaDB & verifying grounding..."):
            t0 = time.time()
            res = answer(query_input.strip())
            elapsed = time.time() - t0

            retrieval_data = retrieve(query_input.strip(), top_k=TOP_K)

        st.markdown("### System Response")
        ans_text = res.get("answer", "")
        sources = res.get("sources", [])

        if ans_text == "[not found]":
            st.markdown(
                f'<div class="not-found-card">⚠️ [not found] &nbsp;&nbsp;'
                f'<span style="font-size: 0.9rem; font-weight: normal; color: #7F1D1D;">'
                f'(Strict Refusal: Parameter unstated or out-of-scope in official manual. Zero hallucination guarantee.)'
                f'</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(f'<div class="grounded-card">{ans_text}</div>', unsafe_allow_html=True)

        st.markdown("")
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.markdown("**Cited Sources:**")
            if sources:
                for s in sources:
                    st.markdown(f'<span class="source-tag">{s}</span>', unsafe_allow_html=True)
            else:
                st.caption("None (Unstated in manual)")

        with col_meta2:
            st.markdown(f"**Execution Latency:** `{elapsed:.2f}s` | **Retrieved:** `{TOP_K} Chunks`")

        st.markdown("---")
        with st.expander("🔎 Inspect Retrieved Context Chunks (Top-6 from ChromaDB)"):
            docs = retrieval_data.get("documents", [[]])[0]
            metas = retrieval_data.get("metadatas", [[]])[0]
            distances = retrieval_data.get("distances", [[]])[0]

            for idx, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
                st.markdown(f"**Chunk #{idx} | Source: `{meta.get('source', 'Unknown')}` | Distance: `{dist:.4f}`**")
                st.code(doc, language="text")

with tab_eval:
    st.markdown("### Official Benchmark Evaluation Suite")
    st.caption("Live results from `evaluation_results.json` run against the 5 canonical assessment tasks.")

    eval_file = Path("evaluation_results.json")
    if eval_file.exists():
        with open(eval_file, "r") as f:
            eval_data = json.load(f)

        for item in eval_data:
            with st.container():
                st.markdown(f"#### **{item['id']}**: {item['category']}")
                st.markdown(f"**Question:** {item['question']}")
                
                if item["answer"] == "[not found]":
                    st.error(f"**Answer:** `{item['answer']}` (Strict Refusal - No Hallucination)")
                else:
                    st.info(f"**Answer:** {item['answer']}")

                st.markdown(f"**Cited Sources:** `{item['sources']}` | **Latency:** `{item['latency_seconds']}s`")
                st.markdown("---")
    else:
        st.warning("evaluation_results.json not found. Run python evaluate.py to generate benchmarks.")

with tab_corpus:
    st.markdown("### 50-Page Synthetic Evaluation Corpus Design")
    st.markdown("""
    The **AeroVantage X9 Industrial Hexacopter Technical Maintenance Manual** contains:
    - **Total Pages:** 50 pages (1-indexed).
    - **47 Pages of Boilerplate Governance:** Repetitive institutional overview language across Modules 1.0 to 50.0.
    - **3 Isolated Technical Pages:**
      - **Page 12 (Section 3.2):** Flight controller rear LED diagnostic blink codes (`ERR_IMU_MAG_07`, 15m safety distance, 6-axis calibration).
      - **Page 29 (Section 6.4):** 6S LiPo battery multi-stage voltage cutoffs (Stage 1 @ 3.60V, Stage 2 Forced RTH @ 3.45V, Stage 3 Auto-Land @ 3.30V).
      - **Page 44 (Section 9.3):** Environmental operating envelope (38 knots / 70 km/h / 19.5 m/s maximum sustained wind speed).
    
    ### Architectural Highlights
    1. **Page-Aware Recursive Chunking (`1200` chars, `200` overlap):** Guarantees technical sections remain complete in single chunks.
    2. **Top-K Retrieval Calibration (`top_k=6`):** Prevents boilerplate dilution from burying technical pages.
    3. **Persistent ChromaDB:** Embeddings are generated once via ONNX runtime and stored in `./chroma_db` (instant reload).
    4. **Dual-Layer Anti-Hallucination:** Strictly returns `[not found]` when parameters are missing.
    """)
