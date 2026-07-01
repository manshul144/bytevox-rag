import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ByteVox RAG Dashboard",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ByteVox RAG Dashboard")
st.caption("Hybrid Retrieval • ChromaDB • BM25 • RRF • CrossEncoder • Groq")

# ===========================
# HEALTH
# ===========================

try:
    health = requests.get(f"{API_URL}/health").json()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "System Status",
            health["status"].upper()
        )

    with col2:
        st.metric(
            "Indexed Chunks",
            health["documents_indexed"]
        )

except Exception:
    st.error("❌ FastAPI server is not running.")
    st.stop()

st.divider()

# ===========================
# QUERY
# ===========================

st.header("Ask a Question")

question = st.text_input(
    "",
    placeholder="Example: What is Nexus AI?"
)

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:

        with st.spinner("Searching documents and generating answer..."):

            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": question
                }
            )

            if response.status_code != 200:
                st.error(response.text)
                st.stop()

            result = response.json()

        st.divider()

        # ===========================
        # ANSWER
        # ===========================

        st.header("Answer")

        st.success(result["answer"])

        # ===========================
        # METRICS
        # ===========================

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Latency (ms)",
                round(result["latency_ms"], 2)
            )

        with col2:
            st.metric(
                "Retrieved Chunks",
                len(result["retrieved_chunks"])
            )

        # ===========================
        # SOURCES
        # ===========================

        st.header("Sources")

        for source in result["sources"]:
            st.write("📄", source)

        # ===========================
        # RETRIEVED CHUNKS
        # ===========================

        st.header("Retrieved Chunks")

        for i, chunk in enumerate(result["retrieved_chunks"], start=1):

            with st.expander(
                f"Chunk {i} • {chunk['source']} • Score: {round(chunk['score'],4)}"
            ):

                st.markdown(f"**Chunk ID:** `{chunk['chunk_id']}`")

                st.markdown(
                    f"**Similarity Score:** `{round(chunk['score'],4)}`"
                )

                st.write(chunk["text"])

st.divider()

# ===========================
# RECENT LOGS
# ===========================

st.header("Recent Queries")

try:

    logs = requests.get(
        f"{API_URL}/logs"
    ).json()

    if logs:

        df = pd.DataFrame(logs)

        if "answer" in df.columns:
            df = df.drop(columns=["answer"])

        if "retrieved_chunks" in df.columns:
            df = df.drop(columns=["retrieved_chunks"])

        st.dataframe(
            df,
            use_container_width=True
        )

        # ===========================
        # DASHBOARD METRICS
        # ===========================

        st.divider()

        st.header("Dashboard Metrics")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Total Queries",
                len(df)
            )

        with c2:
            st.metric(
                "Average Latency",
                round(df["latency_ms"].mean(), 2)
            )

    else:

        st.info("No logs found.")

except Exception:

    st.warning("Unable to load logs.")

st.divider()

if st.button("🔄 Refresh Dashboard"):
    st.rerun()