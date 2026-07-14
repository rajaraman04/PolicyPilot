"""Minimal Streamlit UI for PolicyPilot.

Run: streamlit run ui/app.py
Requires the API running: uvicorn app.main:app --reload

Bare on purpose — a question box and a display area for the answer + its cited
sources. No styling yet.
"""

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("PolicyPilot AI")
st.caption("Ask a policy question. Answers are grounded in the source documents and cite their sources.")

question = st.text_input("Question")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and answering..."):
        try:
            resp = httpx.post(f"{API_URL}/query", json={"question": question}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - surface any error to the user
            st.error(f"Request failed: {exc}")
        else:
            st.subheader("Answer")
            st.write(data.get("answer", ""))

            latency = data.get("latency_ms")
            if latency is not None:
                st.caption(f"Latency: {latency:.0f} ms")

            sources = data.get("sources", [])
            st.subheader("Sources")
            if sources:
                for s in sources:
                    st.markdown(f"**{s['document']}, p.{s['page']}**")
                    if s.get("snippet"):
                        st.write(s["snippet"])
            else:
                st.write("No sources returned.")
