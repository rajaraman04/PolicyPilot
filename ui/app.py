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
            decision = data.get("decision", "—")
            confidence = data.get("confidence")
            badge = {"Approved": "✅", "Denied": "⛔", "Needs-More-Info": "❓"}.get(decision, "")
            st.subheader(f"{badge} {decision}")
            if confidence is not None:
                st.progress(min(max(confidence, 0.0), 1.0), text=f"Confidence: {confidence:.0%}")

            st.subheader("Answer")
            st.write(data.get("answer", ""))

            cols = st.columns(2)
            latency = data.get("latency_ms")
            if latency is not None:
                cols[0].caption(f"Latency: {latency:.0f} ms")
            cost = data.get("cost_usd")
            if cost is not None:
                cols[1].caption(f"Est. cost: ${cost:.5f}")

            citations = data.get("citations", [])
            st.subheader("Sources")
            if citations:
                for c in citations:
                    st.markdown(f"**{c['document']}, p.{c['page']}**")
                    if c.get("snippet"):
                        st.write(c["snippet"])
            else:
                st.write("No sources cited.")
