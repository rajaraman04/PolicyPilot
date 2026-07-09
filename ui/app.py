"""Minimal Streamlit UI.

Run: streamlit run ui/app.py

Sends the question to the FastAPI /query endpoint and renders the decision,
confidence, and citations.
"""

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PolicyPilot AI", page_icon="📋")
st.title("PolicyPilot AI")
st.caption("Ask a policy question — get an Approved / Denied / Needs-More-Info decision with citations.")

question = st.text_area("Your question", placeholder="e.g. Is remote work reimbursement covered under the travel policy?")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Consulting the policy documents..."):
        try:
            resp = httpx.post(f"{API_URL}/query", json={"question": question}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - surface any error to the user
            st.error(f"Request failed: {exc}")
        else:
            st.subheader(data.get("decision", "—"))
            st.write(data.get("answer", ""))
            conf = data.get("confidence")
            if conf is not None:
                st.metric("Confidence", f"{conf:.0%}")
            citations = data.get("citations", [])
            if citations:
                st.markdown("**Citations**")
                for c in citations:
                    st.markdown(f"- {c['document']}, p.{c['page']}")
