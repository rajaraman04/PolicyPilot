"""Node 3: Verifier / Decision.

Checks that every claim in the drafted answer is grounded in the retrieved
text, then emits the Approved / Denied / Needs-More-Info decision and a
confidence score. Never invents citations or unsupported facts.
"""

from app.graph import GraphState


def verifier_node(state: GraphState) -> GraphState:
    """Verify grounding and produce the final decision.

    TODO: compare drafted claims against state["citations"]; set
    state["decision"], state["answer"], state["confidence"].
    """
    raise NotImplementedError("Verifier node pending.")
