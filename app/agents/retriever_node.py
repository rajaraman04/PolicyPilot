"""Node 2: Retriever.

Pulls the top-k relevant chunks from ChromaDB, attaching citations
(document name + page) to state.
"""

from app.graph import GraphState
from app.retriever import Retriever

_retriever = Retriever()


def retriever_node(state: GraphState) -> GraphState:
    """Retrieve evidence chunks for the planned query.

    TODO: use state["plan"]/state["question"] to fetch citations via Retriever.
    """
    raise NotImplementedError("Retriever node pending.")
