from langgraph.graph import Graph, START, END
from nodes.inference import InferenceNode
from nodes.confidence_check import ConfidenceCheckNode
from nodes.fallback import FallbackNode
from typing import Dict, Any
from datetime import datetime

class ClassificationDAG:
    def __init__(self, model_path: str):
        """Initialize the self-healing classification DAG.

        Args:
            model_path: Path to the fine-tuned model directory
        """
        self.graph = Graph()

        # Initialize nodes
        self.inference_node = InferenceNode(model_path)
        self.confidence_node = ConfidenceCheckNode(threshold=0.56)  # Configurable threshold
        self.fallback_node = FallbackNode()

        self._build_workflow()

    def _build_workflow(self) -> None:
        """Construct the DAG workflow with nodes and edges."""
        # Add nodes
        self.graph.add_node("inference", self.inference_node.run)
        self.graph.add_node("confidence_check", self.confidence_node.run)
        self.graph.add_node("fallback", self.fallback_node.run)

        # Define entrypoint
        self.graph.add_edge(START, "inference")  # ✅ Use START from langgraph.graph

        # Define normal edge
        self.graph.add_edge("inference", "confidence_check")

        # Conditional edge based on confidence
        self.graph.add_conditional_edges(
            "confidence_check",
            self._route_based_on_confidence,
            {
                "accept": END,
                "reject": "fallback",
                "retry": "inference"
            }
        )

        # Fallback can either return to inference or end
        self.graph.add_conditional_edges(
            "fallback",
            self._route_after_fallback,
            {
                "retry": "inference",
                "final": END
            }
        )

        self.app = self.graph.compile()

    def _route_based_on_confidence(self, state: Dict[str, Any]) -> str:
        """Determine next step based on confidence check."""
        if state.get("user_override"):
            return "accept"
        return self.confidence_node.decide_next(state)

    def _route_after_fallback(self, state: Dict[str, Any]) -> str:
        """Determine next step after fallback."""
        if state.get("retry_attempts", 0) >= 2:
            return "final"  # Don't allow infinite retries
        if "new_input" in state:
            return "retry"
        return "final"

    def run(self, input_text: str) -> Dict[str, Any]:
        """Execute the DAG with the given input."""
        initial_state = {
            "input": input_text,
            "retry_attempts": 0,
            "history": []
        }

        result = self.app.invoke(initial_state)

        # Add final metadata
        result["final"] = True
        result["timestamp"] = datetime.now().isoformat()
        return result
