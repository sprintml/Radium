from .aggregation import AggregationTest
from .evalue import EvalueTest
from .welch import WelchTest

# Registry maps detection_mode name → test class.
# Add new test types here; evaluation.py and downstream callers use this registry.
REGISTRY: dict[str, type] = {
    "evalue": EvalueTest,
    "welch": WelchTest,
    "aggregation": AggregationTest,
}

__all__ = ["AggregationTest", "EvalueTest", "WelchTest", "REGISTRY"]
