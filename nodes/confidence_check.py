class ConfidenceCheckNode:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
    
    def run(self, state):
        confidence = state["confidence"]
        return {**state, "meets_confidence": confidence >= self.threshold}
    
    def decide_next(self, state):
        if state["meets_confidence"]:
            return "accept"
        return "reject"