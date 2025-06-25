class FallbackNode:
    def run(self, state):
        print(f"\n[FallbackNode] Low confidence prediction: {state['prediction']} ({state['confidence']:.0%})")
        
        # Ask for user clarification
        clarification = input("Could you clarify? (e.g., 'It was positive' or 'It was negative'): ").strip().lower()

        # If user says 'positive' or 'negative'
        if "positive" in clarification:
            return {
                "input": state["input"],
                "user_override": "POSITIVE",
                "prediction": "POSITIVE",
                "confidence": 1.0
            }
        elif "negative" in clarification:
            return {
                "input": state["input"],
                "user_override": "NEGATIVE",
                "prediction": "NEGATIVE",
                "confidence": 1.0
            }
        
        # If user gives a new sentence or presses Enter (empty)
        if clarification:
            return {
                "input": clarification,
                "retry_attempts": state.get("retry_attempts", 0) + 1,
                "history": state.get("history", []) + [state]
            }
        else:
            print("⚠️ No clarification provided. Retrying original input.")
            return {
                "input": state["input"],
                "retry_attempts": state.get("retry_attempts", 0) + 1,
                "history": state.get("history", []) + [state]
            }
