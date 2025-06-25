from utils.model_loader import create_classification_pipeline
import torch

class InferenceNode:
    def __init__(self, model_path):
        self.classifier = create_classification_pipeline(model_path)
        self.id2label = self.classifier.model.config.id2label

    def run(self, state):
        input_text = state["input"]
        with torch.no_grad():
            results = self.classifier(input_text)[0]

        # Use proper label mapping
        formatted_results = []
        for r in results:
            readable_label = self.id2label.get(r["label"], r["label"])
            formatted_results.append({"label": readable_label, "score": r["score"]})

        top_pred = max(formatted_results, key=lambda x: x["score"])

        return {
            "input": input_text,
            "prediction": top_pred["label"],
            "confidence": top_pred["score"],
            "all_predictions": formatted_results
        }
