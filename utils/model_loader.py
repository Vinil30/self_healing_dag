from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from peft import PeftModel, PeftConfig
import torch
from transformers import pipeline

def load_fine_tuned_model(model_path):
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
    from peft import PeftModel, PeftConfig
    import torch

    config = PeftConfig.from_pretrained(model_path)
    base_model = DistilBertForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=2,
        return_dict=True
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()

    # ✅ Fix: set label mapping
    model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    tokenizer = DistilBertTokenizer.from_pretrained(config.base_model_name_or_path)
    model.eval()
    return model, tokenizer

def create_classification_pipeline(model_path):
    """
    Create a ready-to-use text classification pipeline
    Args:
        model_path: Path to the saved model directory
    Returns:
        pipeline: Hugging Face text classification pipeline
    """
    model, tokenizer = load_fine_tuned_model(model_path)
    
    # Create pipeline
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=True
    )
    
    return pipe