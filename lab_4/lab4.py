import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load the pre-trained model and tokenizer
model_name = "bert-base-multilingual-cased"    
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

def predict_masked_word(text, target_words, top_k=10):
    """Predicts the most likely words to fill the masked token in the given text.

    Args:
        text: The input text with a masked token.
        target_words: A list of target words to check.
        top_k: The number of top predictions to return.

    Returns:
        A list of the top k predicted words.
    """

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1].item()
    mask_logits = logits[0, mask_token_index]
    top_k_indices = torch.topk(mask_logits, top_k).indices.tolist()
    top_k_words = tokenizer.batch_decode(top_k_indices)

    # Check if target words are in the top k predictions
    for target_word in target_words:
        if target_word in top_k_words:
            print(f"Target word '{target_word}' found in top {top_k} predictions.")
        else:
            print(f"Target word '{target_word}' not found in top {top_k} predictions.")

    return top_k_words

# Text usage
text = 'Сейчас [MASK] разработка новой версии программы.'

target_words = ["будет", "идёт"]

predictions = predict_masked_word(text, target_words)
print(predictions)