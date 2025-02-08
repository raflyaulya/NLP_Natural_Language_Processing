import numpy as np
import torch  
from transformers import GPT2LMHeadModel, GPT2Tokenizer  

# Set random seed for reproducibility
np.random.seed(42)  # Sets the seed for NumPy's random number generator to ensure reproducibility
torch.manual_seed(42)  # Sets the seed for PyTorch's random number generator for consistent results

# Load model and tokenizer
model_name = "sberbank-ai/rugpt3large_based_on_gpt2"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)  

# Define the function for text generation
def generate(
        model, tok, text,  
        do_sample=True, max_length=100, repetition_penalty=5.0,  
        top_k=5, top_p=0.95, temperature=1,  
        num_beams=None, no_repeat_ngram_size=3):  

    # Encode the input text to model-compatible input IDs (PyTorch tensor)
    input_ids = tok.encode(text, return_tensors="pt")  
    
    # Generate text using the model
    out = model.generate(
        input_ids,  # Encoded input IDs
        max_length=max_length,  # Maximum length of generated text
        repetition_penalty=repetition_penalty,  # Penalizes repeated tokens
        do_sample=do_sample,  # Enables sampling instead of greedy decoding
        top_k=top_k,  
        top_p=top_p,  
        temperature=temperature,  
        num_beams=num_beams,  
        no_repeat_ngram_size=no_repeat_ngram_size  
    )

    # Decode the generated token IDs back into a list of strings (one for each output sequence)
    return list(map(tok.decode, out))


# Create a function to define the prompt
def create_prompt_ask():
    # The prompt that will guide the model's text generation
    prompt = (
        "В городе произошла необычная история: человек долго не мог найти одну вещь. "
        "Местные жители активно обсуждали это событие, предлагая свои идеи. "
        "Они исследовали все места неподалёку от центральной площади."
    )
    return prompt


def main():
    # Get the input prompt for generation
    prompt = create_prompt_ask()

    # Call the generate function with the prompt and model parameters
    generated_text = generate(
              model=model, tok=tokenizer, text=prompt,
              max_length=800,  # Maximum length of the generated text
              repetition_penalty=2.0,  # Controls penalization of repeated phrases
              top_k=50,  # Considers 50 most probable next tokens during sampling
              top_p=.95,  # Includes tokens with cumulative probability ≤ 0.95
              temperature=.7  # Makes predictions slightly more deterministic
       )

    # Print the generated text
    print("Сгенерированный текст:")
    print(generated_text[0])  

    # Validate if the generated text contains the required words
    required_words = ["искать", "рядом"]  # Words that must appear in the generated text
    for word in required_words:
        if word not in generated_text[0]:  # Check if a required word is missing
            print(f"Предупреждение: слово '{word}' отсутствует в тексте!")  
        else:
            print(f"Слово '{word}' найдено в тексте.")  


if __name__ == '__main__':
    main()  