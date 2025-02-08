# искать  рядом 

from sys import argv
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from transformers import pipeline, set_seed
# from generate import generate

np.random.seed(42)
torch.manual_seed(42)

model_name = 'sberbank-ai/rugpt3large_based_on_gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def load_tokenizer_and_model(model_name_or_path):
    return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path)

def generate(
            model, tok, text,
            do_sample=True, max_length=100, repetition_penalty=5.0,
            top_k=5, top_p=0.95, temperature=1,
            num_beams=None,
            no_repeat_ngram_size=3
            ):
          input_ids = tok.encode(text, return_tensors="pt")
        #   print(model.generate.__globals__['__file__'])
          out = model.generate(
              input_ids,
              max_length=max_length,
              repetition_penalty=repetition_penalty,
              do_sample=do_sample,
              top_k=top_k, top_p=top_p, temperature=temperature,
              num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
              )
          return list(map(tok.decode, out))

def create_prompt_ask():
       prompt = ( "Когда туристы приезжают в город, они часто решают, как лучше всего передвигаться по городу и найти лучшие места для посещения. "
        "Некоторые предпочитают пользоваться различными транспортными услугами, чтобы быстрее добраться до достопримечательностей. "
        "В городе также есть много интересных мест, которые туристы могут увидеть. В конечном итоге, ")
       
       return prompt

def main():
       prompt = create_prompt_ask()

       generate_text = generate(
              model=model, tok=tokenizer, text=prompt,
              max_length=800,
              repetition_penalty=2.0, top_k=50, top_p=.95, 
              temperature=.7
       )

       print('Сгнерировать текст:') 
       print(generate_text[0])

if __name__ == '__main__':
       main()

# print(generated[0])

# generator = pipeline('text-generation', model=model, tokenizer=tok)
# generated = generator(argv[1], num_beams=10, max_length=100) # Assuming argv[1] contains the prompt

# print(generated[0]['generated_text'])