from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F 
import torch

# Инициализация
name = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(name)

model = BertForMaskedLM.from_pretrained(name, return_dict= True)

# Вычисления
text = 'В уиверситете студенты ' + tokenizer.mask_token + ' целый день.'
input = tokenizer.encode_plus(text, return_tensors='pt')
mask_index = torch.where(input['input_ids'][0] == tokenizer.mask_token_id)

output= model(**input)

# Вывод
logits = output.logits
softmax = F.softmax(logits, dim=-1)
mask_word = softmax[0, mask_index[0], :]
top= torch.topk(mask_word, 10)

lis = []
for token in top[-1][0].data:
    res = tokenizer.decode([token])
    lis.append(res)
    # print(tokenizer.decode([token]))

print('======================================\n')
print('here below the result:\n\n',lis)