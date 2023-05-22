from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
'''
sentences = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "He found a leprechaun in his walnut shell."
]
'''
'''
sentences = [
    "안녕하세요. 반갑습니다.",
    "오늘 날씨가 좋네요.",
    "한국 음식이 정말 맛있어요.",
    "어제 영화를 보러 갔는데 너무 재미있었어요.",
    "당신은 너무 못생긴 것 같네요"
]
'''
sentences = [
    "Hello. Nice to meet you.",
    "The weather is nice today.",
    "Korean food is really delicious.",
    "I went to the movies yesterday and it was so much fun.",
    "You look so ugly."
]

'''
model_name = 'bert-base-nli-mean-tokens'
model = SentenceTransformer(model_name)
sentence_vecs = model.encode(model_name)

sentence_embeddings = model.encode(sentences)


sim = cosine_similarity(
    [sentence_embeddings[0]],
     sentence_embeddings[1:]
    )
print(sim)
'''
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

# initialize dictionary to store tokenized sentences
tokens = {'input_ids': [], 'attention_mask': []}

for sentence in sentences:
    # encode each sentence and append to dictionary
    new_tokens = tokenizer.encode_plus(sentence, max_length=128,
                                       truncation=True, padding='max_length',
                                       return_tensors='pt')
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])

# reformat list of tensors into single tensor
tokens['input_ids'] = torch.stack(tokens['input_ids'])
tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

outputs = model(**tokens)

embeddings = outputs.last_hidden_state
embeddings

attention_mask = tokens['attention_mask'] # 패딩 토큰에 대한 마스크, 실제 토큰의 위치 나타냄
#attention_mask.shape

# attention_mask 텐서의 차원을 확장하여 마스크 텐서를 생성.
mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
#mask.shape
#[4, 128, 768] - 입력 시퀀스 개수, 입력 시퀀스 최대 길이, hidden size

masked_embeddings = embeddings * mask
#masked_embeddings.shape

summed = torch.sum(masked_embeddings, 1)
#summed.shape

summed_mask = torch.clamp(mask.sum(1), min=1e-9)
#summed_mask.shape

mean_pooled = summed / summed_mask


# convert from PyTorch tensor to numpy array
mean_pooled = mean_pooled.detach().numpy()

# calculate
cos = cosine_similarity(
    [mean_pooled[0]],
    mean_pooled[1:]
)
print(cos)