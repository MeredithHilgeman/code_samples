# Example Code to Start Looking at Embeddings

## Import libraries
import pandas as pd 
import numpy as np 
from vertexai.preview.language_models import TextEmbeddingModel
model = TextEmbeddingModel.from_pretrained("textembedding-gecko")

from gensim.test.utils import common_texts
from gensim.models import Word2Vec

## GCP text embedding model

embedding_a = model.get_embeddings(['king']) 
embedding_b = model.get_embeddings(['queen'])
embedding_c = model.get_embeddings(['boy'])
embedding_d = model.get_embeddings(['girl'])

for embedding in embedding_a:
    vector_a = embedding.values
for embedding in embedding_b:
    vector_b = embedding.values
for embedding in embedding_c:
    vector_c = embedding.values
for embedding in embedding_d:
    vector_d = embedding.values

print(np.dot(np.squeeze(np.array(vector_a)),np.squeeze(np.array(vector_b))))
print(np.dot(np.squeeze(np.array(vector_a)),np.squeeze(np.array(vector_c))))
print(np.dot(np.squeeze(np.array(vector_b)),np.squeeze(np.array(vector_d))))
print(np.dot(np.squeeze(np.array(vector_c)),np.squeeze(np.array(vector_d))))

## Gensim package

model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.train([['king','queen','boy','girl']], total_examples=1, epochs=1)

vector = model.wv['computer']
sims = model.wv.most_similar('computer', topn=10)
print(sims)

## Fun links

# - https://jalammar.github.io/illustrated-word2vec/
# - https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-rag-quickstart
# - https://cloud.google.com/architecture/gen-ai-rag-vertex-ai-vector-search
# - https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/introduction-prompt-design
# - https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-optimizer