from gensim.models import Word2Vec

# Corpus of sentences
sentences = [['I', 'love', 'chocolate'],
             ['I', 'love', 'ice', 'cream'],
             ['I', 'enjoy', 'eating', 'chocolate', 'and', 'ice', 'cream']]

# Train the Word2Vec model
model = Word2Vec(sentences, min_count=1)

# Get word embeddings
chocolate_embedding = model.wv['chocolate']
ice_cream_embedding = model.wv['ice']

# Find similar words
similar_words = model.wv.most_similar('chocolate')

# Perform word arithmetic
result = model.wv.most_similar(positive=['chocolate', 'ice'], negative=['eating'])

# Print results
print("Embedding for 'chocolate':", chocolate_embedding)
print("Embedding for 'ice':", ice_cream_embedding)
print("Similar words to 'chocolate':", similar_words)
print("Word arithmetic result:", result)
