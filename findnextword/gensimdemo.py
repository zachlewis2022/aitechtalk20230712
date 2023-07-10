#
# CHATGPT Generated Python Code
#
# Prompt: Explain word2vec in bullet form and a simple example

# In this example, we have a corpus of sentences represented as a list of lists. We use the Word2Vec class from the Gensim library to train the Word2Vec model on the given sentences.
# After training, we can access word embeddings using the wv attribute of the model. We retrieve the embeddings for specific words like 'chocolate' and 'ice' and store them in variables.
# We can also find similar words to a given word using the most_similar method. In this case, we find similar words to 'chocolate' and print the results.
# Additionally, we demonstrate word arithmetic by finding the word that is similar to 'chocolate' and 'ice' but dissimilar to 'eating'. The `most

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
