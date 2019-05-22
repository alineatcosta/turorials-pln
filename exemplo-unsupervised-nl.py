# LINK: https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1

# Step-1: Email Cleaning
from talon.signature.bruteforce import extract_signature

email = 'Hey man! How r u?\nOther text\n\n--\n\nAqui mais texto\nRegards,\nRoman'

cleaned_email, _ = extract_signature(email)

print(cleaned_email)

# Step-2: Language Detection
# it is not necessary

# Step-3: Sentence Tokenization with NLTK
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(cleaned_email, language = 'english')

print(sentences)

# Step-4: Skip-Thought Encoder (codificador)
# We'll use the model  Word2Vec Skip-Gram treinado para prever as palavras em torno de uma palavra de entrada.

# The 'skipthoughts' module can be found at the root of the GitHub repository linked above
import skipthoughts

# You would need to download pre-trained models first
# https://github.com/jatana-research/email-summarization
model = skipthoughts.load_model()

encoder = skipthoughts.Encoder(model)
encoded =  encoder.encode(sentences)