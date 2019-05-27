# coding:utf-8

# LINK: https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1

# Step-1: Email Cleaning
from talon.signature.bruteforce import extract_signature

email = 'Olawoman man! How r u?\nOther text.\nOther text1.\nOther text2.\nOther text3.\nOther text4.\nOther text5.\nOther text6\nOther text7\n\n--\n\nAqui mais texto\nRegards,\nRoman'

email= 'Hey alineatcosta! A first-party GitHub OAuth application (GitHub Classroom) with repo:invite and user:email scopes was recently authorized to access your account. Visit https://github.com/settings/connections/applications/64a051cf1598b9f0658f for more information. To see this and other security events for your account, visit https://github.com/settings/security If you run into problems, please contact support by visiting https://github.com/contact. Aline.'

email = 'The Federal University of Campina Grande (Portuguese: Universidade Federal de Campina Grande, UFCG) is a public university whose main campus is located in the city of Campina Grande, Paraíba, Brazil. Together with the Federal University of Paraíba, it is the main university of the state of Paraiba, Brazil. Established after splitting from UFPB in 2002, it is one of the leading technological and scientific production institutes of northeastern Brazil, being mentioned in a 2001 edition of the Newsweek magazine as a technopole - among 9 other around the world - that represents a new vision for technology. It was again quoted in 2003 as the Brazilian silicon valley. It is one of the five institutions to have a continuous international concept from the Coordenadoria de Aperfeiçoamento de Pessoal de Nível Superior (grade over 5) in the Electrical Engineering Postgraduate Program, and one of the two to simultaneously hold a concept of 5 stars in the Technological Department (Electrical Engineering and Computer Science) from the Ministry of Education.'
# summarization - Together with the Federal University of Paraíba, it is the main university of the state of Paraiba, Brazil. It was again quoted in 2003 as the Brazilian silicon valley.

cleaned_email, _ = extract_signature(email)

print(cleaned_email)

# Step-2: Language Detection
# it is not necessary

# Step-3: Sentence Tokenization with NLTK
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(cleaned_email, language = 'english')

print(sentences)

print('sentences')

# Step-4: Skip-Thought Encoder (codificador)
# We'll use the model  Word2Vec Skip-Gram treinado para prever as palavras em torno de uma palavra de entrada.

# The 'skipthoughts' module can be found at the root of the GitHub repository linked above
import skipthoughts

# You would need to download pre-trained models first
# https://github.com/jatana-research/email-summarization

# mkdir skip-thoughts/models
# wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
model = skipthoughts.load_model()

encoder = skipthoughts.Encoder(model)
encoded =  encoder.encode(sentences)

print('encoded')
print(encoded)

# Step-5: Clustering
# The number of clusters will be equal to desired number of sentences in the summary.

import numpy as np
from sklearn.cluster import KMeans

n_clusters = np.ceil(len(encoded)**0.5).astype(np.int)
n_clusters = 2
print(n_clusters)
kmeans = KMeans(n_clusters=n_clusters)
kmeans = kmeans.fit(encoded)

# Step-6: Summarization
# The candidate sentence is chosen to be the sentence whose vector representation is closest to the cluster center.

from sklearn.metrics import pairwise_distances_argmin_min

avg = []
for j in range(n_clusters):
    idx = np.where(kmeans.labels_ == j)[0]
    avg.append(np.mean(idx))
    print('DENTRO DO FOR')
    print(idx)
    print(np.mean(idx))
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, encoded) # computa a menor distancia
ordering = sorted(range(n_clusters), key=lambda k: avg[k])
summary = ' '.join([sentences[closest[idx]] for idx in ordering])
print([email[closest[1]]])

print('ordering')
print(ordering)
print('closest')
print(closest)
print('closest')
print(closest,)


print(summary)
