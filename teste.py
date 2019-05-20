from urllib2 import Request, urlopen

link = Request('http://ultimosegundo.ig.com.br/politica/2017-04-25/reforma-da-previdencia.html',
               headers={'User-Agent': 'Mozilla/5.0'})
pagina = urlopen(link).read().decode('utf-8', 'ignore')

from bs4 import BeautifulSoup

soup = BeautifulSoup(pagina, "lxml")
texto = soup.find(id="noticia").text

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

sentencas = sent_tokenize(texto)
palavras = word_tokenize(texto.lower())

from nltk.corpus import stopwords
from string import punctuation

stopwords = set(stopwords.words('portuguese') + list(punctuation))
palavras_sem_stopwords = [palavra for palavra in palavras if palavra not in stopwords]

from nltk.probability import FreqDist

frequencia = FreqDist(palavras_sem_stopwords)

from collections import defaultdict

sentencas_importantes = defaultdict(int)

for i, sentenca in enumerate(sentencas):
    for palavra in word_tokenize(sentenca.lower()):
        if palavra in frequencia:
            sentencas_importantes[i] += frequencia[palavra]

from heapq import nlargest

idx_sentencas_importantes = nlargest(4, sentencas_importantes, sentencas_importantes.get)

for i in sorted(idx_sentencas_importantes):
    print(sentencas[i])