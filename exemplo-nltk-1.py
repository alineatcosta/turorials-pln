import bs4 as bs  
import urllib2 
import re
import nltk

scraped_data = urllib2.urlopen('https://en.wikipedia.org/wiki/Federal_University_of_Campina_Grande')  
article = scraped_data.read()

parsed_article = bs.BeautifulSoup(article,'lxml')

# returns all the paragraphs in the article in the form of a list
paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:  
    article_text += p.text

# print(article_text)

# Removing Square Brackets and Extra Spaces
article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)  
article_text = re.sub(r'\s+', ' ', article_text)  

# Creating another object to remove special characters and digits
formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)  

# print(formatted_article_text)

# tokenize the article into sentences
sentence_list = nltk.sent_tokenize(article_text)  

# print(formatted_article_text)

# To find the frequency of occurrence of each word, we use the formatted_article_text
stopwords = nltk.corpus.stopwords.words('english')

word_frequencies = {}  
for word in nltk.word_tokenize(formatted_article_text):  
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

# print(word_frequencies)

# to find the weighted frequency (divide the number of occurances of all the words by the frequency of the most occurring word)
maximum_frequncy = max(word_frequencies.values())

for word in word_frequencies.keys():  
    word_frequencies[word] = (word_frequencies[word]/ float(maximum_frequncy))

# print(word_frequencies) 
# tudo 0

# to calculate the scores for each sentence 
sentence_scores = {}  
for sent in sentence_list:  
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

print(sentence_scores)

# To summarize the article, we can take top N sentences with the highest scores
import heapq  
summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

summary = ' '.join(summary_sentences)  
print(summary)  


