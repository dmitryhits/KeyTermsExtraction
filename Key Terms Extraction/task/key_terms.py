# Write your code here
import string

from lxml import etree
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
#import nltk


def treebank2wordnet(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V') or treebank_tag == 'MD':
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'



# nltk.download('averaged_perceptron_tagger')
stopwords_list = stopwords.words('english')
punctuation_list = list(string.punctuation)
remove_list = stopwords_list + punctuation_list

tree = etree.parse('news.xml')
root = tree.getroot()
corpus = root[0]
wnl = WordNetLemmatizer()
tfidf = TfidfVectorizer()
dataset = []
titles = []

for news_item in corpus:
    for part in news_item:
        if "name" in part.keys():
            prop = part.get('name')
            if prop == 'head':
                titles.append(part.text + ':')
                # print(part.text, end=':\n')
            elif prop == 'text':
                tokens_pos = pos_tag(word_tokenize(part.text.lower()))
                # collect the basic form of all words
                tokens_lemmas = [wnl.lemmatize(word) for word, tag in tokens_pos]
                # collect all tokens but removing stop words and punctuation marks
                tokens = [word for word in tokens_lemmas if word not in remove_list]
                nouns = [word for word in tokens if pos_tag([word])[0][1] == 'NN']
                # add all the nouns for the story to the dataset for tf-idf scoring
                dataset.append(' '.join(nouns))
                ### this code was used in previous stages ####
                tokens_freq = Counter(tokens).items()
                nouns_freq =Counter(nouns).items()
                # First sort alphabetically in descending order
                alpha_sort = sorted(nouns_freq, key=lambda x: x[0], reverse=True)
                # Then sort by frequency in descending order
                freq_sort = sorted(alpha_sort, key=lambda x: x[1], reverse=True)
                # select 5 most common and sort them by occurence
                most_common_5 = [word for word, freq in Counter(tokens).most_common(5)]
                occurrence_sort = [most_common_5.pop(most_common_5.index(w)) for w in tokens if w in most_common_5]


tfidf_matrix = tfidf.fit_transform(dataset)
for title, row in zip(titles, tfidf_matrix):
    print(title)
    # decode the feature names (nouns) from the second index in the matrix.
    # Essentially convert each row of tf-idf matrix into list of tuples (noun, tf-idf score)
    tfidf_decoded = [(tfidf.get_feature_names()[k[1]], v) for k, v in row.todok().items()]
    # first sort the list by nouns alphabetically
    alpha_sort = sorted(tfidf_decoded, key=lambda x: x[0], reverse=True)
    #  then sort it by tf-idf score
    freq_sort = sorted(alpha_sort, key=lambda x: x[1], reverse=True)
    most_common = 0
    for token, freq in freq_sort:
        most_common += 1
        if most_common <= 5:
            print(token, end=' ')
        else:
            print('\n')
            break
