import spacy
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader import wordnet
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import wordnet

"""
Functions that take a list of sentences (docs) and lemmatize them

Two libraries are available, SpaCy and NLTK 
"""


def spacy_lemmatize(doc):
    lemma_words = [token.lemma_.strip() for token in doc if not token.is_stop and len(token) > 1]
    lemma_doc = ' '.join(lemma_words).strip()
    print(lemma_doc)
    return lemma_doc


def docs_lemma_spacy(docs):
    """
    Speed is not ideal, better try with NLTK for Large dataset
    :param docs:
    :return:
    """
    docs = [re.sub('[^A-Za-z]+', ' ', doc).strip() for doc in docs]
    nlp = spacy.load('en_core_web_sm')
    spacy_docs = [nlp(doc) for doc in docs]
    lemmatized_docs = [spacy_lemmatize(doc) for doc in spacy_docs]
    return lemmatized_docs


def tag_convert(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def nltk_lemma_token(word, pos):
    lemmatizer = WordNetLemmatizer()
    if pos is None:
        return word
    else:
        return lemmatizer.lemmatize(word, pos)


def nltk_lemmatizing_doc(doc):
    sw = stopwords.words('english')
    tagged_doc = nltk.pos_tag(doc)
    wordtagged_doc = [(token[0], tag_convert(token[1])) for token in tagged_doc]
    lemma_doc = [nltk_lemma_token(word=token[0], pos=token[1]) for token in wordtagged_doc]
    lemma_doc_nonstop = [token for token in lemma_doc if not token in sw and len(token) > 1]
    return ' '.join(lemma_doc_nonstop)


def docs_lemma_nltk(docs):
    # Tokenize
    docs = [re.sub('[^A-Za-z]+', ' ', doc).strip() for doc in docs]
    nltk_tokenized = [nltk.word_tokenize(doc) for doc in docs]
    # Lemmatize
    nltk_lemma_docs = [nltk_lemmatizing_doc(doc) for doc in nltk_tokenized]
    return nltk_lemma_docs
