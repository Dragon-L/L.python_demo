from typing import List, Tuple

import gensim
import numpy as np
from nltk import WhitespaceTokenizer

question = 'converting string to list'
candicates = ['Convert Google results object (pure js) to Python object',
              'C# create cookie from string and send it',
              'How to use jQuery AJAX for an outside domain?']


def sentence_to_vector(tokens: str) -> np.ndarray:
    # filter unexpeted words
    tokens = [token for token in tokens if token in wv_embeddings]

    tokens_embeddings = [wv_embeddings[token] for token in tokens]

    return np.mean(tokens_embeddings, axis=0)


def rank_candicates(question: str, candicates: List[str]) -> List[Tuple[int, str]]:
    question_vec = sentence_to_vector(question)
    candicates_vec = [sentence_to_vector(candicate) for candicate in candicates]
    # similarities = cosine_simi


rank_candicates(question, candicates)


def init():
    global wv_embeddings, dimentions
    pre_trained_word_vectors = '/Users/glliao/WorkSpace/Data/GoogleNews-vectors-negative300.bin'
    dimentions = 300
    wv_embeddings = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_word_vectors, binary=True)


def main(question: str, candicates: List[str]) -> None:
    init()

    # Step 1: 分词
    tokenizer = WhitespaceTokenizer()
    question_tokens = tokenizer.tokenize(question)
    candicates_tokens = tokenizer.tokenize(candicates)

    # Step 2: sentence embedding
    sentence_to_vector(question_tokens)


if __name__ == '__main__':
    main(question, candicates)
