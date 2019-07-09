from math import ceil
from random import shuffle

import numpy as np


def read_data(file_path):
    tokens = []
    tags = []

    tweet_tokens = []
    tweet_tags = []
    for line in open(file_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            if tweet_tokens:
                tokens.append(tweet_tokens)
                tags.append(tweet_tags)
            tweet_tokens = []
            tweet_tags = []
        else:
            token, tag = line.split()

            if token.startswith('http://') or token.startswith('https://'):
                token = '<URL>'
            elif token.startswith('@'):
                token = '<USR>'
            tweet_tokens.append(token)
            tweet_tags.append(tag)

    return tokens, tags


def build_dict(tokens_or_tags, special_tokens):
    idx2tok = []

    special_tokens_set = set(special_tokens)
    tokens_set = set([token for tokens_list in tokens_or_tags for token in tokens_list])
    filtered_tokens = list(tokens_set - special_tokens_set)
    idx2tok.extend(special_tokens)
    idx2tok.extend(filtered_tokens)
    tok2idx = {token: index for index, token in enumerate(idx2tok)}
    return tok2idx, idx2tok


def convert_to_index(dict, tokens_or_tags):
    return [[dict[token] for token in tokens_list] for tokens_list in tokens_or_tags]


def create_batch(tokens, tags, batch_size, is_shuffle):
    n_samples = len(tokens)
    tokens = np.array(tokens)
    tags = np.array(tags)
    indexs = np.arange(n_samples)
    if is_shuffle:
        shuffle(indexs)

    num_of_batch = int(ceil(n_samples / batch_size))
    for current_batch in range(num_of_batch - 1):
        start = current_batch * batch_size
        end = min((current_batch + 1) * batch_size, n_samples)
        batch_indexs = indexs[start:end]
        print(batch_indexs)
        yield tokens[batch_indexs], tags[batch_indexs]
    remain_indexs = indexs[end:]
    yield tokens[remain_indexs], tags[remain_indexs]
