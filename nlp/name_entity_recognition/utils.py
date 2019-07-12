from math import ceil

import numpy as np


def read_data(file_path):
    tokens = []
    tags = []

    tweet_tokens = []
    tweet_tags = []
    for line in open(file_path, encoding='utf-8'):
        line = line.strip()
        if line:
            token, tag = line.split()

            if token.startswith('http://') or token.startswith('https://'):
                token = '<URL>'
            elif token.startswith('@'):
                token = '<USR>'
            tweet_tokens.append(token)
            tweet_tags.append(tag)
        else:
            if tweet_tokens:
                tokens.append(tweet_tokens)
                tags.append(tweet_tags)
            tweet_tokens = []
            tweet_tags = []

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


def create_batch(tokens, tags, batch_size, is_shuffle, pad_index, other_index):
    n_samples = len(tokens)
    tokens = np.array(tokens)
    tags = np.array(tags)
    if is_shuffle:
        indexs = np.random.permutation(n_samples)
    else:
        indexs = np.arange(n_samples)

    num_of_batch = int(ceil(n_samples / batch_size))
    for current_batch in range(num_of_batch):
        start = current_batch * batch_size
        end = min((current_batch + 1) * batch_size, n_samples)
        batch_indexs = indexs[start:end]
        tokens_batch, tags_batch = tokens[batch_indexs], tags[batch_indexs]
        max_length = len(max(tokens_batch, key=len))

        tokens_batch = pad_array(max_length, pad_index, tokens_batch)
        tags_batch = pad_array(max_length, other_index, tags_batch)

        yield tokens_batch, tags_batch, max_length


def pad_array(max_length, pad_value, array):
    return [np.pad(sequence_array, (0, max_length - len(sequence_array)), 'constant', constant_values=pad_value) for
            sequence_array in array]
