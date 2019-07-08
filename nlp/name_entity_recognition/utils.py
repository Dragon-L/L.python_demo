from collections import defaultdict


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
    """
        tokens_or_tags: a list of lists of tokens or tags
        special_tokens: some special tokens
    """
    tok2idx = defaultdict(lambda: 0)
    idx2tok = []

    special_tokens_set = set(special_tokens)
    tokens_set = set([token for tokens_list in tokens_or_tags for token in tokens_list])
    filtered_tokens = list(tokens_set - special_tokens_set)
    idx2tok.extend(special_tokens)
    idx2tok.extend(filtered_tokens)
    tok2idx = {token: index for index, token in enumerate(idx2tok)}
    return tok2idx, idx2tok