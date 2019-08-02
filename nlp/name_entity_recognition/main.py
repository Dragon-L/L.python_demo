from model import Model
from utils import *

SPECIAL_TOKENS = ['<UNK>', '<PAD>']
SPECIAL_TAGS = ['O']
BATCH_SIZE = 32
IS_SHUFFLE = True
EPOCH = 4
N_HIDDEN_RNN = 200
EMBEDDING_DIM = 200
DROPOUT_KEEP_PROBABILITY = 0.5
LEARNING_RATE = 0.005
LEARNING_RATE_DECAY = np.sqrt(2)


def main():
    train_tokens, train_tags = read_data('data/train.txt')
    validation_tokens, validation_tags = read_data('data/validation.txt')
    test_tokens, test_tags = read_data('data/test.txt')

    # Create dictionaries
    token2idx, idx2token = build_dict(train_tokens + validation_tokens, SPECIAL_TOKENS)
    tag2idx, idx2tag = build_dict(train_tags, SPECIAL_TAGS)
    pad_token_index = token2idx[SPECIAL_TOKENS[1]]
    pad_tag_index = tag2idx[SPECIAL_TAGS[0]]

    train_x, train_y = convert_to_index(token2idx, train_tokens), convert_to_index(tag2idx, train_tags)
    val_x, val_y = convert_to_index(token2idx, validation_tokens), [tag for sequence in validation_tags for tag in
                                                                    sequence]
    test_tokens = [[token if token in token2idx else '<UNK>' for token in tokens] for tokens in test_tokens]
    test_x, test_y = convert_to_index(token2idx, test_tokens), [tag for sequence in test_tags for tag in sequence]

    model = Model(EMBEDDING_DIM, len(idx2tag), len(idx2token), N_HIDDEN_RNN, pad_token_index, pad_tag_index, idx2tag,
                  DROPOUT_KEEP_PROBABILITY,
                  LEARNING_RATE, LEARNING_RATE_DECAY)
    model.train(train_x, train_y, BATCH_SIZE, IS_SHUFFLE, EPOCH, val_x, val_y, test_x, test_y)


if __name__ == '__main__':
    main()