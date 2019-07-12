from model import Model
from utils import *

SPECIAL_TOKENS = ['<UNK>', '<PAD>']
SPECIAL_TAGS = ['O']
BATCH_SIZE = 32
IS_SHUFFLE = True
EPOCH = 4
N_HIDDEN_RNN = 200
EMBEDDING_DIM = 200


def main():
    train_tokens, train_tags = read_data('data/train.txt')
    validation_tokens, validation_tags = read_data('data/validation.txt')
    test_tokens, test_tags = read_data('data/test.txt')

    # Create dictionaries
    token2idx, idx2token = build_dict(train_tokens + validation_tokens, SPECIAL_TOKENS)
    tag2idx, idx2tag = build_dict(train_tags, SPECIAL_TAGS)
    label_size = len(idx2tag)
    pad_token_index = token2idx[SPECIAL_TOKENS[1]]
    pad_tag_index = tag2idx[SPECIAL_TAGS[0]]

    train_x, train_y = convert_to_index(token2idx, train_tokens), convert_to_index(tag2idx, train_tags)
    # val_x, val_y = convert_to_index(token2idx, validation_tokens), convert_to_index(tag2idx, validation_tags)
    # test_x, test_y = convert_to_index(token2idx, test_tokens), convert_to_index(tag2idx, test_tags)

    model = Model(EMBEDDING_DIM, len(idx2tag), len(idx2token), N_HIDDEN_RNN, pad_tag_index, pad_tag_index, 0.5, 0.001)
    model.train(train_x, train_y, BATCH_SIZE, IS_SHUFFLE, EPOCH)







if __name__ == '__main__':
    main()