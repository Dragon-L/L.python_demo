from utils import *

SPECIAL_TOKENS = ['<UNK>', '<PAD>']
SPECIAL_TAGS = ['O']
BATCH_SIZE = 32
IS_SHUFFLE = True
EPOCH = 4


def main():
    train_tokens, train_tags = read_data('data/train.txt')
    validation_tokens, validation_tags = read_data('data/validation.txt')
    test_tokens, test_tags = read_data('data/test.txt')

    # Create dictionaries
    token2idx, idx2token = build_dict(train_tokens + validation_tokens, SPECIAL_TOKENS)
    tag2idx, idx2tag = build_dict(train_tags, SPECIAL_TAGS)

    train_x, train_y = convert_to_index(token2idx, train_tokens), convert_to_index(tag2idx, train_tags)
    # val_x, val_y = convert_to_index(token2idx, validation_tokens), convert_to_index(tag2idx, validation_tags)
    # test_x, test_y = convert_to_index(token2idx, test_tokens), convert_to_index(tag2idx, test_tags)

    # tokens_batch, tags_batch = create_batch(train_x, train_y, BATCH_SIZE, IS_SHUFFLE)
    for tokens_batch, tags_batch, length in create_batch(train_x, train_y, BATCH_SIZE, IS_SHUFFLE,
                                                         token2idx[SPECIAL_TOKENS[1]], tag2idx[SPECIAL_TAGS[0]]):
        pass
    # model = Model(100, len(idx2tag))
    # model.build_layers()
    # model.train(tokens_batch, tags_batch, EPOCH)







if __name__ == '__main__':
    main()