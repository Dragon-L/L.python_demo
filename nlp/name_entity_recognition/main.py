from nlp.name_entity_recognition.utils import *

special_tokens = ['<UNK>', '<PAD>']
special_tags = ['O']

def words2idxs(tokens_list):
    return [token2idx[word] for word in tokens_list]

def tags2idxs(tags_list):
    return [tag2idx[tag] for tag in tags_list]

def idxs2words(idxs):
    return [idx2token[idx] for idx in idxs]

def idxs2tags(idxs):
    return [idx2tag[idx] for idx in idxs]


def main():
    train_tokens, train_tags = read_data('data/train.txt')
    validation_tokens, validation_tags = read_data('data/validation.txt')
    test_tokens, test_tags = read_data('data/test.txt')


    # Create dictionaries
    token2idx, idx2token = build_dict(train_tokens + validation_tokens, special_tokens)
    tag2idx, idx2tag = build_dict(train_tags, special_tags)






if __name__ == '__main__':
    main()