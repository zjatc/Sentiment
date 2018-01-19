# -*- coding: utf-8 -*-

import csv
from collections import Counter
from random import shuffle
import tensorflow.contrib.keras as kr
import numpy as np

import commonUtils


def read_csv(fileName, has_header=True, has_label=True, do_shuffle=False):
    '''
        read data in .csv file and split to data, offset and label
    '''
    assert fileName.endswith('.csv')

    dataLst, offsetLst, labelLst = [], [], []
    with commonUtils.open_file(fileName) as csvf:
        if has_header:
            reader = csv.DictReader(csvf, delimiter=',', quotechar='"')
            for row in reader:
                indices = row['indices'].split('_')
                assert len(indices) == 2
                data, offset = tokenize_and_calOffset(row['text'], int(indices[0]), int(indices[1]))
                dataLst.append(data)
                offsetLst.append(offset)
                if has_label:
                    labelLst.append(row['label'])
        else:
            reader = csv.reader(csvf, delimiter=',', quotechar='"')
            for row in reader:

                indices = row[2].split('_')
                assert len(indices) == 2
                data, offset = tokenize_and_calOffset(row[1], int(indices[0]), int(indices[1]))
                dataLst.append(data)
                offsetLst.append(offset)
                if has_label:
                    labelLst.append(row[3])
    
    assert len(dataLst) == len(offsetLst)
    if has_label:
        assert len(dataLst) == len(labelLst)

    # shuffle data
    if do_shuffle:
        if has_label:
            zipped = list(zip(dataLst, offsetLst, labelLst))
        else:
            zipped = list(zip(dataLst, offsetLst))

        shuffle(zipped)
        zipLst = [list(t) for t in zip(*zipped)]
        dataLst = zipLst[0]
        offsetLst = zipLst[1]
        if has_label:    
            labelLst = zipLst[2]

    return dataLst, offsetLst, labelLst


def tokenize_and_calOffset(text, target_start, target_end):
    left_text = text[:target_start]
    left_chars = list(filter(lambda x: x>=u'\u4E00' and x <=u'\u9FA5', left_text))
    left_offsets = list(-float(len(left_chars)-i) for i in range(len(left_chars)))
 
    right_text = text[target_end:]
    right_chars = list(filter(lambda x: x>=u'\u4E00' and x <=u'\u9FA5', right_text))
    right_offsets = list(float(i+1) for i in range(len(right_chars)))
    
    chars = left_chars + ['#TARGET'] + right_chars
    offsets = left_offsets + [0.0] + right_offsets
    
    return chars, offsets


def build_vocab(data_train, vocab_dir, vocab_size):
    '''
        build vocabulary based on training data and given vocab_size 
    '''
    all_data = []
    for data in data_train:
        all_data.extend(data)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 2)
    tokens, _ = list(zip(*count_pairs))
    # add a token <PAD> in vocabulary for padded token
    tokens = ['#PAD', '#UNKNOWN'] + list(tokens)

    commonUtils.create_folder(vocab_dir)
    commonUtils.open_file(vocab_dir, mode='w').write('\n'.join(tokens) + '\n')


def read_vocab(vocab_dir):
    '''
        read vocabulary and generate tokens 
    '''
    token = commonUtils.open_file(vocab_dir).read().strip().split('\n')
    token_to_id = dict(zip(token, range(len(token))))
    return token, token_to_id


def build_categoriesId(categories):
    '''
        build category-id mapping
    ''' 
    category_to_id = dict(zip(categories, range(len(categories))))
    id_to_category = dict((k,v) for v, k in category_to_id.items())
    return category_to_id, id_to_category


def encode_categories(labels, cat_to_id):
    '''
        encode categories with one-hot encoding
    '''
    label_ids = []
    for i in range(len(labels)):
        label_ids.append(cat_to_id[labels[i]])
    encoded_label = kr.utils.to_categorical(label_ids)
    return encoded_label


def encode_data(data, token_to_id, max_seq_length):
    '''
        encode token with corresponding id and padding for truncating to max_seq_length 
        if input length less than predefined sequence length, than padding with 0 in sequence tail
        if input length greater than predefined sequence length, than truncating to max_seq_length from sequence tail
    '''
    encoded_data = []
    for instance in data:
        encoded_data.append([token_to_id[x] if x in token_to_id else token_to_id['#UNKNOWN'] for x in instance])
    padded_data = kr.preprocessing.sequence.pad_sequences(encoded_data, max_seq_length, padding='post', truncating='post')
    return padded_data


def encode_offset(offsets, max_seq_length):
    '''
        first divide each offsets by max_seq_length, then padding offsets of each sequence to max_seq_length
        offsets_concat padded 0.0 to end of sequence
        offsets_weight padded max_seq_length to the end of sequence so that for the padded charactors the memory vector will be zeros
    '''
    offsets_concat = kr.preprocessing.sequence.pad_sequences(sequences=offsets, maxlen=max_seq_length, padding='post', truncating='post')
    offsets_weight = kr.preprocessing.sequence.pad_sequences(sequences=offsets, maxlen=max_seq_length, value=max_seq_length, padding='post', truncating='post')
    return offsets_concat, offsets_weight


def batch_iter(input_X, offsets_concat, offsets_weight, batch_size=128, category_Y=None, shuffle=True):
    ''' 
        a generator to generate batched data using given batch_size, input_X and label_Y should be numpy array, batch_size should be a integer 
    '''
    dataset_size = len(input_X)
    num_batch = int((dataset_size - 1) / batch_size) + 1 # minus 1 to avoid exact division situation
    indices = np.arange(dataset_size)
    if shuffle:
        indices = np.random.permutation(np.arange(dataset_size))

    x_shuffle = input_X[indices]
    offset_concat_shuffle = offsets_concat[indices]
    offset_weight_shuffle = offsets_weight[indices]
    if category_Y is not None:
        y_shuffle = category_Y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, dataset_size)
        # yield返回一个生成器generator, 每遇到yield就会返回当前值，并记录当前的位置，下一次调用从yield语句的下一句开始
        if category_Y is not None:
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], offset_concat_shuffle[start_id:end_id], offset_weight_shuffle[start_id:end_id]
        else:
            yield x_shuffle[start_id:end_id], offset_concat_shuffle[start_id:end_id], offset_weight_shuffle[start_id:end_id]


