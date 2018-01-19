# -*- coding: utf-8 -*-
#export CUDA_VISIBLE_DEVICES=0
#export TF_CPP_MIN_LOG_LEVEL=3

import time
import os
import argparse

import tensorflow as tf
import numpy as np
from sklearn import metrics

import dataUtils
import commonUtils
from models import *


parser = argparse.ArgumentParser(description='A subject-based sentiment analysis model')
parser.add_argument('stage', default='predict', choices=['train', 'test', 'predict', 'visual'], help='train, test, visualize or predict, default is predict')
parser.add_argument('-it', '--input_train', default=None, help='File path of training data')
parser.add_argument('-ie', '--input_test', default=None, help='File path of evaluation data')
parser.add_argument('-ip', '--input_predict', default=None, help='File path of data for predict polarity')
parser.add_argument('-v', '--vocab_file', default=None, help='File path of vocabulary file')

parser.add_argument('-l', '--log_dir', default='/jason/subject-sentiment/tensorboard/subj-senti/lstmattention', help='Path for saving Tensorboard')
parser.add_argument('-m', '--model_dir', default=None, help='Path for saving model')

parser.add_argument('-al', '--algorithm', default=None, help='[attention, bilstm, rnn, mlp, lr]')
args = parser.parse_args()


def configure_saver(model_dir, log_dir):
    print('Configuring the Tensorboard and Saver...')
    
    commonUtils.recreate_folder(log_dir)

    # record a tensor(verctor or matrix) value
    # tf.summary.histogram('vec_name', vec) 
    # record a scalar value
    # 使用tf.summary.scalar记录标量 
    # 使用tf.summary.histogram记录数据的直方图 
    # 使用tf.summary.distribution记录数据的分布图 
    # 使用tf.summary.image记录图像数据 
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    # merging all recorded value together and write to --logdir, need run in session
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir+'/train')
    val_writer = tf.summary.FileWriter(log_dir+'/validate')

    model_saver = tf.train.Saver()
    commonUtils.recreate_folder(model_dir)
    model_path = os.path.join(model_dir, 'best_val_model')

    return train_writer, val_writer, model_saver, merged_summary, model_path


def feed_data(x_batch, offset_concat_batch, offset_weight_batch, keep_prob, y_batch=None):
    if y_batch is not None:
        feed_dict = {
        model.X: x_batch,
        model.Y: y_batch,
        model.Offset_concat: offset_concat_batch,
        model.Offset_weight: offset_weight_batch,
        model.keep_prob: keep_prob
    }
    else:
        feed_dict = {
        model.X: x_batch,
        model.Offset_concat: offset_concat_batch,
        model.Offset_weight: offset_weight_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_, offset_concat, offset_weight):
    '''get average loss and accuracy in a dataset'''
    data_count = len(x_)
    eval_batch = dataUtils.batch_iter(x_, offset_concat, offset_weight, category_Y=y_)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch, offset_concat_batch, offset_weight_batch in eval_batch:
        batch_count = len(x_batch)
        feed_dict = feed_data(x_batch, offset_concat_batch, offset_weight_batch, ModelConfig.dropout_keep_prob, y_batch)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_count
        total_acc += acc * batch_count
    return total_loss/data_count, total_acc/data_count


def train(datas, offsets, labels, model_dir, vocab_dir, log_dir):
    train_writer, val_writer, model_saver, model_summary, model_saved_path = configure_saver(model_dir,log_dir)

    print('Loading vocabulary and categories...')
    _, token_to_id = dataUtils.read_vocab(vocab_dir)
    category_to_id, id_to_category = dataUtils.build_categoriesId(TrainConfig.categories)

    print('Loading training data...')
    start_time = time.time()
    X_padded = dataUtils.encode_data(datas, token_to_id, TrainConfig.seq_length)
    Y_onehot = dataUtils.encode_categories(labels, category_to_id)
    Offset_concat, Offset_weight = dataUtils.encode_offset(offsets, TrainConfig.seq_length)
    spent_time = commonUtils.get_spent_times(start_time)
    print('Time usage:', spent_time)

    # split datas into training dataset and validation dataset
    total_len = len(X_padded)
    split_len = int(total_len*0.8)
    train_data = X_padded[:split_len]
    train_label = Y_onehot[:split_len]
    train_offset_concat = Offset_concat[:split_len]
    train_offset_weight = Offset_weight[:split_len]
    val_data = X_padded[split_len:]
    val_label = Y_onehot[split_len:]
    val_offset_concat = Offset_concat[split_len:]
    val_offset_weight = Offset_weight[split_len:]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_writer.add_graph(sess.graph)
    val_writer.add_graph(sess.graph)

    print('Traning and Validation...')
    start_time = time.time()
    current_batch = 0
    best_val_acc = 0.0
    unimproved_epoch_num = 0

    for epoch in range(TrainConfig.num_epoch):
        print('Epoch: {}...'.format(epoch+1))
        train_batch = dataUtils.batch_iter(train_data, train_offset_concat, train_offset_weight, batch_size=TrainConfig.batch_size, category_Y=train_label)
        for X_batch, Y_batch, offset_concat_batch, offset_weight_batch in train_batch:  
            
            if current_batch % TrainConfig.print_per_batch == 0:
                feed_dict_train_print = feed_data(X_batch, offset_concat_batch, offset_weight_batch, ModelConfig.dropout_keep_prob, Y_batch)
                train_summary, train_loss, train_acc = sess.run([model_summary,model.loss, model.acc], feed_dict=feed_dict_train_print)
                train_writer.add_summary(train_summary, current_batch)
                
            #   train_loss, train_acc = sess.run([model.loss, model.acc], feed_dict=feed_dict_train_print)
                feed_dict_val = feed_data(val_data, val_offset_concat, val_offset_weight, ModelConfig.dropout_keep_prob, val_label)
                val_summary, val_loss, val_acc = sess.run([model_summary, model.loss, model.acc], feed_dict=feed_dict_val)
                val_writer.add_summary(val_summary, current_batch)
                spent_time = commonUtils.get_spent_times(start_time)
                msg = 'Iteration: {0:>6d}, Train Loss: {1:>.2f}, Train Acc: {2:>.2%}, Val Loss: {3:>.2f}, Val Acc: {4:.2%}, Time: {5}'
                print(msg.format(current_batch, train_loss, train_acc, val_loss, val_acc, spent_time))
            
            # training
            feed_dict_train = feed_data(X_batch, offset_concat_batch, offset_weight_batch, TrainConfig.dropout_keep_prob, Y_batch)
            #print("X_batch:\n",X_batch)
            #print("====================================attend shape:\n",sess.run(tf.shape(model.attend_mul), feed_dict=feed_dict_train))
            #print("====================================bilstm shape:\n",sess.run(tf.shape(model.last_bilstm_output), feed_dict=feed_dict_train))
            #print("====================================add shape:\n",sess.run(tf.shape(model.linear), feed_dict=feed_dict_train))
            #print("bilstm:\n",sess.run(model.bilstm_output, feed_dict=feed_dict_train))
            #print("Y_batch:\n",Y_batch)
            sess.run(model.optim, feed_dict = feed_dict_train)
            current_batch += 1

        # early stop
        val_loss, val_acc = evaluate(sess, val_data, val_label, val_offset_concat, val_offset_weight)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            unimproved_epoch_num = 0
            model_saver.save(sess=sess, save_path=model_saved_path)
        else:
            unimproved_epoch_num += 1

        if unimproved_epoch_num >= TrainConfig.early_stopping_interval:
            print('No optimization for {} epoch, auto early stopping...'.format(TrainConfig.early_stopping_interval))
            break

    train_writer.close()
    val_writer.close()
    print('Finished training, please find model at {}'.format(model_saved_path))


def test(test_dir, model_dir, vocab_dir):
    print('Loading vocabulary and categories...')
    _, token_to_id = dataUtils.read_vocab(vocab_dir)
    category_to_id, id_to_category = dataUtils.build_categoriesId(ModelConfig.categories)

    print('Loading test data...')
    start_time = time.time()
    test_data, test_offsets, test_labels = dataUtils.read_csv(test_dir)

    test_data_padded = dataUtils.encode_data(test_data, token_to_id, ModelConfig.seq_length)
    test_label_onehot = dataUtils.encode_categories(test_labels, category_to_id)
    test_offset_concat, test_offset_weight = dataUtils.encode_offset(test_offsets, ModelConfig.seq_length)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=os.path.join(model_dir, 'best_val_model'))

    print('Starting testing...')
    test_loss, test_acc = evaluate(sess, test_data_padded, test_label_onehot, test_offset_concat, test_offset_weight)
    msg = 'Test Loss: {0:>.2f}, Test Acc: {1:>.2%}'
    print(msg.format(test_loss, test_acc))

    y_pred_cls = np.zeros(shape=len(test_data), dtype=np.int32)
    test_batch = dataUtils.batch_iter(test_data_padded, test_offset_concat, test_offset_weight, shuffle=False)
    start_idx = 0
    for X_batch, offset_concat_batch, offset_weight_batch in test_batch:
            feed_dict = feed_data(X_batch, offset_concat_batch, offset_weight_batch, ModelConfig.dropout_keep_prob)
            y_pred_cls[start_idx:(start_idx+len(X_batch))] = sess.run(model.y_pred_cls, feed_dict=feed_dict)
            start_idx += len(X_batch)

    # evaluation
    y_test_cls = np.argmax(test_label_onehot, 1)
    print('Precision, Recall and F1-Score...')
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=ModelConfig.categories, digits=4))

    # confusion matrix
    print('Confusion Matrix...')
    print(metrics.confusion_matrix(y_test_cls, y_pred_cls))

    spent_time = commonUtils.get_spent_times(start_time)
    print('Time usage: {} \n'.format(spent_time))


def predict(data_dir, model_dir, vocab_dir):
    print('Loading vocabulary and categories...')
    _, token_to_id = dataUtils.read_vocab(vocab_dir)
    category_to_id, id_to_category = dataUtils.build_categoriesId(ModelConfig.categories)

    print('Loading data for prediction...')
    start_time = time.time()
    _data, _offsets, _ = dataUtils.read_csv(data_dir, has_label=False)

    _data_padded = dataUtils.encode_data(_data, token_to_id, ModelConfig.seq_length)
    _offset_concat, _offset_weight = dataUtils.encode_offset(_offsets, ModelConfig.seq_length)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=os.path.join(model_dir, 'best_val_model'))

    print('Starting predicting...')
    _pred_cls = np.zeros(shape=len(_data), dtype=np.int32)
    _batch = dataUtils.batch_iter(_data_padded, _offset_concat, _offset_weight, shuffle=False)
    start_idx = 0
    for X_batch, offset_concat_batch, offset_weight_batch in _batch:
            feed_dict = feed_data(X_batch, offset_concat_batch, offset_weight_batch, ModelConfig.dropout_keep_prob)
            _pred_cls[start_idx:(start_idx+len(X_batch))] = sess.run(model.y_pred_cls, feed_dict=feed_dict)
            start_idx += len(X_batch)

    # predict result
    print('\n'.join([id_to_category[x] for x in _pred_cls]) + '\n')
    spent_time = commonUtils.get_spent_times(start_time)
    print('Finished prediction. Time usage:', spent_time)


def visualize(data_dir, model_dir, vocab_dir):
    print('Loading vocabulary and categories...')
    _, token_to_id = dataUtils.read_vocab(vocab_dir)
    category_to_id, id_to_category = dataUtils.build_categoriesId(ModelConfig.categories)

    print('Loading data for prediction...')
    start_time = time.time()
    _data, _offsets, _ = dataUtils.read_csv(data_dir, has_label=False)

    _data_padded = dataUtils.encode_data(_data, token_to_id, ModelConfig.seq_length)
    _offset_concat, _offset_weight = dataUtils.encode_offset(_offsets, ModelConfig.seq_length)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=os.path.join(model_dir, 'best_val_model'))

    print('Starting visualization...')
    feed_dict = feed_data(_data_padded, _offset_concat, _offset_weight, ModelConfig.dropout_keep_prob)
    _align, _attention, _pred_cls = sess.run([model.attention_layer.g_t, model.attention_layer.a_t, model.y_pred_cls], feed_dict=feed_dict)
    print('attention\n{}\n'.format(_attention.tolist()))
    print('align\n{}\n'.format(_align.tolist()))

    #_pweight, _pconcat, _pred_cls = sess.run([model.memory_layer.pos_weight, model.memory_layer.offset_concat, model.y_pred_cls], feed_dict=feed_dict)
    #print('weight\n{}\n'.format(_pweight.tolist()))
    #print('concat\n{}\n'.format(_pconcat.tolist()))

    # predict result
    print('\n'.join([id_to_category[x] for x in _pred_cls]) + '\n')
    spent_time = commonUtils.get_spent_times(start_time)
    print('Finished prediction. Time usage:', spent_time)


if __name__ == '__main__':

    assert args.vocab_file is not None
    assert args.model_dir is not None

    stage = args.stage
    if stage not in ['train', 'test', 'predict', 'visual']:
        raise ValueError("""usage: python run_model.py[train/test/predict/visual]""")

    vocab_dir = args.vocab_file
    model_dir = args.model_dir
    model = LstmMultiAttentionModel(args.algorithm)

    # train stage
    if stage == 'train':
        assert args.input_train is not None
        commonUtils.check_file_exist(args.input_train, fileType='training data')

        # if vacabulary isn't existed, then build one
        datas, offsets, labels = dataUtils.read_csv(args.input_train, do_shuffle=True)
        if not commonUtils.check_file_exist(vocab_dir, fileType='vocabulary', need_return=True):
            dataUtils.build_vocab(datas, vocab_dir, TrainConfig.vocab_size)

        log_dir = args.log_dir
        train(datas, offsets, labels, model_dir, vocab_dir, log_dir)
    else:
        commonUtils.check_file_exist(vocab_dir, fileType='vocabulary')
        commonUtils.check_file_exist(model_dir, fileType='model')
    
        if stage == 'test':
            assert args.input_test is not None
            test_dir = args.input_test
            commonUtils.check_file_exist(test_dir, fileType='testing data')
            test(test_dir, model_dir, vocab_dir)
        elif stage == 'visual':
            assert args.input_predict is not None
            predict_dir = args.input_predict
            commonUtils.check_file_exist(predict_dir, fileType='to be predicted data')
            visualize(predict_dir, model_dir, vocab_dir)
        else:
            assert args.input_predict is not None
            predict_dir = args.input_predict
            commonUtils.check_file_exist(predict_dir, fileType='to be predicted data')
            predict(predict_dir, model_dir, vocab_dir)
