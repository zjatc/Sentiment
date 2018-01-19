# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from layers import *



class ModelConfig(object):
    '''
        configuration for the attention subject-based sentiment analysis model.
        this configuration fit for test and predict stage, for training please use TrainConfig instead  
    '''
    seq_length = 150 # based on character and special token <TARGET> treat as one character
    vocab_size = 1500 # 1500 most common characters in training dataset
    categories = ['-1', '0', '1'] # -1 for negative, 0 for neutral, 1 for positive

    embedding_dim = 150
    embedding_trainable = True
    
    num_bilstm_layer = 1
    bilstm_hidden_dim =100

    rnn_decoder = 'gru'
    num_rnn_layer = 2
    rnn_hidden_dim = 100

    # following configuration may changed in training stage
    dropout_keep_prob = 1.0

class TrainConfig(ModelConfig):
    '''
        configuration for training model
    '''
    # override configuration in common stage
    dropout_keep_prob = 0.8

    # specific configuration for training stage
    learning_rate = 1e-3
    batch_size = 128
    num_epoch = 50

    print_per_batch = 10
    early_stopping_interval = 3



class LstmMultiAttentionModel(object):

    def __init__(self, model):
        self.X = tf.placeholder(tf.int64, [None, ModelConfig.seq_length], name='input_x')
        self.Offset_concat = tf.placeholder(tf.float32, [None, ModelConfig.seq_length], name='token_offset_concat')
        self.Offset_weight = tf.placeholder(tf.float32, [None, ModelConfig.seq_length], name='token_offset_weight')
        self.Y = tf.placeholder(tf.float32, [None, len(ModelConfig.categories)], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        if model == 'attentionrnn':
            self.sentiment_attention_rnn_model()
        elif model == 'attention':
            self.sentiment_attention_model()
        elif model == 'bilstm':
            self.sentiment_bilstm_model()
        elif model == 'rnn':
            self.sentiment_rnn_model()
        elif model == 'mlp':
            self.sentiment_mlp_model()
        elif model == 'lr':
            self.sentiment_lr_model()

    def sentiment_attention_model(self):
        '''
            model for target-based model using multi attention mechanism
        '''
        X_length = length(self.X)

        # build embedding layer
        embedding_layer = EmbeddingLayer(vocab_size=ModelConfig.vocab_size, emb_dim=ModelConfig.embedding_dim, trainable=ModelConfig.embedding_trainable)
        # embedded_X shape = [batch_size, seq_length, embedding_dim]
        embedded_X = embedding_layer(padded_input_idx=self.X)

        # build bi-lstm layer as encoder
        bilstm_layer = BiLSTMLayer(hidden_dim=ModelConfig.bilstm_hidden_dim, num_layers=ModelConfig.num_bilstm_layer, dropout_keep=self.keep_prob)
        # bilstm shape = [batch_size, seq_length, 2*bilstm_hidden_dim]
        bilstm_output = bilstm_layer(input_t=embedded_X, seq_length=X_length)

        # build location weighted memory
        #self.memory_layer = PositionMemoryLayer(ModelConfig.seq_length)
        # memory shape = [batch_size, seq_length, 2*bilstm_hidden_dim+1]
        #memory = self.memory_layer(input_t=bilstm_output, offset_concat_t=self.Offset_concat, offset_weight_t=self.Offset_weight)

        # build attention layer
        self.attention_layer = AttentionLayer(input_dim=(2*ModelConfig.bilstm_hidden_dim))
        # attended_output shape = [batch_size, 1, 2*bilstm_hidden_dim+1]
        self.attention_output = self.attention_layer(memory=bilstm_output, seq_length=X_length)

        # add last bilstm output in attention output
        #W_p = tf.get_variable('attention_weights', [(2*ModelConfig.bilstm_hidden_dim+1), 100], initializer=tf.contrib.layers.xavier_initializer(seed=1234))
        #W_x = tf.get_variable('bilstm_weights', [(2*ModelConfig.bilstm_hidden_dim), 100], initializer=tf.contrib.layers.xavier_initializer(seed=1234))
        #last_bilstm_output = last_rnn_output(bilstm_output, X_length) 
        #linear = tf.tensordot(self.attention_output, W_p, axes=[[-1],[0]]) + tf.tensordot(last_bilstm_output, W_x, axes=[[-1],[0]])
        #attend_bilstm_output = tf.tanh(linear)

        # build a dense layer
        #dense_layer = DenseLayer(input_dim=(2*ModelConfig.bilstm_hidden_dim), output_dim=100, activation='relu', name='dense_layer')
        #dense_output = dense_layer(self.attention_output)

        # build output dense layer
        output_layer = DenseLayer(input_dim=(2*ModelConfig.bilstm_hidden_dim), output_dim=len(ModelConfig.categories), name='output_layer')
        # logits shape = [batch_size, category_size]
        self.logits = tf.squeeze(output_layer(self.attention_output))

        # classfication
        # y_pred_cls = [batch_size, 1]
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        # build loss function and optimizer
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        self.loss = tf.reduce_mean(cross_entropy)
        self.optim = tf.train.AdamOptimizer(learning_rate=TrainConfig.learning_rate).minimize(self.loss)

        # evaluation
        # Y = [batch_size, 1]
        correct_pred = tf.equal(tf.argmax(self.Y, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    def sentiment_attention_rnn_model(self):
        '''
            model for target-based model using multi attention mechanism
        '''
        X_length = length(self.X)

        # build embedding layer
        embedding_layer = EmbeddingLayer(vocab_size=ModelConfig.vocab_size, emb_dim=ModelConfig.embedding_dim, trainable=ModelConfig.embedding_trainable)
        # embedded_X shape = [batch_size, seq_length, embedding_dim]
        embedded_X = embedding_layer(padded_input_idx=self.X)

        # build bi-lstm layer as encoder
        bilstm_layer = BiLSTMLayer(hidden_dim=ModelConfig.bilstm_hidden_dim, num_layers=ModelConfig.num_bilstm_layer, dropout_keep=self.keep_prob)
        # bilstm shape = [batch_size, seq_length, 2*bilstm_hidden_dim]
        bilstm_output = bilstm_layer(input_t=embedded_X, seq_length=X_length)

        # build location weighted memory
        memory_layer = PositionMemoryLayer(ModelConfig.seq_length)
        # memory shape = [batch_size, seq_length, 2*bilstm_hidden_dim+1]
        memory = memory_layer(input_t=bilstm_output, offset_concat_t=self.Offset_concat, offset_weight_t=self.Offset_weight)

        # build attention layer
        self.attention_layer = TencentAttentionRnnLayer(hidden_dim=ModelConfig.rnn_hidden_dim, input_dim=(2*ModelConfig.bilstm_hidden_dim+1), cell_type=ModelConfig.rnn_decoder, num_layers=ModelConfig.num_rnn_layer, dropout_keep=self.keep_prob)
        # attended_output shape = [batch_size, 1, rnn_hidden_dim]
        self.attended_output = self.attention_layer(memory=memory, seq_length=X_length)

        # build a dense layer
        # dense_layer = DenseLayer(input_dim=ModelConfig.rnn_hidden_dim, output_dim=50, activation='relu', name='dense_layer')
        # dense_output = dense_layer(self.attended_output)

        # build output dense layer
        output_layer = DenseLayer(input_dim=ModelConfig.rnn_hidden_dim, output_dim=len(ModelConfig.categories), name='output_layer')
        # logits shape = [batch_size, category_size]
        self.logits = tf.squeeze(output_layer(self.attended_output))

        # classfication
        # y_pred_cls = [batch_size, 1]
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        # build loss function and optimizer
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        self.loss = tf.reduce_mean(cross_entropy)
        self.optim = tf.train.AdamOptimizer(learning_rate=TrainConfig.learning_rate).minimize(self.loss)

        # evaluation
        # Y = [batch_size, 1]
        correct_pred = tf.equal(tf.argmax(self.Y, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    def sentiment_bilstm_model(self):
        '''
            model for target-based model using bilstm
        '''
        X_length = length(self.X)

        # build embedding layer
        embedding_layer = EmbeddingLayer(vocab_size=ModelConfig.vocab_size, emb_dim=ModelConfig.embedding_dim, trainable=ModelConfig.embedding_trainable)
        # embedding shape = [batch_size, seq_length, embedding_dim]
        embedded_X = embedding_layer(padded_input_idx=self.X)

        # build bi-lstm layer as encoder
        bilstm_layer = BiLSTMLayer(hidden_dim=ModelConfig.bilstm_hidden_dim, num_layers=ModelConfig.num_bilstm_layer, dropout_keep=self.keep_prob)
        # bilstm shape = [batch_size, seq_length, 2*bilstm_hidden_dim]
        bilstm_output = bilstm_layer(input_t=embedded_X, seq_length=X_length)

        # get the output of last time step
        # last_output shape = [batch_size, 1, 2*bilstm_hidden_dim]
        last_output = last_rnn_output(bilstm_output, X_length)
        #last_output = bilstm_output[:, -1, :]

        # build a dense layer
        dense_layer = DenseLayer(input_dim=(2*ModelConfig.bilstm_hidden_dim), output_dim=100, activation='relu', name='dense_layer')
        dense_output = dense_layer(last_output)

        # build output dense layer
        output_layer = DenseLayer(input_dim=100, output_dim=len(ModelConfig.categories), name='output_layer')
        # logits shape = [batch_size, category_size]
        self.logits = tf.squeeze(output_layer(dense_output))

        # classfication
        # y_pred_cls = [batch_size, 1]
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        # build loss function and optimizer
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        self.loss = tf.reduce_mean(cross_entropy)
        self.optim = tf.train.AdamOptimizer(learning_rate=TrainConfig.learning_rate).minimize(self.loss)

        # evaluation
        # Y = [batch_size, 1]
        correct_pred = tf.equal(tf.argmax(self.Y, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    def sentiment_rnn_model(self):
        '''
            model for target-based model using bilstm
        '''
        rnn_type = 'gru'

        self.X_length = length(self.X)
        # build embedding layer
        embedding_layer = EmbeddingLayer(vocab_size=ModelConfig.vocab_size, emb_dim=ModelConfig.embedding_dim, trainable=ModelConfig.embedding_trainable, name="rnn_emb_layer")
        # embedding shape = [batch_size, seq_length, embedding_dim]
        self.embedded_X = embedding_layer(padded_input_idx=self.X)

        # build rnn (gru or lstm) layer as encoder
        rnn_layer = RNNLayer(hidden_dim=100, cell_type=rnn_type, num_layers=1, dropout_keep=self.keep_prob)
        # rnn_output shape = [batch_size, seq_length, 100]
        rnn_output = rnn_layer(input_t=self.embedded_X, seq_length=self.X_length)

        # get the output of last time step
        # last_output shape = [batch_size, 1, 100]
        last_output = last_rnn_output(rnn_output, self.X_length)
        #last_output = rnn_output[:, -1, :]

        # build output dense layer
        output_layer = DenseLayer(input_dim=100, output_dim=len(ModelConfig.categories), name='output_layer')
        # logits shape = [batch_size, category_size]
        self.logits = tf.squeeze(output_layer(last_output))

        # classfication
        # y_pred_cls = [batch_size, 1]
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
        #self.y_pred_cls = tf.argmax(self.logits, 1)

        # build loss function and optimizer
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        self.loss = tf.reduce_mean(cross_entropy)
        self.optim = tf.train.AdamOptimizer(learning_rate=TrainConfig.learning_rate).minimize(self.loss)

        # evaluation
        # Y = [batch_size, 1]
        correct_pred = tf.equal(tf.argmax(self.Y, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def sentiment_mlp_model(self):
        '''
            model for target-based model using mlp
        '''
        # build embedding layer
        embedding_layer = EmbeddingLayer(vocab_size=ModelConfig.vocab_size, emb_dim=ModelConfig.embedding_dim, trainable=ModelConfig.embedding_trainable, name='mlp_emb_layer')
        # embedding shape = [batch_size, seq_length, embedding_dim]
        self.embedded_X = embedding_layer(padded_input_idx=self.X)

        # build dense layer
        # transform shape from [batch_size, seq_length, emb_dim] to [batch_size, seq_length*emb_dim]
        flat_embedding = tf.reshape(self.embedded_X, [-1, ModelConfig.embedding_dim*ModelConfig.seq_length])
        #dense_layer = DenseLayer(input_dim=(ModelConfig.embedding_dim*ModelConfig.seq_length), output_dim=100, activation='relu', name='mlp_dense_layer')
        # logits shape = [batch_size, 100]
        #dense_ouput = dense_layer(flat_embedding)

        # build a dense layer
        dense_layer = DenseLayer(input_dim=(ModelConfig.embedding_dim*ModelConfig.seq_length), output_dim=100, activation='tanh', name='hidden_layer')
        dense_output = dense_layer(flat_embedding)

        # build output dense layer
        output_layer = DenseLayer(input_dim=100, output_dim=len(ModelConfig.categories), name='output_layer')
        # logits shape = [batch_size, category_size]
        self.logits = tf.squeeze(output_layer(dense_output))

        # classfication
        # y_pred_cls = [batch_size, 1]
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
        #self.y_pred_cls = tf.argmax(self.logits, 1)

        # build loss function and optimizer
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        self.loss = tf.reduce_mean(cross_entropy)
        self.optim = tf.train.AdamOptimizer(learning_rate=TrainConfig.learning_rate).minimize(self.loss)

        # evaluation
        # Y = [batch_size, 1]
        correct_pred = tf.equal(tf.argmax(self.Y, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def sentiment_lr_model(self):
        '''
            model for target-based model using mlp
        '''
        # build embedding layer
        embedding_layer = EmbeddingLayer(vocab_size=ModelConfig.vocab_size, emb_dim=ModelConfig.embedding_dim, trainable=ModelConfig.embedding_trainable, name='mlp_emb_layer')
        # embedding shape = [batch_size, seq_length, embedding_dim]
        self.embedded_X = embedding_layer(padded_input_idx=self.X)

        # build dense layer
        # transform shape from [batch_size, seq_length, emb_dim] to [batch_size, seq_length*emb_dim]
        flat_embedding = tf.reshape(self.embedded_X, [-1, ModelConfig.embedding_dim*ModelConfig.seq_length])
        #dense_layer = DenseLayer(input_dim=(ModelConfig.embedding_dim*ModelConfig.seq_length), output_dim=100, activation='relu', name='mlp_dense_layer')
        # logits shape = [batch_size, 100]
        #dense_ouput = dense_layer(flat_embedding)

        # build output dense layer
        output_layer = DenseLayer(input_dim=(ModelConfig.embedding_dim*ModelConfig.seq_length), output_dim=len(ModelConfig.categories), name='output_layer')
        # logits shape = [batch_size, category_size]
        self.logits = tf.squeeze(output_layer(flat_embedding))

        # classfication
        # y_pred_cls = [batch_size, 1]
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
        #self.y_pred_cls = tf.argmax(self.logits, 1)

        # build loss function and optimizer
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        self.loss = tf.reduce_mean(cross_entropy)
        self.optim = tf.train.AdamOptimizer(learning_rate=TrainConfig.learning_rate).minimize(self.loss)

        # evaluation
        # Y = [batch_size, 1]
        correct_pred = tf.equal(tf.argmax(self.Y, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# calculate actual length of each sequence
def length(sequence):
    #used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    used = tf.sign(tf.abs(sequence))
    leng = tf.reduce_sum(used, axis=1)
    leng = tf.cast(leng, tf.int32)
    return leng

# get last valid timestep output of RNN
def last_rnn_output(output, length):
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    output = tf.expand_dims(relevant, 1)
    return output