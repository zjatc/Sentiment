# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.keras as kr


class EmbeddingLayer(object):

    def __init__(self, vocab_size, emb_dim, pretrain_emb=None, trainable=True, name='embedding_layer'):
        '''
            generate embedding lookup-tabel, if the pretrained embedding will be used, pass with 'pretrain_embedding' parameter
        '''
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.name = name
        self.pretrain_emb = pretrain_emb
        self.trainable = trainable

        # generate random embeddings if no pre-trained embedding provided
        if self.pretrain_emb is None:
            self.embeddings = tf.get_variable(self.name+'_emb', [self.vocab_size, self.emb_dim], initializer = tf.contrib.layers.xavier_initializer(seed=1234), 
                                                trainable=self.trainable)
        else:
            self.embeddings = tf.get_variable(self.name+'_emb', initializer=self.pretrain_emb, trainable=self.trainable)
        self.embedding_name = self.name + '_emb'

    def __call__(self, padded_input_idx):
        '''
            return the embeddings of the given input indices
        '''
        output_embeddings = tf.nn.embedding_lookup(params=self.embeddings, ids=padded_input_idx)
        variable_summaries("emb_ouput_", output_embeddings)

        return output_embeddings


class RNNLayer(object):

    def __init__(self, hidden_dim, cell_type='gru', num_layers=1, dropout_keep=1.0, output_state=False, name='rnn_layer'):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_keep = dropout_keep
        self.output_state = output_state
        if cell_type == 'gru':
            self.name = 'gru_layer'
            self.cell = tf.contrib.rnn.GRUCell(self.hidden_dim, activation=tf.sigmoid)
        elif cell_type == 'lstm':
            self.name = 'lstm_layer'
            self.cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=Ture)
        elif cell_type == 'rnn':
            self.name = 'basic_rnn_layer'
            self.cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_dim)

        assert 1.0 <= num_layers


    def __call__(self, input_t, seq_length):
        self.input = input_t
        variable_summaries(self.name+'_input_', self.input)

        self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.dropout_keep)
        if self.num_layers > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * self.num_layers, state_is_tuple=True)

        self.outputs, states = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.input, dtype=tf.float32, sequence_length=seq_length)

        tf.summary.histogram(self.name+'_output', self.outputs)
        if self.output_state:
            return self.outputs, states
        else:
            return self.outputs


class BiLSTMLayer(object):

    def __init__(self, hidden_dim, num_layers=1, dropout_keep=1.0, output_state=False, name='bilstm_layer'):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_keep = dropout_keep
        self.output_state = output_state
        self.name = name

        assert 1 <= num_layers
        self.lstm_fw_cell = []
        self.lstm_bw_cell = []
        for i in range(self.num_layers):
            self.lstm_fw_cell.append(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True), output_keep_prob=self.dropout_keep))
            self.lstm_bw_cell.append(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True), output_keep_prob=self.dropout_keep))
        
        self.lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(self.lstm_fw_cell)
        self.lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(self.lstm_bw_cell)

    def __call__(self, input_t, seq_length):
        self.input = input_t
        output_fw_bw, cell_states_fw_bw = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell, self.lstm_bw_cell, self.input, sequence_length=seq_length, dtype=tf.float32, scope=self.name)
        
        self.outputs = tf.concat(values=output_fw_bw, axis=2)
        if self.output_state:
            return self.outputs, cell_states_forward_backward
        else:
            return self.outputs

# TODO need to use acutal sequence length as the max_seq_length
class PositionMemoryLayer(object):

    def __init__(self, seq_length):
        self.seq_length = seq_length

    def __call__(self, input_t, offset_concat_t, offset_weight_t):
        self.input = input_t
        #self.mask = tf.sign(tf.reduce_max(tf.abs(self.input), 2, keep_dims=True))
        #self.seq_length = tf.expand_dims(tf.expand_dims(seq_length,1), 1)
        #self.seq_length = tf.cast(tf.tile(self.seq_length, [1,self.max_seq_length,1]), tf.float32)

        # u_t = (t - tau) / t_max
        # padded 0.0 to end of each offset's sequences, so that the concated offset to the memory vector will be 0.0
        self.offset_concat = tf.reshape(offset_concat_t, shape=[-1, self.seq_length,1])
        self.offset_concat = tf.divide(self.offset_concat, self.seq_length)
        # padded max_seq_length to end of each offset's sequences, so that the weight of padded token will be 0.0
        self.offset_weight = tf.reshape(offset_weight_t, shape=[-1, self.seq_length,1])
        self.offset_weight = tf.divide(self.offset_weight, self.seq_length)
        #self.offset_weight = tf.reshape(offset_concat_t, shape=[-1, self.max_seq_length,1])
        #self.offset_weight = tf.divide(self.offset_weight, self.seq_length)
        # w_t = 1 - |u_t| = 1- |t-tau|/t_max
        self.pos_weight = tf.subtract(1.0, tf.abs(self.offset_weight))
        #self.pos_weight *= self.mask
        # for all padded tokens, will get a zero vector
        # m_t = (w_t * m_t^*, u_t)
        self.outputs = tf.concat(values=[self.pos_weight * self.input, self.offset_concat], axis=2)
        return self.outputs

# ref. "Attention is all you need", attention(Q, K, V) = softmax(QK^t/sqr(d_k))V
class AttentionLayer(object):

    def __init__(self, input_dim, name='attention_layer'):
        self.input_dim = input_dim

        self.W_t = tf.get_variable(name+'_t_weights', [self.input_dim, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1234))
        self.b_t = tf.get_variable(name+'_t_bias', [1], initializer=tf.contrib.layers.xavier_initializer(seed=1234))

    def __call__(self, memory, seq_length):
        self.memory = memory

        batch_size = tf.shape(self.memory)[0]
        max_seq_length = tf.shape(self.memory)[1]
        acutal_seq_length = tf.expand_dims(seq_length, 1)
        self.padded_exp_sum = tf.cast(tf.expand_dims(max_seq_length - acutal_seq_length, 1), tf.float32) 
        self.mask = tf.sign(tf.reduce_max(tf.abs(self.memory), 2, keep_dims=True))

        # g_t = W_t*m + b_t
        self.g_t = tf.tensordot(self.memory, self.W_t, axes=[[-1],[0]]) + self.b_t
        # attention a_t = softmax(g_t)
        # here needs use the actual sequence length in softmax, not the padded one
        self.a_t = tf.exp(self.g_t) / (tf.reduce_sum(tf.exp(self.g_t), 1, keep_dims=True) - self.padded_exp_sum)
        self.a_t *= self.mask
        # self.a_t = tf.nn.softmax(logits=self.g_t, dim=1)
        # context attened_output = a_t^T * m
        self.attened_output = tf.matmul(tf.transpose(self.a_t, perm=[0,2,1]), self.memory)

        return self.attened_output

'''
ref. "Recurrent Attention Network on Memory for Aspect Sentiment Analysis"
'''
class TencentAttentionRnnLayer(object):

    def __init__(self, hidden_dim, input_dim, cell_type='gru', num_layers=1, dropout_keep=1.0, output_state=False, name='attended_layer'):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dropout_keep = dropout_keep
        self.output_state = output_state
        self.cell_type = cell_type

        self.rnn = RNNLayer(hidden_dim=self.hidden_dim, cell_type=self.cell_type, dropout_keep=self.dropout_keep, output_state=self.output_state, name='attended_gru')
        
        self.W_t = tf.get_variable(name+'_t_weights', [self.hidden_dim+self.input_dim, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1234))
        self.b_t = tf.get_variable(name+'_t_bias', [1], initializer=tf.contrib.layers.xavier_initializer(seed=1234))

        assert 1.0 <= num_layers


    def __call__(self, memory, seq_length):
        self.memory = memory
        self.state = tf.zeros(shape=[1, 1, self.hidden_dim], dtype=tf.float32)

        batch_size = tf.shape(self.memory)[0]
        max_seq_length = tf.shape(self.memory)[1]
        acutal_seq_length = tf.expand_dims(seq_length, 1)
        self.padded_exp_sum = tf.cast(tf.expand_dims(max_seq_length - acutal_seq_length, 1), tf.float32) 
        self.mask = tf.sign(tf.reduce_max(tf.abs(self.memory), 2, keep_dims=True))

        def calcul_attention(iter_num):
            if iter_num==0:
                self.tile_shape = tf.stack([batch_size, max_seq_length, 1])
            else:
                self.tile_shape = tf.stack([1, max_seq_length, 1])
            # g_t = W_t(m, e_t-1) + b_t
            self.mem_state = tf.concat([self.memory, tf.tile(self.state, self.tile_shape)], -1)
            self.g_t = tf.tensordot(self.mem_state, self.W_t, axes=[[-1],[0]]) + self.b_t
            # attention a_t = softmax(g_t)
            # here needs use the actual sequence length in softmax, not the padded one
            self.a_t = tf.exp(self.g_t) / (tf.reduce_sum(tf.exp(self.g_t), 1, keep_dims=True) - self.padded_exp_sum)
            self.a_t *= self.mask
            # self.a_t = tf.nn.softmax(logits=self.g_t, dim=1)
            # context i_t = a_t^T * m
            self.i_t = tf.matmul(tf.transpose(self.a_t, perm=[0,2,1]), self.memory)

        rnn_input_length = seq_length / seq_length
        for i in range(self.num_layers):
            # attention
            calcul_attention(i);
            # rnn
            if self.output_state:
                self.outputs, states = self.rnn(input_t=self.i_t, seq_length=rnn_input_length)
            else:
                self.outputs = self.rnn(input_t=self.i_t, seq_length=rnn_input_length)
            self.state = self.outputs


        if self.output_state:
            return self.outputs, states
        else:
            return self.outputs


class DenseLayer(object):

    def __init__(self, input_dim, output_dim, use_bias=True, activation='linear', initial='uniform', name='dense_layer'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.name = name
        if activation == 'linear':
            self.activation = None
        elif activation == 'tanh':
            self.activation = tf.tanh
        elif activation == 'sigmoid':
            self.activation = tf.sigmoid
        elif activation == 'softmax':
            self.activation = tf.nn.softmax
        elif activation == 'relu':
            self.activation = tf.nn.relu
        elif activation is not None:
            raise Exception('Do not support activation function ' % activation)

        # initialize the weights and bias
        if initial == 'uniform':
            self.weights = tf.get_variable(name+'_weights', [self.input_dim, self.output_dim], initializer=tf.contrib.layers.xavier_initializer(seed=1234))
            self.bias = tf.get_variable(name+'_bias', [self.output_dim], initializer=tf.contrib.layers.xavier_initializer(seed=1234))
        elif initial == 'normal':
            self.weights = tf.get_variable(name+'_weights', [self.input_dim, self.output_dim], initializer=tf.truncated_normal_initializer(seed=1234))
            self.bias = tf.get_variable(name+'_bias', [self.output_dim], initializer=tf.truncated_normal_initializer(seed=1234))
        elif initial == 'zero':
            self.weights = tf.get_variable(name+'_weights', [self.input_dim, self.output_dim], initializer=tf.constant_initializer(0.0))
            self.bias = tf.get_variable(name+'_bias', [self.output_dim], initializer=tf.constant_initializer(0.0))
        elif initial is not None:
            raise Exception('Do not support initializer function ' % initial)

        if self.use_bias:
            self.params = [self.weights, self.bias]
        else:
            self.params = [self.weights]


    def __call__(self, input_t):

        variable_summaries("dense_weight_", self.weights)
        variable_summaries("dense_bias_", self.bias)

        self.input = input_t
        self.linear = tf.tensordot(self.input, self.weights, axes=[[-1],[0]])
        if self.use_bias:
            self.linear += self.bias

        if self.activation == None:
            self.output = self.linear
        else:
            self.output = self.activation(self.linear)

        tf.summary.histogram('dense_layer_output', self.output)
        return self.output


def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name+'summaries'):
        # calculate means of parameters, use tf.summary.scaler to record
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # calculate standard deviation
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # use tf.summary.scalar to rcord stddev, max, min values
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

        # use histogram to record the distribution of variables
        tf.summary.histogram('histogram', var)