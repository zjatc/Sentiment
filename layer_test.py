import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from layers import *

def main():
    tf.set_random_seed(1)

    # input data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    test_data = mnist.test.images[:128].reshape((-1, 28, 28))
    test_label = mnist.test.labels[:128]

    # hyperparameters configuration
    learning_rate = 0.001
    batch_size = 128
    training_iters = 100000

    embedding_size = 28
    seq_length = 28
    rnn_hidden_dim = 100
    n_classes = 10

    x = tf.placeholder(tf.float32, [None, seq_length, embedding_size])
    y = tf.placeholder(tf.float32, [None, n_classes])

    pred = test(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        step = 0
        while step * batch_size < training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, seq_length, embedding_size])
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys,
            })
            if step % 20 == 0:
                print("train: ", sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
                print("test: ", sess.run(accuracy, feed_dict={
                x: test_data,
                y: test_label,
            }))
            step += 1

   
def test(X):

    #X = tf.reshape(X, [-1, 28])
    #X_in = tf.matmul(X, weights['in']) + biases['in']
    #X_in = tf.reshape(X_in, [-1, 28, 100])


    # build rnn (gru or lstm) layer as encoder
    rnn_layer = RNNLayer(hidden_dim=100, cell_type='gru', num_layers=1, dropout_keep=0.8)
    # rnn_output shape = [batch_size, seq_length, 100]
    rnn_output = rnn_layer(input_t=X)

    # 把 outputs 变成 列表 [(batch, outputs)..] * steps
    #outputs = tf.unstack(tf.transpose(rnn_output, [1,0,2]))
    outputs = rnn_output[:, -1, :]

    # build output dense layer
    output_layer = DenseLayer(input_dim=100, output_dim=10, name='output_layer')
    # logits shape = [batch_size, category_size]
    results = tf.squeeze(output_layer(outputs))

    #results = tf.nn.softmax(logits)

    #results = tf.matmul(outputs, weights['out']) + biases['out']    #选取最后一个 output
    return results


if __name__ == '__main__':
    main()