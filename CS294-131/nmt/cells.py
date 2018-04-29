import tensorflow as tf


class GRUCellWithBahdanauAttention(tf.contrib.rnn.GRUCell):

    def call(self, inputs, state):
        return inputs, state
        ### YOUR CODE HERE
        ### Make sure all variable definitions are within the scope!
        raise NotImplementedError("Need to implement the GRU cell \
                                   with Bahdanau-style attention.")
        ### END YOUR CODE




class GRUCellWithLuongAttention(tf.contrib.rnn.GRUCell):

    def call(self, inputs, state):
        return inputs, state
        ### YOUR CODE HERE
        ### Make sure all variable definitions are within the scope!
        raise NotImplementedError("Need to implement the GRU cell \
                                   with Luong-style attention.")
        ### END YOUR CODE

