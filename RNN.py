import numpy as np

np.random.seed(1234)
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Embedding, Flatten, Input, merge, Reshape, Merge, Dropout, Dense, SimpleRNN, LSTM
from keras.optimizers import Adam, SGD
from keras.regularizers import l2


import theano
import DataParser
import Evaluation
import numpy as np
from datetime import datetime as dt
import random
import sys


class RNN():
    def __init__(self, uNum, vNum, WINDOW, latent_dim,Lambda, enableCycle=False, opt=Adam()):
        self.WINDOW = WINDOW
        self.enableCycle = enableCycle

        def identity_loss(y_true, y_pred):
            return K.mean(y_pred - 0 * y_true)

        def rnn_triplet_loss(X):

            if enableCycle:
                positive_item_latent, negative_item_latent, user_latent, pos_user_temporal_latent, neg_user_temporal_latent, pos_user_backward_temporal_latent, neg_user_backward_temporal_laten = X
                reg = Lambda * (
                K.sum(user_latent ** 2, axis=-1, keepdims=True) + K.sum(positive_item_latent ** 2, axis=-1,
                                                                        keepdims=True) + K.sum(
                    negative_item_latent ** 2, axis=-1, keepdims=True) + K.sum(pos_user_temporal_latent ** 2,
                                                                               axis=-1) + K.sum(
                    neg_user_temporal_latent ** 2, axis=-1) + K.sum(pos_user_backward_temporal_latent ** 2, axis = -1) + K.sum(neg_user_backward_temporal_laten ** 2 , axis=-1))

                loss = (1 - K.sigmoid(
                    K.sum((pos_user_temporal_latent + pos_user_backward_temporal_latent + user_latent) * positive_item_latent, axis=-1, keepdims=True) -
                    K.sum((neg_user_temporal_latent + neg_user_backward_temporal_laten + user_latent) * negative_item_latent, axis=-1,
                          keepdims=True))) + reg
            else:
                positive_item_latent, negative_item_latent, user_latent, pos_user_temporal_latent, neg_user_temporal_latent = X
                reg = Lambda * (
                K.sum(user_latent ** 2, axis=-1, keepdims=True) + K.sum(positive_item_latent ** 2, axis=-1,
                                                                        keepdims=True) + K.sum(
                    negative_item_latent ** 2, axis=-1, keepdims=True) + K.sum(pos_user_temporal_latent ** 2,
                                                                               axis=-1) + K.sum(
                    neg_user_temporal_latent ** 2, axis=-1))

                loss = (1 - K.sigmoid(
                    K.sum((pos_user_temporal_latent + user_latent) * positive_item_latent, axis=-1, keepdims=True) -
                    K.sum((neg_user_temporal_latent + user_latent) * negative_item_latent, axis=-1, keepdims=True))) + reg



            #   TODO if user_temporal_latent[0][0] == [0,,,,,0] ignore, else sum all loss



            return loss

        # Shared embedding layer for positive and negative items

        item_embedding_layer = Embedding(
            vNum + 1, latent_dim, name='item_embedding', input_length=WINDOW)

        rnnModel = SimpleRNN(latent_dim, return_sequences=True)
        backword_rnnModel = SimpleRNN(latent_dim, return_sequences=True, go_backwards=True)

        #     user_temporal_embedding_layer = lstm
        self.user_temporal_embedding_layer = Sequential(name="user_temporal_embedding")
        self.user_temporal_embedding_layer.add(item_embedding_layer)
        self.user_temporal_embedding_layer.add(rnnModel)

        if enableCycle:
            self.user_backward_temporal_embedding_layer = Sequential(name="user_backward_temporal_embedding")
            self.user_backward_temporal_embedding_layer.add(item_embedding_layer)
            self.user_backward_temporal_embedding_layer.add(backword_rnnModel)

        positive_item_input = Input((WINDOW,), name='positive_item_input')
        negative_item_input = Input((WINDOW,), name='negative_item_input')
        user_input = Input((WINDOW,), name='user_input')

        positive_item_embedding = Flatten()(item_embedding_layer(positive_item_input))
        negative_item_embedding = Flatten()(item_embedding_layer(negative_item_input))

        user_embedding = Flatten()(Embedding(
            uNum, latent_dim, name='user_embedding', input_length=1)(
            user_input))

        pos_user_temporal_embedding = Flatten()(self.user_temporal_embedding_layer(positive_item_input))
        neg_user_temporal_embedding = Flatten()(self.user_temporal_embedding_layer(negative_item_input))

        if enableCycle:
            pos_user_backward_temporal_embedding = Flatten()(
                self.user_backward_temporal_embedding_layer(positive_item_input))
            neg_user_backward_temporal_embedding = Flatten()(
                self.user_backward_temporal_embedding_layer(negative_item_input))

        if enableCycle:
            loss = merge(
                [positive_item_embedding, negative_item_embedding, user_embedding, pos_user_temporal_embedding,
                 neg_user_temporal_embedding, pos_user_backward_temporal_embedding,
                 neg_user_backward_temporal_embedding],
                mode=rnn_triplet_loss,
                name='loss',
                output_shape=(WINDOW,))

        else:
            loss = merge(
                [positive_item_embedding, negative_item_embedding, user_embedding, pos_user_temporal_embedding,
                 neg_user_temporal_embedding],
                mode=rnn_triplet_loss,
                name='loss',
                output_shape=(WINDOW,))

        self.model = Model(
            input=[positive_item_input, negative_item_input, user_input],
            output=[loss])
        self.model.compile(loss=identity_loss, optimizer=opt)

    def predict(self, uid, pids, t):

        visited = self.tuv[t][uid].tocoo().col

        sequence = np.zeros(WINDOW)
        sequence[t] = visited[-1]

        for t2 in range(WINDOW):
            if t == t2:
                continue

            visited2 = tuv[t2][uid].tocoo().col
            if (len(visited2) > 0):
                sequence[t2] = random.choice(visited2)


        user_vector = self.model.get_layer('user_embedding').get_weights()[0][uid]
        item_matrix = self.model.get_layer('item_embedding').get_weights()[0][pids]
        user_temporal_vector = self.user_temporal_embedding_layer.predict(sequence.reshape(1, self.WINDOW))[0][
            t]

        # TODO user_backward_temporal_vector

        scores = (np.dot(user_vector + user_temporal_vector,
                         item_matrix.T))

        return scores

    def setTUV(self, tuv):
        self.tuv = tuv

if __name__ == "__main__":

    start_time = dt.now()
    theano.config.openmp = True
    OMP_NUM_THREADS = 4

    # params
    latent_dim = 10
    mode = "day"
    WINDOW = 7

    Lambda = 0.001

    # df, df_test, uNum, vNum = DataParser.readCheckinData('../../datasets/gowallaTrain', '../../datasets/gowallaTest')
    df, df_test, uNum, vNum = DataParser.readCheckinData(sys.argv[1], sys.argv[2])
    # convert to sparse matrix for each time window
    tuv = DataParser.sparseTUV(df, mode, WINDOW, uNum, vNum)
    tuv_test = DataParser.sparseTUV(df_test, mode, WINDOW, uNum, vNum)

    # rnn_pair_u, rnn_pair_i, rnn_pair_j, rnn_last_pair = DataParser.generate_rnn_pairs(tuv, uNum, vNum, WINDOW)

    rnn_last_pair = []


    num_samples = 0
    for t in range(WINDOW):
        num_samples = num_samples + tuv[t].nonzero()[0].shape[0]


    def generate(tuv):

        rnn_pair_u, rnn_pair_i, rnn_pair_j, = [], [], []

        for u in range(uNum):
            for t in range(WINDOW):
                visited = tuv[t][u].tocoo().col

                for i in visited:
                    pos = np.zeros(WINDOW)
                    neg = np.zeros(WINDOW)
                    pos[t] = i
                    neg[t] = DataParser.sampleNeg(visited, vNum)

                    for t2 in range(WINDOW):
                        if t == t2:
                            continue

                        visited2 = tuv[t2][u].tocoo().col
                        if (len(visited2) > 0):
                            pos[t2] = random.choice(visited2)
                            neg[t2] = DataParser.sampleNeg(visited2, vNum)

                    u_ = np.zeros(WINDOW)
                    u_.fill(u)
                    rnn_pair_u.append(u_)
                    rnn_pair_i.append(pos)
                    rnn_pair_j.append(neg)

            X = {
                'positive_item_input': np.array(rnn_pair_i),
                'negative_item_input': np.array(rnn_pair_j),
                'user_input': np.array(rnn_pair_u),
            }

            yield (X, np.ones(len(rnn_pair_u)))

    rnn = RNN(uNum, vNum, WINDOW, latent_dim,Lambda)
    rnn.setTUV(tuv)

    rnn.model.fit_generator( generate(tuv),
               samples_per_epoch=num_samples,
               nb_epoch=100,
               verbose=2)


    print(Evaluation.eval(tuv_test, rnn, "rnn", WINDOW, vNum))
    elapsed_time = dt.now() - start_time
    print("Time Spent:", elapsed_time)



