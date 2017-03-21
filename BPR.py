import numpy as np
np.random.seed(1234)
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Embedding, Flatten, Input, merge
from keras.optimizers import Adam, SGD

import theano
import DataParser
import Evaluation
import numpy as np
from datetime import datetime as dt
import sys

class BPR():
    def __init__(self, uNum, vNum, latent_dim, Lambda, mode, opt=Adam()):

        def identity_loss(y_true, y_pred):
            return K.mean(y_pred - 0 * y_true)

        def bpr_triplet_loss(X):
            positive_item_latent, negative_item_latent, user_latent = X

            reg = Lambda * (K.sum(user_latent ** 2, axis=-1, keepdims=True) + K.sum(positive_item_latent ** 2, axis=-1,
                                                                                   keepdims=True) + K.sum(
                negative_item_latent ** 2, axis=-1, keepdims=True))

            if mode == 1:
                loss = 1 - K.log(K.sigmoid(
                    K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
                    K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True)))

            elif mode == 2:
                loss = 1 - K.log(K.sigmoid(
                    K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
                    K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))) - reg

            elif mode == 3:
                loss = 1 - K.log(K.sigmoid(
                    K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
                    K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))) + reg

            elif mode == 4:
                loss = 1 - K.sigmoid(
                    K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
                    K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

            elif mode == 5:
                loss = 1 - K.sigmoid(
                    K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
                    K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True)) + reg

            elif mode == 6:
                loss = 1 - K.sigmoid(
                    K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
                    K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True)) - reg

            return loss

        # Shared embedding layer for positive and negative items
        item_embedding_layer = Embedding(
            vNum + 1, latent_dim, name='item_embedding', input_length=1)

        positive_item_input = Input((1,), name='positive_item_input')
        negative_item_input = Input((1,), name='negative_item_input')
        user_input = Input((1,), name='user_input')

        positive_item_embedding = Flatten()(item_embedding_layer(positive_item_input))
        negative_item_embedding = Flatten()(item_embedding_layer(negative_item_input))

        user_embedding = Flatten()(Embedding(
            uNum + 1, latent_dim, name='user_embedding', input_length=1)(
            user_input))

        loss = merge(
            [positive_item_embedding, negative_item_embedding, user_embedding],
            mode=bpr_triplet_loss,
            name='loss',
            output_shape=(1,))

        self.model = Model(
            input=[positive_item_input, negative_item_input, user_input],
            output=[loss])
        #     model.compile(loss=identity_loss, optimizer=SGD())
        self.model.compile(loss=identity_loss, optimizer=opt)

    def predict(self, uid, vids):
        user_vector = self.model.get_layer('user_embedding').get_weights()[0][uid]
        item_matrix = self.model.get_layer('item_embedding').get_weights()[0][vids]

        scores = (np.dot(user_vector,
                         item_matrix.T))
        return scores


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
    uv = DataParser.sparseUV(df,uNum, vNum)
    tuv_test = DataParser.sparseTUV(df_test, mode, WINDOW, uNum, vNum)
    # pos/neg samples
    # pos_samples, neg_samples = DataParser.generate_train_pairs(tuv)
    pair_u, pair_i, pair_j = DataParser.generate_bpr_pairs(uv, uNum, vNum)

    bpr = BPR(uNum, vNum, latent_dim, Lambda, int(sys.argv[3]))
    # bpr = BPR(uNum, vNum, latent_dim, Lambda, 4)

    X = {
        'positive_item_input': pair_i,
        'negative_item_input': pair_j,
        'user_input': pair_u
    }

    bpr.model.fit(X,
                np.ones(pair_i.shape[0]),
                batch_size=100000,
                nb_epoch=100,
                verbose=1,
                shuffle=True)

    print(Evaluation.eval(tuv_test, bpr, "bpr", WINDOW, vNum))
    elapsed_time = dt.now() - start_time
    print("Time :", elapsed_time)
