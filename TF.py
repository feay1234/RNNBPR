import numpy as np

np.random.seed(1234)
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Embedding, Reshape, Merge
from keras.optimizers import Adam, SGD
from keras.regularizers import l2


import theano
import DataParser
import Evaluation
import numpy as np
from datetime import datetime as dt
import sys



class TF(Sequential):
    def __init__(self, uNum, vNum, latent_dim, WINDOW, Lambda, **kwargs):
        def dot3D(x):
            P, Q, T = x;
            return K.sum(P * Q * T, axis=-1, keepdims=True);

        P = Sequential()
        P.add(Embedding(uNum, latent_dim, input_length=1, W_regularizer=l2(Lambda)))
        P.add(Reshape((latent_dim,)))
        Q = Sequential()
        Q.add(Embedding(vNum, latent_dim, input_length=1, W_regularizer=l2(Lambda)))
        Q.add(Reshape((latent_dim,)))
        T = Sequential()
        T.add(Embedding(WINDOW, latent_dim, input_length=1, W_regularizer=l2(Lambda)))
        T.add(Reshape((latent_dim,)))

        super(TF, self).__init__(**kwargs)
        self.add(Merge([P, Q, T], mode=dot3D, output_shape=[1, ]))


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
    tuv_test = DataParser.sparseTUV(df_test, mode, WINDOW, uNum, vNum)

    tf = TF(uNum, vNum, WINDOW, latent_dim, Lambda)
    tf.compile(loss='mse', optimizer=Adam())

    if mode == "day":
        tf.fit([df.uid.values, df.vid.values, df.day.values], np.ones(len(df.uid.values)), verbose=1, nb_epoch=100, batch_size=100000)
    elif mode == "hour":
        tf.fit([df.uid.values, df.vid.values, df.hour.values], np.ones(len(df.uid.values)), verbose=1, nb_epoch=100, batch_size=100000)


    print(Evaluation.eval(tuv_test, tf, "tf", WINDOW, vNum))
    elapsed_time = dt.now() - start_time
    print("Time Spent:", elapsed_time)
