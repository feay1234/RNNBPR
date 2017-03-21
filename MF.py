import numpy as np

np.random.seed(1234)
from keras.models import Model, Sequential
from keras.layers import Embedding, Reshape, Merge
from keras.optimizers import Adam
from keras.regularizers import l2

import theano
import DataParser
import Evaluation
import numpy as np
from datetime import datetime as dt
import sys

class MF(Sequential):

    def __init__(self, uNum, vNum, latent_num, Lambda, **kwargs):
        P = Sequential()
        P.add(Embedding(uNum, latent_num, input_length=1, W_regularizer=l2(Lambda)))
        P.add(Reshape((latent_num,)))
        Q = Sequential()
        Q.add(Embedding(vNum, latent_num, input_length=1, W_regularizer=l2(Lambda)))
        Q.add(Reshape((latent_num,)))
        super(MF, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='dot', dot_axes=1))



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

    mf = MF(uNum, vNum, latent_dim, Lambda)
    mf.compile(loss='mse', optimizer=Adam())

    mf.fit([df.uid.values, df.vid.values], np.ones(len(df.uid.values)), verbose=1, nb_epoch=100, batch_size=100000)

    print(Evaluation.eval(tuv_test, mf, "mf", WINDOW, vNum))
    elapsed_time = dt.now() - start_time
    print("Time Spent:", elapsed_time)
