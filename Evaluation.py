import pandas as pd
import theano
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
import letor_metrics as lm
import DataParser
import numpy as np
from datetime import datetime as dt
import sys

# start_time = dt.now()
# theano.config.openmp = True
# OMP_NUM_THREADS = 4
#
# # params
# latent_dim = 10
#
# mode = "day"
# WINDOW = 7
#
# print("Start Time:", dt.now())
#
# df, df_test, uNum, vNum = DataParser.readCheckinData('../datasets/gowallaTrain', '../datasets/gowallaTest')
# # df, df_test, uNum, vNum = DataParser.readCheckinData(sys.argv[2], sys.argv[3])
# # convert to sparse matrix for each time window
# tuv = DataParser.sparseDataset(df, mode, WINDOW, uNum, vNum)
# tuv_test = DataParser.sparseDataset(df_test, mode, WINDOW, uNum, vNum)
# # pos/neg samples
# # pos_samples, neg_samples = DataParser.generate_train_pairs(tuv)
#
# u_input = []
# for u in range(uNum):
#     ut = []
#     for i in range(WINDOW):
#         ut.append(u)
#     u_input.append(ut)
# u_input = np.array(u_input)
#
#
#
# modelName = "bpr"
# # modelName =sys.argv[1]
# # if modelName == "bpr":
# #     m = Model.BPR(uNum, vNum, WINDOW, latent_dim, int(sys.argv[4]))
# # elif modelName == "rnn":
# #     m = Model.RNN(uNum, vNum, WINDOW, latent_dim)
# # m = Model.RNN(uNum, vNum, WINDOW, latent_dim, True)
# m = Model.BPR(uNum, vNum, WINDOW, latent_dim, 4)
#
# # for epoch in range(10):
# #     print('Epoch %s' % epoch)
# #
# #     pos_samples, neg_samples = DataParser.generate_train_pairs(tuv)
#
# X = {
#     'positive_item_input': pos_samples,
#     'negative_item_input': neg_samples,
#     'user_input': u_input
# }
#
# m.model.fit(X,
#               np.ones(uNum),
#               batch_size=uNum,
#               nb_epoch=100,
#               verbose=1,
#               shuffle=True)
#
# scores = {"p": [], "ndcg": [], "auc": [], "map": []}
#
# for t in range(WINDOW):
#     stop = 0
#     grnd = tuv_test[t].toarray()
#     for u in range(uNum):
#
#         # check if user visit any venues at time t
#         if np.all(grnd[u] == 0):
#             continue
#
#         if modelName == "bpr":
#             pred = m.predict(u, np.arange(vNum))
#         elif modelName == "rnn":
#             pred = m.predict(u, np.arange(vNum), pos_samples,t)
#
#         scores['auc'].append(roc_auc_score(grnd[u], pred))
#         scores['map'].append(average_precision_score(grnd[u], pred))
#         scores['p'].append(lm.ranking_precision_score(grnd[u], pred))
#         scores['ndcg'].append(lm.ndcg_score(grnd[u], pred))
#
#         stop = stop + 1
#         if stop == 100:
#             break
#
# print('AUC, P@10, NDCG, MAP %s %s %s %s' % (sum(scores['auc']) / len(scores['auc']), sum(scores['p']) / len(scores['p']),
#       sum(scores['ndcg']) / len(scores['ndcg']), sum(scores['map']) / len(scores['map'])))
#
# elapsed_time = dt.now() - start_time
# print("Time :", elapsed_time)


def eval(ground_truth, model, modelName, WINDOW, vNum):
    scores = {"p": [], "ndcg": [], "auc": [], "map": []}

    for t in range(WINDOW):

        ground = ground_truth[t].tocsr()

        no_users, no_items = ground.shape

        pid_array = np.arange(no_items, dtype=np.int32)

        stop = 0
        for uid, row in enumerate(ground):
            if modelName == "mf":
                usr = np.empty(vNum)
                usr.fill(uid)
                predictions = model.predict([usr, pid_array])
            if modelName == "rnn":
                predictions = model.predict( uid, pid_array, t)
            if modelName == "bpr":
                predictions = model.predict( uid, pid_array)

            if modelName == "tf":
                usr = np.empty(vNum)
                time = np.empty(vNum)
                usr.fill(uid)
                time.fill(t)
                predictions = model.predict([usr, pid_array, time])

            true_pids = row.indices[row.data == 1]

            grnd = np.zeros(no_items, dtype=np.int32)
            grnd[true_pids] = 1

            if len(true_pids) > 0:
                scores['auc'].append(roc_auc_score(grnd, predictions.reshape((vNum,))))
                scores['map'].append(average_precision_score(grnd, predictions.reshape((vNum,))))
                scores['p'].append(lm.ranking_precision_score(grnd, predictions.reshape((vNum,))))
                scores['ndcg'].append(lm.ndcg_score(grnd, predictions.reshape((vNum,))))

            # stop = stop + 1
            # if stop == 1:
            #     break

    return sum(scores['auc']) / len(scores['auc']), sum(scores['p']) / len(scores['p']), sum(scores['ndcg']) / len(
        scores['ndcg']), sum(scores['map']) / len(scores['map'])
