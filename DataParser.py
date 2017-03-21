import pandas as pd
import numpy as np
import scipy.sparse as sp
import random

# use to generate training files
def start():
    filename = '../../checkins/standford_checkins/loc-gowalla_totalCheckins.txt'
    separater = '\t'
    names = ['uid','time','lat','lon','vid']

    userCheckins, uNum, vNum = read_data(filename, separater, names)

    print(userCheckins)

def read_data(filename, separater, names):
    df = pd.read_csv(filename, sep=separater, names=names)
    # covert time to datetime column
    df['time'] = pd.to_datetime(df['time'])
    # extract datetime to time window
    df['day'] = df.time.dt.dayofweek
    df['hour'] = df.time.dt.hour

    # filtering user&venue with less than 10 check-ins
    df = (df
     .merge(df.groupby('uid').vid.nunique().reset_index().rename(columns={'vid': 'num_uniq_vid'}), on='uid', how='left')
     .merge(df.groupby('vid').uid.nunique().reset_index().rename(columns={'uid': 'num_uniq_uid'}), on='vid', how='left'))
    df = df[(df.num_uniq_vid >= 10) & ((df.num_uniq_uid >= 10))]

    # indexing uid&vid and sort by uid and time
    dataset = df.merge(pd.Series(df.uid.unique()).reset_index().rename(columns={'index': 'new_uid', 0:'uid'}), left_on='uid', right_on='uid').merge(pd.Series(df.vid.unique()).reset_index().rename(columns={'index': 'new_vid', 0:'vid'}), left_on='vid', right_on='vid')
    del dataset['uid']
    del dataset['vid']
    dataset = dataset.rename(columns={'new_uid':'uid', 'new_vid':'vid'})
    uNum = dataset.uid.nunique()
    vNum = dataset.vid.nunique()
    sparsity = (dataset.shape[0] / (uNum*vNum))*100
    print("sparsity  %.2f% %" % sparsity)
    sort = dataset[['uid','vid','hour','time']].sort_values(['uid','time'])

    # generate userCheckins
    gb = sort.groupby('uid')    
    userCheckins = [gb.get_group(x) for x in gb.groups]
    assert len(userCheckins) == uNum
    return userCheckins,uNum, vNum



def generate_train_pairs(tuv):
    pos_samples = []
    neg_samples = []

    uNum,vNum = tuv[0].shape
    WINDOW = len(tuv)

    for user in range(uNum):
        pos = []
        neg = []
        for t in range(WINDOW):
            venues_at_t = tuv[t][user].tocoo().col
            if(len(venues_at_t) > 0):
                pos.append(random.choice(venues_at_t))
                tmp = random.randint(0, vNum)
                while(tmp in venues_at_t):
                    tmp = random.randint(0, vNum)
                neg.append(tmp)
            else:
                pos.append(0)
                neg.append(0)
        
        pos_samples.append(pos)
        neg_samples.append(neg)
    return np.array(pos_samples), np.array(neg_samples)


def generate_train_pairs(tuv):
    pos_samples = []
    neg_samples = []
    
    uNum, vNum = tuv[0].shape
    WINDOW = len(tuv)

    for user in range(uNum):
        pos = []
        neg = []
        for t in range(WINDOW):
            venues_at_t = tuv[t][user].tocoo().col
            if(len(venues_at_t) > 0):
                pos.append(random.choice(venues_at_t))
                tmp = random.randint(0, vNum)
                while(tmp in venues_at_t):
                    tmp = random.randint(0, vNum)
                neg.append(tmp)
            else:
                pos.append(0)
                neg.append(0)
        
        pos_samples.append(pos)
        neg_samples.append(neg)
    return np.array(pos_samples), np.array(neg_samples)


def generate_bpr_pairs(uv, uNum, vNum):

    pair_u, pair_i, pair_j = [], [], []

    for u in range(uNum):
        visited = uv[u].tocoo().col
        for i in visited:

            j = sampleNeg(visited, vNum)

            pair_u.append(u)
            pair_i.append(i)
            pair_j.append(j)

    return np.array(pair_u), np.array(pair_i), np.array(pair_j)

def generate_rnn_pairs(tuv, uNum, vNum, WINDOW):
    rnn_pair_u, rnn_pair_i, rnn_pair_j, rnn_last_pair = [],[],[],[]

    for u in range(uNum):
        for t in range(WINDOW):
            visited = tuv[t][u].tocoo().col

            for i in visited:
                pos = np.zeros(WINDOW)
                neg = np.zeros(WINDOW)
                pos[t] = i
                neg[t] = sampleNeg(visited, vNum)

                for t2 in range(WINDOW):
                    if t == t2:
                        continue

                    visited2 = tuv[t2][u].tocoo().col
                    if (len(visited2) > 0):
                        pos[t2] = random.choice(visited2)
                        neg[t2] = sampleNeg(visited2, vNum)

                u_ = np.zeros(WINDOW)
                u_.fill(u)
                rnn_pair_u.append(u_)
                rnn_pair_i.append(pos)
                rnn_pair_j.append(neg)
            rnn_last_pair.append(pos)

    return np.array(rnn_pair_u), np.array(rnn_pair_i), np.array(rnn_pair_j),np.array(rnn_last_pair).reshape((uNum, WINDOW,WINDOW))


def sampleNeg(visited, vNum):
    j = np.random.randint(vNum)
    while j in visited:
        j = np.random.randint(vNum)
    return j

def sparseTUV(df, mode, WINDOW, uNum, vNum):
    tuv = []
    
    for i in range(WINDOW):
        tuv.append(sp.lil_matrix((uNum, vNum), dtype=np.int32))
    if mode == "day":
        for u,v,t in zip(df.uid.values, df.vid.values, df.day.values):
            tuv[t][u,v] = 1
    elif mode == "hour":
        for u,v,t in zip(df.uid.values, df.vid.values, df.hour.values):
            tuv[t][u,v] = 1
    return tuv

def sparseUV(df, uNum, vNum):
    uv = sp.lil_matrix((uNum, vNum), dtype=np.int32)
    for u, v in zip(df.uid.values, df.vid.values):
        uv[u, v] = 1
    return uv

def readCheckinData(trainDir,testDir, separater='\t'):
    names = ['uid','vid','hour','day','time']
    df = pd.read_csv(trainDir, sep=separater, names=names)
    uNum = df.uid.unique().max()+1
    vNum = df.vid.unique().max()+1
    df_test = pd.read_csv(testDir, sep=separater, names=names)
    return df, df_test,uNum,vNum





