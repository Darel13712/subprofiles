import time

import joblib
import pandas as pd
from rs_metrics import pandas_to_dict, ndcg, a_ndcg
from sklearn.cluster import MeanShift

from rs_datasets import MovieLens
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

from old_subprofiles import get_old_subprofiles, get_ms_subprofiles
from subprofiles import get_subprofiles
from reranking import rerank, neighborhood
from utils import get_knn, user_split
from rs_tools import to_csc, encode

res = pd.Series(dtype=float)

version = '10m'
df = MovieLens(version).ratings
df, ue, ie = encode(df)
train, test = train_test_split(df, test_size=0.2, random_state=1, stratify=df['user_id'])
# train, test = user_split(df, random_state=1337)
mui = to_csc(train)
items_in_train = np.unique(mui.nonzero()[1])
dts_dict = dict(enumerate(items_in_train))
std_dict = {v: k for k, v in dts_dict.items()}
pred = joblib.load('/mnt2/darel/als.jbl')
emb = joblib.load('/mnt2/darel/emb.jbl')
score = pandas_to_dict(pred, item_col='rating')
pred = pandas_to_dict(pred)
test = pandas_to_dict(test)
res.loc['no rerank'] = ndcg(test, pred)
for metric in tqdm(['euclidean']):
    knn = get_knn(mui.T, 100, metric)
    labels = MeanShift().fit_predict(emb)
    labels = {dts_dict[num]: val for num, val in enumerate(labels)}
    clusters = dict()
    for k, v in labels.items():
        if v in clusters:
            clusters[v].append(k)
        else:
            clusters[v] = [k]

    # start = time.time()
    op = get_old_subprofiles(mui, knn)
    ope = get_ms_subprofiles(mui, labels, clusters)
    # time_old = round(time.time() - start, 2)


    rerank_old = rerank(pred, score, op, 10, knn)
    rerank_emb = rerank(pred, score, ope, 10, None)
    # rerank_new = rerank(pred, score, sp, 10)

    # base_score = ndcg(test, pred)
    score_old = ndcg(test, rerank_old)
    score_emb = ndcg(test, rerank_emb)
    # score_new = ndcg(test, rerank_new)
    res.loc[metric + ' classic'] = score_old
    res.loc[metric + ' emb'] = score_emb
    # print(metric, score_old)

# pd.DataFrame(res).to_csv(f'metrics_{version}.csv', index=False)
print(res)
joblib.dump(res, 'ms.jbl')