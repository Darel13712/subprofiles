import time
import pandas as pd
from rs_metrics import pandas_to_dict, ndcg, a_ndcg

from rs_datasets import MovieLens
from sklearn.model_selection import train_test_split

from old_subprofiles import get_old_subprofiles
from subprofiles import get_subprofiles, rerank
from utils import to_csc, encode, get_knn, user_split

res = pd.DataFrame()

for version in ['1m']:
    df = MovieLens(version).ratings
    df, ue, ie = encode(df)
    train, test = train_test_split(df, test_size=0.2, random_state=1, stratify=df['user_id'])
    # train, test = user_split(df, random_state=1337)
    m = to_csc(train)
    knn = get_knn(m, 100, 'cosine')

    # start = time.time()
    # sp = get_subprofiles(m, knn)
    # time_new = round(time.time() - start, 2)

    start = time.time()
    op = get_old_subprofiles(m, knn)
    time_old = round(time.time() - start, 2)

    pred = pd.read_csv(f'als_02_{version}.csv')
    score = pandas_to_dict(pred, item_col='relevance')
    pred = pandas_to_dict(pred)
    test = pandas_to_dict(test)

    rerank_old = rerank(pred, score, op, 10, knn)
    # rerank_new = rerank(pred, score, sp, 10)

    base_score = ndcg(test, pred)
    score_old = ndcg(test, rerank_old)
    # score_new = ndcg(test, rerank_new)

    res = res.append(
        {
            'base ndcg@10': base_score,
            'old ndcg@10': score_old,
            # 'new ndcg@10': score_new,
            'old time': time_old,
            # 'new time': time_new,
            'dataset': version,
        },
        ignore_index=True,
    )

res.to_csv('proportion_split/results.csv', index=False)
print(res)