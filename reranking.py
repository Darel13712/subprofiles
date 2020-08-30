from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd


def rerank(items, scores, subprofiles, k, knn=None, lmbd=0.5):
    users = items.keys()
    with Pool(cpu_count()) as p:
        res = p.starmap(
            user_ranking,
            [(items[user], scores[user], subprofiles[user], k, knn, lmbd) for user in users],
        )
    return dict(zip(users, res))


def user_ranking(items, scores, subprofiles, k, knn, lmbd):
    scores = list(scores)
    subprofile_prob = np.array([len(sp) for sp in subprofiles])
    subprofile_prob = subprofile_prob / subprofile_prob.sum()
    if knn is None:
        extended_subprofiles = subprofiles
    else:
        extended_subprofiles = [neighborhood(sp, knn) for sp in subprofiles]

    items = pd.Series(items)
    item_prob_den = np.array([sum(items.isin(esp) * scores) for esp in extended_subprofiles])
    items = list(items)
    not_from_subprofiles = item_prob_den == 0
    if not_from_subprofiles.any():
        item_prob_den[not_from_subprofiles] = 1 # другое число?

    res = []
    penalty = [1] * len(subprofiles)
    for i in range(k):
        obj = [
            (1 - lmbd) * score + lmbd * diversity(
                item, penalty, extended_subprofiles, subprofile_prob, score / item_prob_den
            )
            for item, score in zip(items, scores)
        ]
        pos = np.argmax(obj)
        pick = items[pos]
        res.append(pick)
        for j, sp in enumerate(extended_subprofiles):
            if pick in sp:
                penalty[j] *= 1 - scores[pos]/item_prob_den[j]
        items.pop(pos)
        scores.pop(pos)
    return res


def diversity(item, penalty, extended_subprofiles, subprofile_prob, item_prob):
    div = 0 # что делатль с другими айтемами?
    for i, sp in enumerate(extended_subprofiles):
        if item in sp:
            div += subprofile_prob[i] * item_prob[i] * penalty[i]
    return div


def neighborhood(sp, knn):
    res = set()
    for item in list(sp):
        res |= set(knn[item]).union([item])
    return res