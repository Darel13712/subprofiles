import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count


def get_items(ui_matrix, user):
    return np.unique(ui_matrix[user].nonzero()[1])


def drop_liked(sp, items):
    items = set(items)
    res = [s - items for s in sp]
    res = [s for s in res if s]
    return res


def get_user_subprofiles(ui_matrix, user, knn, threshold):
    """Subprofiles for a single user"""
    items = get_items(ui_matrix, user)
    candidates = [knn[item] for item in items]
    candidates.sort(key=len, reverse=True)
    res = merge_subprofiles(candidates, threshold)
    res = drop_liked(res, items)
    return res


def merge_subprofiles(candidates, threshold):
    merging = True
    while merging:
        merging = False
        for j in range(len(candidates)):
            sp = candidates.pop(0)
            restart = True
            while restart:
                restart = False
                for i, c in enumerate(candidates):
                    denominator = c if len(c) < len(sp) else sp
                    coef = len(sp & c) / len(denominator)
                    if coef > threshold:
                        sp |= candidates.pop(i)
                        restart = True
                        merging = True
            candidates.append(sp)
    return candidates


def get_subprofiles(ui_matrix, knn, threshold=0.5):
    users = np.unique(ui_matrix.nonzero()[0])
    with Pool(cpu_count()) as p:
        res = p.starmap(
            get_user_subprofiles, [(ui_matrix, user, knn, threshold) for user in users],
        )
        return dict(zip(users, res))


def rerank(items, scores, subprofiles, k, lmbd=0.5):
    users = items.keys()
    with Pool(cpu_count()) as p:
        res = p.starmap(
            user_ranking,
            [(items[user], scores[user], subprofiles[user], k, lmbd) for user in users],
        )
    return dict(zip(users, res))


def user_ranking(items, scores, subprofiles, k, lmbd):
    subprofile_prob = np.array([len(sp) for sp in subprofiles])
    subprofile_prob /= sum(subprofile_prob)

    items = pd.Series(items)
    item_prob_den = np.array([sum(items.isin(sp) * scores) for sp in subprofiles])
    not_from_subprofiles = item_prob_den == 0
    if not_from_subprofiles.any():
        item_prob_den[not_from_subprofiles] = 1

    res = []
    penalty = [1] * len(subprofiles)
    for i in range(k):
        obj = [
            (1 - lmbd) * score + lmbd * diversity(
                item, penalty, subprofiles, subprofile_prob, score / item_prob_den
            )
            for item, score in zip(items, scores)
        ]
        pos = np.argmax(obj)
        pick = items[pos]
        res.append(pick)
        for j, sp in enumerate(subprofiles):
            if pick in sp:
                penalty[j] *= 1 - scores[pos]*item_prob_den[j]
        items.pop(pos)
        scores.pop(pos)
    return res


def diversity(item, penalty, subprofiles, subprofile_prob, item_prob):
    div = 0
    for i, sp in enumerate(subprofiles):
        if item in sp:
            div += subprofile_prob[i] * item_prob[i] * penalty[i]
    return div
