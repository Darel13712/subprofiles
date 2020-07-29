import warnings

import numpy as np
from multiprocessing import Pool, cpu_count
from sklearn.neighbors import NearestCentroid


def is_subset_of_any(s, subprofiles):
    for sp in subprofiles:
        if s.issubset(sp):
            return True
    return False


def get_items(ui_matrix, user):
    return np.unique(ui_matrix[user].nonzero()[1])


def get_user_subprofiles(ui_matrix, user, knn, threshold):
    """Subprofiles for a single user"""
    items = get_items(ui_matrix, user)
    candidates = [knn[item] for item in items]
    candidates.sort(key=len, reverse=True)
    res = merge_subprofiles(candidates, threshold)
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


def new_item_subprofiles(
    old_items, iu_submatrix_for_user, knn, metric, subprofiles, ui_matrix
):
    candidates = collect_new_neighbors(knn, old_items)
    y = mark_labels(old_items, subprofiles)
    if len(np.unique(y)) > 1:
        clf = NearestCentroid(metric)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(iu_submatrix_for_user, y)
        x = ui_matrix[:, candidates].T
        pred = clf.predict(x)
    else:
        pred = np.zeros(len(candidates), int)
    sp = [set() for _ in range(len(subprofiles))]
    for item, label in zip(candidates, pred):
        sp[label] |= {item}
    return sp


def mark_labels(items, subprofiles):
    """get subprofile index for every item"""
    y = np.zeros(len(items)).astype(int) - 1
    for i in range(len(items)):
        for num, sp in enumerate(subprofiles):
            if y[i] < 0 and items[i] in sp:
                y[i] = num
                break
    return y


def collect_new_neighbors(knn, items):
    candidates = set()
    for item in items:
        candidates |= set(knn[item])
    candidates -= set(items)
    return list(candidates)


def subprofiles(ui_matrix, knn, threshold=0.5):
    users = np.unique(ui_matrix.nonzero()[0])
    with Pool(cpu_count()) as p:
        subprofiles = p.starmap(
            get_user_subprofiles, [(ui_matrix, user, knn, threshold) for user in users],
        )
        return dict(zip(users, subprofiles))
