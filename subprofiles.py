import warnings

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.neighbors import NearestNeighbors, NearestCentroid


def is_subset_of_any(s, subprofiles):
    for sp in subprofiles:
        if s.issubset(sp):
            return True
    return False


def get_items(ui_matrix, user):
    items = np.unique(ui_matrix[user].nonzero()[1])
    iu_matrix = ui_matrix[:, items].T
    return items, iu_matrix


def get_user_subprofiles(items, iu_matrix, k, metric):
    """Subprofiles for a single user"""
    knn = get_knn(items, iu_matrix, k, metric)
    candidate_subprofiles = [
        {base_item for base_item in items if item in knn[base_item]}.union({item})
        for item in items
    ]
    candidate_subprofiles.sort(key=len, reverse=True)
    subprofiles = []
    for s in candidate_subprofiles:
        if not is_subset_of_any(s, subprofiles):
            subprofiles.append(s)
    return subprofiles


def get_knn(items, iu_matrix, k, metric):
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(iu_matrix)
    ind = nbrs.kneighbors(iu_matrix, return_distance=False)  # new indices
    knn = [{items[item] for item in nn} for nn in ind]  # old indices
    knn = dict(zip(items, knn))
    return knn


def process_user(ui_matrix, user, k, metric, knn, drop_old):
    items, iu_submatrix_for_user = get_items(ui_matrix, user)
    subprofiles = get_user_subprofiles(items, iu_submatrix_for_user, k, metric)
    if knn is None:
        return subprofiles
    sp = new_item_subprofiles(
        items, iu_submatrix_for_user, knn, metric, subprofiles, ui_matrix
    )
    if drop_old:
        sp = [s for s in sp if len(s) > 0]
        return sp
    subprofiles = [old.union(new) for old, new in zip(subprofiles, sp)]
    return subprofiles


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


def subprofiles(ui_matrix, k=10, metric='cosine', target='old'):
    if target not in ['new', 'both', 'old']:
        raise ValueError(f'target must be one of "old", "new", "both", got {target}')

    knn = None
    drop_old = True if target == 'new' else False

    if target != 'old':
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(ui_matrix.T)
        knn = nbrs.kneighbors(ui_matrix.T, return_distance=False)

    users = np.unique(ui_matrix.nonzero()[0])
    with Pool(cpu_count()) as p:
        subprofiles = p.starmap(
            process_user,
            [(ui_matrix, user, k, metric, knn, drop_old) for user in users],
        )
        return dict(zip(users, subprofiles))
