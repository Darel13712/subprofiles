import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from sklearn.cluster import MeanShift


def is_subset_of_any(s, subprofiles):
    for sp in subprofiles:
        if s.issubset(sp):
            return True
    return False


def get_items(ui_matrix, user):
    return np.unique(ui_matrix[user].nonzero()[1])


def add_neighbours(items, knn):
    res = set(items)
    for item in items:
        res = res.union(set(knn[item]))
    return list(res)


def get_user_subprofiles(items, knn):
    """Subprofiles for a single user"""
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


def process_user(ui_matrix, user, knn):
    items = get_items(ui_matrix, user)
    subprofiles = get_user_subprofiles(items, knn)
    return subprofiles


def get_old_subprofiles(ui_matrix, knn):
    users = np.unique(ui_matrix.nonzero()[0])
    with Pool(cpu_count()) as p:
        subprofiles = p.starmap(
            process_user,
            [(ui_matrix, user, knn) for user in users],
        )
        return dict(zip(users, subprofiles))


def process_user_ms(ui_matrix, user, labels, clusters):
    items = get_items(ui_matrix, user)
    items = np.array(items)
    cl = list(set([labels[item] for item in items]))
    subprofiles = [set(clusters[c]) for c in cl]
    return subprofiles


def get_ms_subprofiles(ui_matrix, labels, clusters):
    users = np.unique(ui_matrix.nonzero()[0])
    with Pool(cpu_count()) as p:
        subprofiles = p.starmap(
            process_user_ms,
            [(ui_matrix, user, labels, clusters) for user in users],
        )
        return dict(zip(users, subprofiles))