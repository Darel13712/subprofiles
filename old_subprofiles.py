import numpy as np
from multiprocessing import Pool, cpu_count
from sklearn.neighbors import NearestNeighbors


def is_subset_of_any(s, subprofiles):
    for sp in subprofiles:
        if s.issubset(sp):
            return True
    return False


def get_items(ui_matrix, user):
    return np.unique(ui_matrix[user].nonzero()[1])


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


def new_item_subprofiles(subprofiles, knn, k, drop_old):
    res = [collect_new_neighbors(knn, k, list(sp), drop_old) for sp in subprofiles]
    if drop_old:
        res = [s for s in res if len(s) > 0]
    return res


def collect_new_neighbors(knn, k, items, drop_old):
    candidates = set(items)
    for item in items:
        candidates |= set(list(knn[item])[:k])
    if drop_old:
        candidates -= set(items)
    return candidates


def get_old_subprofiles(ui_matrix, knn):
    users = np.unique(ui_matrix.nonzero()[0])
    with Pool(cpu_count()) as p:
        subprofiles = p.starmap(
            process_user,
            [(ui_matrix, user, knn) for user in users],
        )
        return dict(zip(users, subprofiles))
