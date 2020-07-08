import numpy as np
from multiprocessing import Pool, cpu_count
from sklearn.neighbors import NearestNeighbors


def is_subset_of_any(s, subprofiles):
    for sp in subprofiles:
        if s.issubset(sp):
            return True
    return False


def get_items(user, ui_matrix):
    items = np.unique(ui_matrix[user].nonzero()[1])
    iu_matrix = ui_matrix[:, items].T
    return items, iu_matrix


def get_user_subprofiles(items, iu_matrix, k, metric):
    """Subprofiles for single user"""
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(iu_matrix)
    ind = nbrs.kneighbors(iu_matrix, return_distance=False)  # new indices
    knn = [{items[item] for item in nn} for nn in ind]  # old indices
    knn = dict(zip(items, knn))
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


def subprofiles(ui_matrix, k=10, metric='cosine'):
    users = np.unique(ui_matrix.nonzero()[0])
    with Pool(cpu_count()) as p:
        subprofiles = p.starmap(
            get_user_subprofiles,
            [(*get_items(user, ui_matrix), k, metric) for user in users],
        )
        return dict(zip(users, subprofiles))
