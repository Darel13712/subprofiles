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


def get_user_subprofiles(items, iu_matrix, k, metric):
    """Subprofiles for a single user"""
    knn = get_user_knn(items, iu_matrix, k, metric)
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


def get_user_knn(items, iu_matrix, k, metric):
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(iu_matrix)
    ind = nbrs.kneighbors(iu_matrix, return_distance=False)  # new indices
    knn = [{items[item] for item in nn} for nn in ind]  # old indices
    knn = dict(zip(items, knn))
    return knn


def process_user(ui_matrix, user, k, metric, knn, drop_old):
    items = get_items(ui_matrix, user)
    iu_submatrix_for_user = ui_matrix[:, items].T
    subprofiles = get_user_subprofiles(items, iu_submatrix_for_user, k, metric)
    if knn is None:
        return subprofiles
    sp = new_item_subprofiles(subprofiles, knn, drop_old)
    return sp


def new_item_subprofiles(subprofiles, knn, drop_old):
    res = [collect_new_neighbors(knn, list(sp), drop_old) for sp in subprofiles]
    if drop_old:
        res = [s for s in res if len(s) > 0]
    return res


def collect_new_neighbors(knn, items, drop_old):
    candidates = set()
    for item in items:
        candidates |= set(knn[item])
    if drop_old:
        candidates -= set(items)
    return candidates


def get_old_subprofiles(ui_matrix, k, metric='cosine', knn=None, target='old'):
    if target not in ['new', 'both', 'old']:
        raise ValueError(f'target must be one of "old", "new", "both", got {target}')
    if target != 'old' and knn is None:
        raise ValueError('knn parameter must be provided if target != old')

    drop_old = True if target == 'new' else False
    users = np.unique(ui_matrix.nonzero()[0])
    with Pool(cpu_count()) as p:
        subprofiles = p.starmap(
            process_user,
            [(ui_matrix, user, k, metric, knn, drop_old) for user in users],
        )
        return dict(zip(users, subprofiles))
