import numpy as np
from multiprocessing import Pool, cpu_count


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


def subprofiles(ui_matrix, knn, threshold=0.5):
    users = np.unique(ui_matrix.nonzero()[0])
    with Pool(cpu_count()) as p:
        subprofiles = p.starmap(
            get_user_subprofiles, [(ui_matrix, user, knn, threshold) for user in users],
        )
        return dict(zip(users, subprofiles))
