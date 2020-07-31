import pandas as pd
import pytest

from old_subprofiles import is_subset_of_any, get_user_knn, collect_new_neighbors
from subprofiles import merge_subprofiles, get_items
from utils import to_csc


@pytest.fixture
def matrix():
    df = pd.DataFrame(
        {'user_id': [1, 1, 2, 2], 'item_id': [1, 2, 2, 3], 'rating': [5, 5, 5, 5]}
    )
    m = to_csc(df)
    return m


def test_merge():
    candidates = [
        {1, 2, 3},
        {3, 4, 5},
        {2, 3, 5},
        {5, 7, 8},
        {3, 7, 8},
        {6, 5, 9},
        {9, 0, 5},
    ]
    assert merge_subprofiles(candidates, 0.5) == [
        {3, 5, 7, 8},
        {0, 5, 6, 9},
        {1, 2, 3, 4, 5},
    ]


def test_is_subset_of_any():
    assert is_subset_of_any({1, 2}, [{1, 2}, {3}]) is True
    assert is_subset_of_any({1, 2}, [{1}, {2}]) is False


def test_get_items(matrix):
    items = get_items(matrix, 1)
    assert list(items) == [1, 2]


def test_get_knn(matrix):
    items = get_items(matrix, 1)
    mm = matrix[:, items].T
    knn = get_user_knn(items, mm, 2, 'cosine')
    assert knn[1] == {1, 2}
    assert knn[1] == knn[2]
    assert len(knn) == 2


def test_collect_new_neighbors():
    knn = {1: [2, 3], 2: [4, 5]}
    assert collect_new_neighbors(knn, [1], True) == {2, 3}
    assert collect_new_neighbors(knn, [1, 2], True) == {3, 4, 5}
    assert collect_new_neighbors(knn, [1, 2], False) == {2, 3, 4, 5}
