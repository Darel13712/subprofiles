import pandas as pd
import pytest

from subprofiles import (
    is_subset_of_any,
    get_items,
    get_knn,
    mark_labels,
    collect_new_neighbors,
)
from utils import to_csr


@pytest.fixture
def matrix():
    df = pd.DataFrame(
        {'user_id': [1, 1, 2, 2], 'item_id': [1, 2, 2, 3], 'rating': [5, 5, 5, 5]}
    )
    m = to_csr(df)
    return m


def test_is_subset_of_any():
    assert is_subset_of_any({1, 2}, [{1, 2}, {3}]) == True
    assert is_subset_of_any({1, 2}, [{1}, {2}]) == False


def test_get_items(matrix):
    items, mm = get_items(matrix, 1)
    assert list(items) == [1, 2]
    assert mm.shape == (2, 3)


def test_get_knn(matrix):
    items, mm = get_items(matrix, 1)
    knn = get_knn(items, mm, 2, 'cosine')
    assert knn[1] == {1, 2}
    assert knn[1] == knn[2]
    assert len(knn) == 2


def test_mark_labels():
    sp = [{1, 2}, {2, 3}]
    labels = mark_labels([1, 2, 3, 4], sp)
    assert list(labels) == [0, 0, 1, -1]


def test_collect_new_neighbors():
    knn = {1: [2, 3], 2: [4, 5]}
    assert collect_new_neighbors(knn, [1]) == [2, 3]
    assert collect_new_neighbors(knn, [1, 2]) == [3, 4, 5]
