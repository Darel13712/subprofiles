import pandas as pd
import pytest

from subprofiles import (
    is_subset_of_any,
    mark_labels,
    collect_new_neighbors,
    merge_subprofiles,
)
from utils import to_csc


@pytest.fixture
def matrix():
    df = pd.DataFrame(
        {'user_id': [1, 1, 2, 2], 'item_id': [1, 2, 2, 3], 'rating': [5, 5, 5, 5]}
    )
    m = to_csc(df)
    return m


def test_is_subset_of_any():
    assert is_subset_of_any({1, 2}, [{1, 2}, {3}]) == True
    assert is_subset_of_any({1, 2}, [{1}, {2}]) == False


def test_mark_labels():
    sp = [{1, 2}, {2, 3}]
    labels = mark_labels([1, 2, 3, 4], sp)
    assert list(labels) == [0, 0, 1, -1]


def test_collect_new_neighbors():
    knn = {1: [2, 3], 2: [4, 5]}
    assert collect_new_neighbors(knn, [1]) == [2, 3]
    assert collect_new_neighbors(knn, [1, 2]) == [3, 4, 5]


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
