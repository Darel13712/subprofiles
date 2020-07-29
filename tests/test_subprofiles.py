import pandas as pd
import pytest

from subprofiles import merge_subprofiles
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
