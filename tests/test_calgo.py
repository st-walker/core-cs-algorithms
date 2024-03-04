import pytest

from calgo import (school_multiply, karatsuba_multiply,
                   merge_sort, bubble_sort, quick_sort,
                   insertion_sort, selection_sort,
                   inversion_count, closest_pair)

UNSORTED_TEST_LIST = [5, 3, 8, 9, 1, 7, 0, 2, 6, 4]
SORTED_TEST_LIST = sorted(UNSORTED_TEST_LIST)


BIG_INT_ONE = 3141592653589793238462643383279502884197169399375105820974944592
BIG_INT_TWO = 2718281828459045235360287471352662497757247093699959574966967627

@pytest.fixture
def unsorted_list():
    return UNSORTED_TEST_LIST.copy()

def test_school_multiply():
    assert school_multiply(5678, 1234) == 7_006_652

def test_karatsuba_multiply():
    assert karatsuba_multiply(BIG_INT_ONE, BIG_INT_TWO) == (BIG_INT_ONE * BIG_INT_TWO)
    from IPython import embed; embed()
    # Some more trivial cases...
    assert karatsuba_multiply(0, 100) == 0
    assert karatsuba_multiply(1, 100) == 100
    # When number of digits in both multiplicands odd
    assert karatsuba_multiply(100, 100) == 100*100
    # When one multiplicand has odd number of digits and the other has even
    assert karatsuba_multiply(5678, 123) == 5678 * 123


def test_merge_sort(unsorted_list):
    assert merge_sort(unsorted_list) == SORTED_TEST_LIST

def test_bubble_sort():
    assert bubble_sort(unsorted_list) == SORTED_TEST_LIST

def test_quick_sort():
    assert quick_sort(unsorted_list) == SORTED_TEST_LIST

def test_insertion_sort():
    assert insertion_sort(unsorted_list) == SORTED_TEST_LIST

def test_selection_sort():
    assert selection_sort(unsorted_list) == SORTED_TEST_LIST

def test_inversion_count():
    assert inversion_count([1, 3, 5, 2, 4, 6]) == 3

def test_closest_pair():
    assert False
