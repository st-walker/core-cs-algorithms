from typing import Iterable
import logging
import math
from random import randrange

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def school_multiply(a, b):
    pass

def _ndigits(x: int) -> int:
    """Returns the number of digits for the given integer"""

    if x < 0:
        x = abs(x)

    if x == 0:
        return 1
    else:
        return int(math.log10(x)) + 1

def karatsuba_multiply(x: int, y: int) -> int:
    """Karatsuba multiply two integers a and b."""
    # We do string operations here because otherwise we would have to
    # use division and we want to pretend we are doing a recursive
    # algorithm in terms of very basic primitive ops.
    nx = _ndigits(x)
    ny = _ndigits(y)

    if nx == 1 and ny == 1: # base case
        return x * y

    # We insist on n being either 1 or an even number, we need to find
    # the appropriate n and pad the number to get it up to even an even value if necessary.
    n = max(nx, ny)
    if n % 2 != 0:
        n += 1

    # Convert to strings for extracting halves
    x = str(x).zfill(n)
    y = str(y).zfill(n)

    # Get first and second halves of x
    a = int(x[:n // 2])
    b = int(x[n // 2:])
    # Get first and second halves of y.
    c = int(y[:n // 2])
    d = int(y[n // 2:])

    # Assume we can add two arbitrarily large integers at least.
    p = a + b
    q = c + d

    # Gauss trick
    ac = karatsuba_multiply(a, c)
    bd = karatsuba_multiply(b, d)
    pq = karatsuba_multiply(p, q)
    adbc = pq - ac - bd

    # Multiplying by factors of 10 we consider to be a primitive operation in this instance...
    return 10**n * ac + 10**(n//2) * adbc + bd


def merge_sort[T](iterable: Iterable[T]) -> list[T]:
    """Do a sort of the given iterable and return a list of its sorted elements."""
    iterable = list(iterable)
    # Need to convert to list to use len, e.g. if a generator...

    if len(iterable) == 1: # Base case
        return iterable

    # Divide in two
    left = iterable[:len(iterable)//2]
    right = iterable[len(iterable)//2:]
    # Merge the two sorted halves
    return _merge(merge_sort(left), merge_sort(right))

def _merge[T](left: Iterable[T], right: Iterable[T]) -> list[T]:
    """The merge part of the merge sort"""
    result = []
    i = 0
    j = 0
    LOG.debug(f"Merging {left} and {right}")
    while True:
        lhead = left[i]
        rhead = right[j]
        if lhead < rhead:
            result.append(lhead)
            i += 1
        else:
            result.append(rhead)
            j += 1

        # If we have exhausted one of the two sublists then copy the
        # remaining ones into the final result list
        if i == len(left):
            result.extend(right[j:])
            break
        elif j == len(right):
            result.extend(left[i:])
            break

    return result

def _merge_and_count_inversions[T](left: Iterable[T], right: Iterable[T]) -> list[T]:
    """The merge part of the merge sort"""
    result = []
    i = 0
    j = 0
    LOG.debug(f"Merging {left} and {right}")
    inversion_count = 0
    while True:
        lhead = left[i]
        rhead = right[j]
        if lhead < rhead:
            result.append(lhead)
            i += 1
        else:
            result.append(rhead)
            j += 1
            # Add up number of elements that are still in the left branch.  If we know we have a split inversion,
            # because both subarrays are sorted, everything to the right of the left element we're at
            # will also be a split inversion.
            # So e.g. (1, 3, 5), (2, 4, 6), when we are comparing (3, 2), we also know that everything to the right
            # Of the 3 will also be a split inversion, because we know everything to the right of 3 is bigger than 3,
            # And so everything will be bigger than the 2, because the comparison is transitive.
            inversion_count += len(left[i:])


        # If we have exhausted one of the two sublists then copy the
        # remaining ones into the final result list
        if i == len(left):
            result.extend(right[j:])
            break
        elif j == len(right):
            result.extend(left[i:])
            break

    return result, inversion_count

def inversion_count[T](iterable: Iterable[T]) -> int:
    if len(iterable) <= 1:
        return iterable, 0

    left = iterable[:len(iterable)//2]
    right = iterable[len(iterable)//2:]

    lsorted, lcount = inversion_count(left)
    rsorted, rcount = inversion_count(right)

    merged_sorted, scount = _merge_and_count_inversions(lsorted, rsorted)
    # print(lcount, scount, rcount)
    return merged_sorted, lcount + scount + rcount


def closest_pair(iterable):
    pass

def bubble_sort(sequence: list[int | float]) -> None:
    """Bubble sort IN-PLACE"""
    n = len(sequence)
    swapped = False
    for i, inext in zip(range(n), range(1, n)):
        xi = sequence[i]
        xnext = sequence[inext]

        if xi > xnext:
            swapped = True
            _swap(sequence, i, inext)

    # If we did a swap then we go again, otherwise we are done.
    if swapped:
        bubble_sort(sequence)


def _choose_pivot(sequence, l, r, policy):
    if policy == "first":
        return l
    elif policy == "last":
        return r
    elif policy == "random":
        return randrange(l, r)
    elif policy == "3median":
        n = (r + 1) - l
        lval = sequence[l]
        rval = sequence[r]

        m = n // 2
        if n % 2 == 0:
            m -= 1

        mval = sequence[m]
        print(lval, mval, rval)

        median = _3median(lval, mval, rval)
        if median == mval:
            return m
        elif median == lval:
            return l
        elif median == rval:
            return r


def _3median(a, b, c):
    if (a - b) * (c - a) >= 0:
        return a
    elif (b - a) * (c - b) >= 0:
        return b
    else:
        return c


def _swap(l: list, i: int, j: int) -> None:
    l[i], l[j] = l[j], l[i]


def quicksort(sequence: list, policy="first") -> int:
    count = _quicksort_impl(sequence, 0, len(sequence) - 1, policy)
    return count


def _quicksort_impl(sequence: list, left: int, right: int, policy="random") -> int:
    if left >= right: # 0- or 1-element subarray
        return 0

    chosen_pivot = _choose_pivot(sequence, left, right, policy=policy)

    _swap(sequence, left, chosen_pivot) # Move pivot to start of the subsequence

    # Partition the subsequence.  Returns the final position of the
    # pivot, everything less than this is less than the pivot (but
    # unsorted) and everything to the right of this point is greater
    # than the pivot (but also otherwise unsorted).
    sorted_pivot_index = _partition(sequence, left, right)

    # Recurse on left side of pivot.  Don't include pivot as we
    # already know it's exactly in the right place, so subtract 1.
    m = len(sequence[left:right+1]) - 1

    m1 = _quicksort_impl(sequence, left, sorted_pivot_index - 1, policy=policy)
    # Recurse on right side of pivot.  Don't include the pivot as we
    # already know it's exactly in the right place, so add 1.
    m2 = _quicksort_impl(sequence, sorted_pivot_index + 1, right, policy=policy)
    return m + m1 + m2

def _partition(sequence: list, l: int, r: int) -> int:
    """Input: a sequence with the left and right endpoints demarcating
    a subsequence to be partioned.  The element at sequence[l] is used
    as the pivot.

    Output: final position of the pivot element.

    """

    pivot = sequence[l]
    i = l + 1 # +1 to go one past the pivot
    # r + 1 so that we include the r element.
    for j in range(l + 1, r + 1):
        if sequence[j] < pivot:
            _swap(sequence, i, j)
            i += 1

    # Place pivot where it belongs
    _swap(sequence, l, i - 1)

    return i - 1


def insertion_sort(iterable):
    pass

def selection_sort(iterable):
    pass
