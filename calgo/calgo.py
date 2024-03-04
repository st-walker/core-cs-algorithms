from typing import Iterable
import logging
import math

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

def bubble_sort(iterable):
    pass

def quick_sort(iterable):
    pass

def insertion_sort(iterable):
    pass

def selection_sort(iterable):
    pass

def inversion_count(iterable):
    pass

def closest_pair(iterable):
    pass
