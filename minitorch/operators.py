"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Any, Callable, Generator, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float:
    'Multiplies two numbers'
    return x * y


def id(x: float) -> float:
    'Returns the input unchanged'
    return x


def add(x: float, y: float) -> float:
    'Adds two numbers'
    return x + y


def neg(x: float) -> float:
    'Negates a number'
    return float(-x)


def lt(x: float, y: float) -> float:
    'Checks if one number is less than another'
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    'Checks if two numbers are equal'
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    'Returns the larger of two numbers'
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    'Checks if two numbers are close in value'
    return math.fabs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    'Calculates the sigmoid function'
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    'Applies the ReLU activation function'
    return max(x, 0.0)


def log(x: float) -> float:
    'Calculates the natural logarithm'
    return math.log(x)


def exp(x: float) -> float:
    'Calculates the exponential function'
    return math.exp(x)


def inv(x: float) -> float:
    'Calculates the reciprocal'
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    'Computes the derivative of log times a second arg'
    return y / x


def inv_back(x: float, y: float) -> float:
    'Computes the derivative of reciprocal times a second arg'
    return - y / x ** 2


def relu_back(x: float, y: float) -> float:
    'Computes the derivative of ReLU times a second arg'
    return y if x > 0.0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(func: Callable[[Any], Any], iterable: Iterable[Any]) -> Generator[Any, None, None]:
    for item in iterable:
        yield func(item)


def zipWith(iterable1: Iterable[Any], iterable2: Iterable[Any]) -> Generator[Any, None, None]:
    iterator1 = iter(iterable1)
    iterator2 = iter(iterable2)

    while True:
        try:
            yield (next(iterator1), next(iterator2))
        except StopIteration:
            return


def reduce(func: Callable[[float, float], float], iterable: Iterable[float]) -> float:
    iterator = iter(iterable)
    try:
        res = next(iterator)
    except StopIteration:
        return 0
    for item in iterator:
        res = func(res, item)
    return res


def negList(lst: list[float]) -> list[float]:
    return list(map(neg, lst))


def addLists(lst1: list[float], lst2: list[float]) -> list[float]:
    return list(f + s for f, s in zipWith(lst1, lst2))


def sum(lst: list[float]) -> float:
    return reduce(add, lst)


def prod(lst: list[float]) -> float:
    return reduce(mul, lst)
