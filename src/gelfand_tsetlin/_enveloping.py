"""Exact PBW expressions in the matrix units of gl_N."""

from collections.abc import Mapping
from fractions import Fraction
from math import isqrt
from operator import index
from sys import maxsize
from types import MappingProxyType

from ._uea import multiply_ordered_pbw, multiply_pbw


def _checked_rank(rank):
    rank = index(rank)
    if not 1 <= rank <= isqrt(maxsize):
        raise ValueError("rank must be positive and rank squared must fit a native index")
    return rank


def _matrix_unit_id(rank, i, j):
    i, j = index(i), index(j)
    if not (1 <= i <= rank and 1 <= j <= rank):
        raise ValueError("matrix-unit indices must be within the rank")
    return (i - 1) * rank + j - 1


def _rational(value):
    if not isinstance(value, (int, Fraction)):
        raise TypeError("enveloping coefficients must be integers or fractions")
    return Fraction(value) if isinstance(value, Fraction) else int(value)


def _add_terms(destination, source):
    for word, value in source.items():
        total = destination.get(word, 0) + value
        if total:
            destination[word] = total
        else:
            destination.pop(word, None)


class EnvelopingElement:
    """A sparse rational element of U(gl_N) in row-major PBW order.

    Each term key is a word of one-based matrix-unit pairs. The empty word
    denotes the identity. The constructor normal-orders arbitrary words.
    """

    __slots__ = ("_rank", "_terms")

    def __init__(self, rank, terms: Mapping):
        self._rank = _checked_rank(rank)
        packed = {}
        for word, value in terms.items():
            key = tuple(_matrix_unit_id(self._rank, i, j) for i, j in word)
            total = packed.get(key, 0) + _rational(value)
            if total:
                packed[key] = total
            else:
                packed.pop(key, None)
        self._terms = multiply_pbw(self._rank, {(): 1}, packed)

    @classmethod
    def _from_ordered(cls, rank, terms):
        element = cls.__new__(cls)
        element._rank = rank
        element._terms = terms
        return element

    @classmethod
    def _sum_ordered(cls, rank, elements):
        terms = {}
        for element in elements:
            _add_terms(terms, element._terms)
        return cls._from_ordered(rank, terms)

    @classmethod
    def matrix_unit(cls, rank, i, j):
        """Return the generator E[i,j] with one-based indices."""
        rank = _checked_rank(rank)
        return cls._from_ordered(rank, {(_matrix_unit_id(rank, i, j),): 1})

    @classmethod
    def scalar(cls, rank, coefficient):
        """Return a scalar times the unit element."""
        rank = _checked_rank(rank)
        value = _rational(coefficient)
        return cls._from_ordered(rank, {(): value} if value else {})

    @property
    def rank(self):
        return self._rank

    @property
    def terms(self):
        """Read-only mapping from PBW words to exact coefficients."""
        terms = {
            tuple((unit // self.rank + 1, unit % self.rank + 1) for unit in word): value
            for word, value in sorted(self._terms.items())
        }
        return MappingProxyType(terms)

    def __repr__(self):
        return f"{type(self).__name__}({self.rank}, {dict(self.terms)!r})"

    def __bool__(self):
        return bool(self._terms)

    def __eq__(self, other):
        if isinstance(other, EnvelopingElement):
            return self.rank == other.rank and self._terms == other._terms
        if isinstance(other, (int, Fraction)):
            return self._terms == ({(): other} if other else {})
        return NotImplemented

    def _as_element(self, other):
        if isinstance(other, EnvelopingElement):
            if self.rank != other.rank:
                raise ValueError("enveloping elements must have the same rank")
            return other
        if isinstance(other, (int, Fraction)):
            return self.scalar(self.rank, other)
        return NotImplemented

    def __add__(self, other):
        other = self._as_element(other)
        if other is NotImplemented:
            return NotImplemented
        terms = self._terms.copy()
        _add_terms(terms, other._terms)
        return self._from_ordered(self.rank, terms)

    __radd__ = __add__

    def __neg__(self):
        return self._from_ordered(self.rank, {word: -value for word, value in self._terms.items()})

    def __sub__(self, other):
        other = self._as_element(other)
        return NotImplemented if other is NotImplemented else self + (-other)

    def __rsub__(self, other):
        other = self._as_element(other)
        return NotImplemented if other is NotImplemented else other + (-self)

    def __mul__(self, other):
        if isinstance(other, (int, Fraction)):
            value = _rational(other)
            terms = {word: coefficient * value for word, coefficient in self._terms.items()} if value else {}
            return self._from_ordered(self.rank, terms)
        other = self._as_element(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_ordered(
            self.rank, multiply_ordered_pbw(self.rank, self._terms, other._terms)
        )

    def __rmul__(self, other):
        if isinstance(other, (int, Fraction)):
            return self * other
        return NotImplemented

    def __pow__(self, exponent):
        exponent = index(exponent)
        if exponent < 0:
            raise ValueError("exponent must be nonnegative")
        result = self.scalar(self.rank, 1)
        power = self
        while exponent:
            if exponent & 1:
                result = result * power
            exponent //= 2
            if exponent:
                power = power * power
        return result
