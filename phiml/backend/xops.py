"""
Extra operators.

This module is an extension to the built-in operators.
"""


class ExtraOperator(Exception):
    pass


def save_div(numerator, denominator):
    raise ExtraOperator


def gamma_inc_l(a, x):
    raise ExtraOperator


def gamma_inc_u(a, x):
    raise ExtraOperator


def arctan2(tan, divide_by):
    raise ExtraOperator


def minimum(x1, x2):
    raise ExtraOperator


def maximum(x1, x2):
    raise ExtraOperator
