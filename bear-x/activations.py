"""
We are gonna start with simple
and powerful activation function.
Later on I am gonna add more functions
"""


def relu(n):
    if n < 0:
        return 0
    else:
        return n
