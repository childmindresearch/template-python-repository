import pytest

from APP_NAME import algorithms


def test_fibonacci_success_0():
    """Test that fibonacci() returns the correct value for valid input."""

    output = algorithms.fibonacci(0)

    assert output == 0


def test_fibonacci_success_18():
    """Test that fibonacci() returns the correct value for valid input."""

    output = algorithms.fibonacci(19)

    assert output == 4181


def test_fibonacci_negative():
    """Test that fibonacci() raises an exception for negative input."""

    with pytest.raises(ValueError):
        algorithms.fibonacci(-1)


def test_fibonacci_non_integer():
    """Test that fibonacci() raises an exception for non-integer input."""

    with pytest.raises(ValueError):
        algorithms.fibonacci(3.14)
