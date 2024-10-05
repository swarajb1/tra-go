def is_sum_digits_8(n):
    """This function returns True if the sum of the digits of n is 4."""

    sum_1 = sum(int(digit) for digit in str(n))

    if sum_1 > 10:
        sum_1 = sum(int(digit) for digit in str(sum_1))

    return sum_1 == 8


for i in range(256, 1024):
    if is_sum_digits_8(i):
        print(i)
