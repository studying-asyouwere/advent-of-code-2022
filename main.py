with open('input.txt', 'r') as file:
    data = file.read()

fuels = { '=': -2, '-': -1, '0': 0, '1': 1, '2': 2 }
decimals = dict(map(reversed, fuels.items()))

numbers = []

def to_decimal(number):
    result = sum([(5 ** ii) * fuels[c] for ii, c in enumerate(reversed(number))])
    return result

def to_fuel(number):
    value = []

    while number > 0:
        remainder = number % 5
        if remainder > 2:
            number += remainder
            value.append(decimals[remainder - 5])
        else: 
            value.append(str(remainder))

        number //= 5

    return ''.join(reversed(value))

for line in data.splitlines():
    numbers.append(to_decimal(line))

snafu = to_fuel(sum(numbers))

print('The SNAFU number to supply to the console is: ' + snafu)



