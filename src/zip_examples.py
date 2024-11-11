# Just looking at iterator behavior using zip
from pypractice import tribble

numbers1 = [1, 2, 3]
numbers2 = [4, 5, 6]
letters = ['a', 'b', 'c']
fruits = ['apple', 'blueberry', 'canteloupe']

zipped = zip(numbers1, letters, fruits)
list(zipped)


[x + y for x in numbers1 for y in numbers2]
[x + y for x, y in zip(numbers1, numbers2)]
[print(f'{x} is for {y}') for x, y in zip(letters, fruits)]

tribble(
    ["x", "y", "z"],
    1,    4.5, 6.2,
    2,    4.2,  88,
    3,    3.7, 10.2
)