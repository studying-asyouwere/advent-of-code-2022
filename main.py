with open('input.txt', 'r') as file:
    data = file.read().strip()

# construct the cave layout first
def interpolate(start, end): # structure interpolation
    x1, y1 = start
    x2, y2 = end

    if x1 == x2: # horizontal struct
        for y in range(min(y1, y2), max(y1, y2) + 1): # horizontal line
            yield x1, y

    if y1 == y2: # vertical struct
        for x in range(min(x1, x2), max(x1, x2) + 1): # vertical line
            yield x, y1

def parse(data): # parsing fun
    grid = dict()
    bottom = 0

    for line in data.splitlines():
        coords = line.split(' -> ') # extract coords from each line
        coords = [tuple(int(c) for c in coord.split(',')) for coord in coords] # list of tuples of coordinates

        for start, end in zip(coords, coords[1:]): # go through each structure 
            for pos in interpolate(start, end): # get all the positions of the struct
                grid[pos] = '#'
                if pos[1] > bottom:
                    bottom = pos[1] # rising up the bottom due to struct

    return grid, bottom

# define what happens when sand drops
def drop_sand(grid, src, bottom):
    x, y = src

    while y < bottom:
        for move in [(x, y + 1), (x - 1, y + 1), (x + 1, y + 1)]: # implement stone drop logic
            if move not in grid: # stay within grid
                x, y = move
                break
        else:
            return True, (x, y) # return has stopped boolean and position

    return False, (x, y)

# Now we are ready to simulate the falling sand.
def part1(grid):
    grid, bottom = grid
    grid = grid.copy()

    src = (500, 0)
    cnt = 0

    while True:
        has_stopped, pos = drop_sand(grid, src, bottom)
        if not has_stopped:
            break
        grid[pos] = 'o' # sand
        cnt += 1

    return cnt

# There is no more abyss
def part2(grid):
    grid, bottom = grid

    src = (500, 0)
    cnt = 0

    while True:
        _, pos = drop_sand(grid, src, bottom + 1)
        grid[pos] = 'o' # sand
        cnt += 1
        if pos == src:
            break

    return cnt

print('Part 1: the units of sand to come to rest before sand starts flowing into the abyss is: ' + str(part1(parse(data))))
print('Part 2: the units of sand to come to rest before sand starts flowing into the abyss is: ' + str(part2(parse(data))))

