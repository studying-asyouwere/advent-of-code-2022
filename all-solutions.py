# Day 1

with open('input.txt') as file:
    file.seek(0)
    elf_pockets = file.read()

elf_pockets_split = elf_pockets.split("\n\n")

food_calories = []
for elf_pocket in elf_pockets_split:
    elf_pocket_str = elf_pocket.splitlines()
    food_item_int = []
    for food_item_str in elf_pocket_str:
        food_item_int.append(int(food_item_str))
    food_calories.append(sum(food_item_int))

food_calories.sort(reverse = True)

print('The elf carrying the most Calories is carrying: ' + str(sum(food_calories[:1])) + ' Calories.')
print('The top three elves carrying the most Calories are carrying: ' + str(sum(food_calories[:3])) + ' Calories.')

# Day 2

with open('input.txt') as file:
    file.seek(0)
    games = file.read()

games = games.split("\n")
games = games[:-1]

# First strategy
total_score_first = 0
for game in games:
    game_score = 0
    opponent_hand = game[0]
    my_hand = game[2]
    if my_hand == 'X':
        game_score += 1
        if opponent_hand == 'A':
            game_score += 3
        elif opponent_hand == 'B':
            game_score += 0
        elif opponent_hand == 'C':
            game_score += 6
    elif my_hand == 'Y':
        game_score += 2
        if opponent_hand == 'A':
            game_score += 6
        elif opponent_hand == 'B':
            game_score += 3
        elif opponent_hand == 'C':
            game_score += 0
    elif my_hand == 'Z':
        game_score += 3
        if opponent_hand == 'A':
            game_score += 0
        elif opponent_hand == 'B':
            game_score += 6
        elif opponent_hand == 'C':
            game_score += 3
    total_score_first += game_score
    
# Second strategy
total_score_second = 0
for game in games:
    game_score = 0
    opponent_hand = game[0]
    my_strat = game[2]
    if my_strat == 'X':
        if opponent_hand == 'A':
            game_score += 3
        elif opponent_hand == 'B':
            game_score += 1
        elif opponent_hand == 'C':
            game_score += 2
    elif my_strat == 'Y':
        game_score += 3
        if opponent_hand == 'A':
            game_score += 1
        elif opponent_hand == 'B':
            game_score += 2
        elif opponent_hand == 'C':
            game_score += 3
    elif my_strat == 'Z':
        game_score += 6
        if opponent_hand == 'A':
            game_score += 2
        elif opponent_hand == 'B':
            game_score += 3
        elif opponent_hand == 'C':
            game_score += 1
    total_score_second += game_score

print('The total score of the game using the first strategy is: ' + str(total_score_first))
print('The total score of the game using the second strategy is: ' + str(total_score_second))

# Day 3

with open('input.txt', 'r') as file:
    data = file.read().strip()

rucksacks = data.split('\n')

priority_q1 = 0
for rs in rucksacks:
    L = len(rs)
    comp1 = rs[:L // 2]
    comp2 = rs[L // 2:]
    for item in comp1:
        if item in comp2:
            value = ord(item) - 96 if item.islower() else ord(item) - 64 + 26
            priority_q1 += value
            break

print('The priority for the first task is: ' + str(priority_q1))

ii = 0
priority_q2 = 0
while ii < len(rucksacks):
    bag1 = rucksacks[ii]
    bag2 = rucksacks[ii + 1]
    bag3 = rucksacks[ii + 2]
    for item in bag1:
        if item in bag2 and item in bag3:
            value = ord(item) - 96 if item.islower() else ord(item) - 64 + 26
            priority_q2 += value
            ii += 3
            break

print('The priority for the second task is: ' + str(priority_q2))

# Day 4

with open('input.txt', 'r') as file:
    data = file.read().strip()

sections = data.split('\n')

n_overlap = 0
for s in sections:
    s_split = s.split(',')
    first_sect_start, first_sect_end = s_split[0].split('-')
    second_sect_start, second_sect_end = s_split[1].split('-')
    if (int(first_sect_start) <= int(second_sect_start) and int(second_sect_end) <= int(first_sect_end)) or (int(second_sect_start) <= int(first_sect_start) and int(first_sect_end) <= int(second_sect_end)): 
        n_overlap += 1

print(n_overlap)

n_overlap = 0
for s in sections:
    s_split = s.split(',')
    first_sect_start, first_sect_end = map(int, s_split[0].split('-'))
    second_sect_start, second_sect_end = map(int, s_split[1].split('-'))
    if set(range(first_sect_start, first_sect_end + 1)) & set(range(second_sect_start, second_sect_end + 1)): 
        n_overlap += 1

print(n_overlap)

# Day 5

import re, copy

with open('input.txt', 'r') as file:
    stack_txt, instruction_data = file.read().split('\n\n')
    stack_txt = stack_txt.split('\n')
    instruction_data = instruction_data.split('\n')

stack_last = stack_txt.pop()

stack = {}
loc = {}
ordering = []
for ii in range(len(stack_last)):
    if stack_last[ii] != ' ':
        stack[stack_last[ii]] = []
        loc[stack_last[ii]] = ii
        ordering.append(stack_last[ii])

for line in reversed(stack_txt):
    for key in loc.keys():
        if line[loc[key]] != ' ':
            stack[key].append(line[loc[key]])

stack2 = copy.deepcopy(stack)

for line in instruction_data:
    if 'move' in line:
        inst_values = re.findall(r'(\d+)', line)
        count = int(inst_values[0])
        ff = inst_values[1]
        tt = inst_values[2]

        for ii in range(count):
            pop_val = stack[ff].pop()
            stack[tt].append(pop_val)

        stack2[tt] += stack2[ff][-count:]
        stack2[ff] = stack2[ff][:-count]

print('The answer for the first part is:')
for ii in ordering:
    print(stack[ii][-1], end = '')
print()

print('The answer for the second part is:')
for ii in ordering:
    print(stack2[ii][-1], end = '')

# Day 6

# read in the data
with open('input.txt', 'r') as file:
    data = file.read()

# define the detector function
def detector(data: str, size: int):
    # data: is the signal data
    # size: is the length of the expected marker we are searching for
    ell = len(data) # length of the data
    for ii in range(ell - size + 1): # go through all the marker possibilities
        s = set(data[ii:ii + size]) # get unique elements in this marker possibility
        if len(s) == size: # if all the elements are unique
            return ii + size
    return print('No marker found!')

# print the solutions
print('The answer for the first question is: ' + str(detector(data, 4)))
print('The answer for the first question is: ' + str(detector(data, 14)))

# Day 7

import os

with open('input.txt', 'r') as file:
    data = file.read().strip().split('\n')

# initialise dicts
subdirs = {} # store the subdirectories here
direct_directory_size = {} # store the directory sizes here (excluding all the subdirectories)

# extracting the folder structure
for line in data:
    if line[0] == '$':
        ds, cmd, *ddir = line.split()
        if cmd == 'cd':
            path = ddir[0]
            if path == '/': # for the top directory
                curdir = path # this is the current directory
            else:
                curdir = os.path.normpath(os.path.join(curdir, path))
            if curdir not in subdirs: # if we have not seen our current directory before
                subdirs[curdir] = []
                direct_directory_size[curdir] = 0

    else: # if we get 'dir' or numbers instead of '$'
        fsize, fname = line.split()
        if fsize != 'dir': # if we get numbers
            direct_directory_size[curdir] += int(fsize)
        else: # if we get a new directory
            subdirs[curdir].append(os.path.normpath(os.path.join(curdir, fname)))

# size calculation function
def compute_dirsize(dirname: str):
    dirsize = direct_directory_size[dirname] # initialise the size by the size of the directory excluding the subdirectories
    for ii in subdirs[dirname]: # go through all the subdirectories
        if ii in subdirs:
            dirsize += compute_dirsize(ii) # looping the subdirectory size addition until we no longer find the subdirectories
    return dirsize

# Part 1: Sum of all the directories with a total size of at most 100000.
sol_pt1 = 0
for ii in subdirs:
    dirsize = compute_dirsize(ii)
    if dirsize <= 100000:
        sol_pt1 += dirsize
print('The answer to the first questions is: ' + str(sol_pt1))

# Part 2: The smallest possible file to delete to free up the required space.
total_space = 70000000
space_required = 30000000
space_used = compute_dirsize('/')

delete_this_directory = total_space
for ii in direct_directory_size:
    dirsize = compute_dirsize(ii)
    if dirsize >= space_required - (total_space - space_used) and dirsize <= delete_this_directory:
        delete_this_directory = dirsize
print('The answer to the first questions is: ' + str(delete_this_directory))

# Day 8

import numpy as np

# put the input file in a matrix format
grid = np.array([list(x.strip()) for x in open('input.txt')], int)

# get number of rows and columns
nrow, ncol = np.shape(grid)

best_view_score = 1 # initialising the best view score (at least one for any tree)
n_tree_visible = ncol * 2 + (nrow - 2) * 2 # initialising visible trees (trees on the edge)
for ii in range(1, nrow - 1): # go through all the rows inside
    for iii in range(1, ncol - 1): # go through all the columns inside
        tree = grid[ii, iii] # tree of interest
        tree_up = grid[:ii, iii] # trees up
        tree_down = grid[ii + 1:, iii] # trees down
        tree_right = grid[ii, iii + 1:] # trees right
        tree_left = grid[ii, :iii] # trees left

        # compute the tallest tree in each direction
        tallest_tree_up = max(tree_up)
        tallest_tree_down = max(tree_down)
        tallest_tree_right = max(tree_right)
        tallest_tree_left = max(tree_left)

        # count the number of trees visible looking up
        count_visible_tree_up = 0 # initialising
        for tt in range(len(tree_up)):
            count_visible_tree_up += 1
            if tree_up[len(tree_up) - 1 - tt] >= tree: # if we get to the tree that is the same height or taller we break
                break

        # count the number of trees visible looking down
        count_visible_tree_down = 0 # initialising
        for tt in range(len(tree_down)):
            count_visible_tree_down += 1
            if tree_down[tt] >= tree: # if we get to the tree that is the same height or taller we break
                break

        # count the number of trees visible looking right
        count_visible_tree_right = 0 # initialising
        for tt in range(len(tree_right)):
            count_visible_tree_right += 1
            if tree_right[tt] >= tree: # if we get to the tree that is the same height or taller we break
                break

        # count the number of trees visible looking left
        count_visible_tree_left = 0 # initialising
        for tt in range(len(tree_left)):
            count_visible_tree_left += 1
            if tree_left[len(tree_left) - 1 - tt] >= tree: # if we get to the tree that is the same height or taller we break
                break

        # if tree is taller than the maxes in any directions we can see the tree
        if tree > tallest_tree_up or tree > tallest_tree_down or tree > tallest_tree_right or tree > tallest_tree_left:
            n_tree_visible += 1

        # if the new view score is higher we overwrite the best score
        view_score = count_visible_tree_up * count_visible_tree_down * count_visible_tree_right * count_visible_tree_left
        if view_score > best_view_score:
            best_view_score = view_score

print('Part 1: number of visible trees is: ' + str(n_tree_visible))
print('Part 2: the best view score is is: ' + str(best_view_score))

# Day 9

def make_moves(moves, rope_len):
    # initialise 
    xs = [0] * rope_len
    ys = [0] * rope_len
    visited = { (xs[-1], ys[-1]) }

    # go through all the moves and record the visited spaces
    for (mx, my), distance in moves:
        for _ in range(distance):
            # make moves on x and y
            xs[0] += mx
            ys[0] += my
            for ii in range(rope_len - 1):
                # distance created with the move
                dx = xs[ii + 1] - xs[ii] 
                dy = ys[ii + 1] - ys[ii]
                if abs(dx) == 2 or abs(dy) == 2: # bring this up to speed diagonally 
                    xs[ii + 1] = xs[ii] + int(dx / 2) 
                    ys[ii + 1] = ys[ii] + int(dy / 2)
            # add on the visitied spot
            visited.add( (xs[-1], ys[-1]) )

    # return the number of visited spots 
    return len(visited)

# define the directions
dirs = {'L': (-1, 0), 'R': (1, 0), 'D': (0, -1), 'U': (0, 1)}

# define the moves
moves = [(dirs[line[0]], int(line[1:])) for line in open('input.txt')]

# print results
print("Part 1: the number of positions the tail visited is: " + str(make_moves(moves, 2)))
print("Part 2: the number of positions the tail visited is: " + str(make_moves(moves, 10)))

# Day 10

with open('input.txt', 'r') as file:
    data = file.read().strip().split('\n')

x = 1 # initialising x
x_list = [x] # store the progress here
for line in data: # go line by line
    if 'add' in line: # if we are told to add
        x_list.extend([x, x]) # add two current x
        x += int(line[5:]) # add onto x
    else: # if we get noop
        x_list.append(x)

signal_strength = sum(x_list[cycle] * cycle for cycle in range(20, len(x_list), 40))

print('The sum of the signal strengths is: ' + str(signal_strength))

for yy in range(6): # go over the rows
    crt_line = ''
    for xx in range(40): # go over the columns
        cycle = xx + yy * 40 # cycle number
        crt_line += '.' if abs(xx - x_list[cycle + 1]) <= 1 else ' ' 
    print(crt_line)

# Day 11

import re

with open('input.txt', 'r') as file:
    data = file.read().strip().split('\n\n')

# define monkey behaviours
start_items = {}
operation = {}
div_test = {}
div_test_true = {}
div_test_false = {}
n_handle_items = {}

# loop through monkey data and populate
m_number = 0 # monkey number init
modd = 1 # division booster val
for monkey in data:
    monkey_data = monkey.split('\n')
    n_handle_items[m_number] = 0 # number of handled items init
    for mdata in monkey_data:
        if 'Operation' in mdata: # operation
            operation[m_number] = mdata[23:]
        elif 'Starting' in mdata: # starting items
            start_items[m_number] = re.findall(r'\d+', mdata)
        elif 'Test' in mdata: # division test
            div_test[m_number] = int(re.findall(r'\d+', mdata)[0])
        elif 'true' in mdata: # if test pass
            div_test_true[m_number] = int(re.findall(r'\d+', mdata)[0])
        elif 'false' in mdata: # if test fail
            div_test_false[m_number] = int(re.findall(r'\d+', mdata)[0])
    modd *= div_test[m_number]
    m_number += 1 # count up monkey number
   

# process the rounds and compute monkey business
n_round = 10000 # number of rounds
live_items = start_items # store the live item updates here
for ii in range(n_round): # for each round
    for mm in range(m_number): # go through each monkey
        # get monkey info
        worry_op = operation[mm] # monkey worry operation
        div_test_val = div_test[mm] # value for division test
        div_test_true_val = div_test_true[mm] # value for division test true
        div_test_false_val = div_test_false[mm] # value for division test fail
        # compute new worry due to monkey
        for tt in live_items[mm]: # go through each item monkey has
            if '+' in worry_op and 'old' not in worry_op: # + int
                tt_new = int(tt) + int(re.findall(r'\d+', worry_op)[0]) 
            elif '+' in worry_op and 'old' in worry_op: # + old
                tt_new = int(tt) * 2
            elif '*' in worry_op and 'old' not in worry_op: # * int
                tt_new = int(tt) * int(re.findall(r'\d+', worry_op)[0]) 
            elif '*' in worry_op and 'old' in worry_op: # * old
                tt_new = int(tt) ** 2
            # monkey has stopped playing!
            # tt_new //= 3 # only relevant for part 1
            tt_new %= modd # for part 2 booster
            # where will monkey throw?
            if tt_new % div_test_val == 0:
                live_items[div_test_true_val].append(tt_new)
            else: 
                live_items[div_test_false_val].append(tt_new)
            # update n handled items and empty monkey's hand
            n_handle_items[mm] += len(live_items[mm])
            live_items[mm] = []

# Get the monkey business
n_handles = list(n_handle_items.values())
n1 = max(n_handles)
n_handles.remove(n1)
n2 = max(n_handles)
monkey_business = n1 * n2
print('Part 1: monkey business after 20 rounds is: ' + str(monkey_business))

# Day 12

with open('input.txt', 'r') as file:
    data = file.read()

def can_go(g, p1, p2):
    return(p2 in g and 
    ((g[p1] == 'E' and g[p2] in 'yz') or 
    (g[p2] == 'S' and g[p1] in 'ab') or 
    (g[p2] != "S" and g[p1] != "E" and ord(g[p1]) - ord(g[p2]) <= 1)))


# defining grid
grid = {x + y * 1j: h for y, line in enumerate(data.split('\n'))
                        for x, h in enumerate(line)}

# find the starting and ending position
start = [p for p, h in grid.items() if h == 'S'][0]
end = [p for p, h in grid.items() if h == 'E'][0]

# starting from the 'end' find distances to get to 'start'
distance = {end: 0} # init dist dict from end
queue = [end] # init place we can get to
while queue: # keep walking around as long as we can get somewhere
    p1 = queue.pop(0) # we are here
    for p2 in [p1 - 1, p1 + 1, p1 + 1j, p1 - 1j]: # options around us
        if p2 not in distance and can_go(grid, p1, p2): # if we haven't been here before and we can get there  
            distance[p2] = distance[p1] + 1 # record distance from the end
            queue.append(p2) # add it to our places to go

# Part 2 - find the position in distance for S or any a
short_dist = sorted(distance[p] for p in distance if grid[p] in "Sa")[0]            

print('Part 1: the shortest distance from the start to the end is: ' + str(distance[start]))
print('Part 2: the shortest distance from the start to the end is: ' + str(short_dist))

# Day 13

from functools import cmp_to_key

with open('input.txt', 'r') as file:
    data = file.read().split('\n\n')

# lambdas to be used in the function
I = lambda x:isinstance(x, int)
L = lambda x:isinstance(x, list)

# funtion to compare left and right
def cmp(l, r):
    if I(l) and I(r): # comparing integers
        if l < r: return -1
        return l > r
    if L(l) and L(r): # comparing lists
        for ii in range(min(len(l), len(r))):
            c = cmp(l[ii], r[ii])
            if c: return c
        return cmp(len(l), len(r))
    if I(l) and L(r): # comparing int and list
        return cmp([l], r)
    if L(l) and I(r): # comparing list and int
        return cmp(l, [r])

p = [] # init item storage
n = 0 # initialise the sum
for ii, ss in enumerate(data):
    l, r = [eval(x) for x in ss.split()] # split up left and right values 
    if cmp(l, r) <= 0: n += ii + 1
    p.append(l); p.append(r)

p.append([[2]]); p.append([[6]])

p.sort(key = cmp_to_key(cmp))

print("Part 1: the sum is " + str(n))

print("Part 2: the decoder key for the distress signal is: " + str( (p.index([[2]]) + 1) * (p.index([[6]]) + 1) ))

# Day 14

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

# Day 15

import re

y_spot = 2000000 
ys = set()
beacons_y = set()

with open("input.txt", 'r') as file:
    for ll in file.readlines():
        # get sensor and corresponding closest beacon coordinates
        sx, sy, bx, by = map(int, re.findall(r'(?<=\=)(.*?)(?=,|\:|\n)', ll))

        if by == y_spot: # if the beacon is in the specified row
            beacons_y.add(bx)

        # compute the threshold distance
        d = abs(bx - sx) + abs(by - sy)
        d -= abs(y_spot - sy)

        # add anything within the range
        for x in range(sx - d, sx + d + 1):
            ys.add(x)

print('Part 1: the solution is: ' + str(len(ys - beacons_y)))

mx = 4000000
y_ranges = [[] for _ in range(mx + 1)]
with open("input.txt", 'r') as file:
    for ll in file.readlines():
        # get sensor and corresponding closest beacon coordinates
        sx, sy, bx, by = map(int, re.findall(r'(?<=\=)(.*?)(?=,|\:|\n)', ll))

        # get potential y ranges and corresponding x ranges
        d = abs(bx - sx) + abs(by - sy)
        dy = 0
        while d > 0: # will countdown the distance
            xl = max(0, sx - d)
            xr = min(mx, sx + d)
            if (sy - dy >= 0):
                y_ranges[sy - dy].append([xl, xr])
            if (sy + dy <= mx and dy):
                y_ranges[sy + dy].append([xl, xr])
            dy += 1
            d -= 1

    # the answer for y is in there somewehre
    for ans_y in range(mx + 1):
        xs = y_ranges[ans_y]
        if not xs:
            continue
        xs.sort()

        if xs[0][0] != 0:
            ans_x = 0
            break

        # sorry guys -- bit tipsy right now :D 
        last_e = xs[0][1]
        for ii in range(1, len(xs)):
            if last_e >= xs[ii][0] - 1:
                last_e = max(last_e, xs[ii][1])
            else:
                break

        if last_e != mx:
            ans_x = last_e + 1
            break

# can you hear my computer? running like crazy right now....
print('Part 2: the solution is: ' + str(mx * ans_x + ans_y))   

# Day 16

import collections as c, itertools, re, functools

r = r'Valve (\w+) .*=(\d*); .* valves? (.*)'

V, F, D = set(), dict(), c.defaultdict(lambda: 1000)

for v, f, us, in re.findall(r, open('input.txt').read()):
    V.add(v) # store node here
    if f != '0': F[v] = int(f) # stroe flow here
    for u in us.split(', ') : D[u, v] = 1

for k, i, j in itertools.product(V, V, V):
    D[i, j] = min(D[i, j], D[i, k] + D[k, j])

@functools.cache
def search(t, u = 'AA', vs = frozenset(F), e = False):
    tt = max([F[v] * (t - D[u, v] - 1) + search(t - D[u, v] - 1, v, vs - {v}, e)
    for v in vs if D[u, v] < t] + [search(26, vs = vs) if e else 0])
    return tt

print('Part 1: the most pressure you can release is: ' + str(search(30)))
print('Part 2: the most pressure you and the elephant can release is: ' + str(search(26, e = True)))

# Day 17
import collections

def read_input(path: str = 'input.txt'):
    with open(path) as file:
        pattern = file.read().rstrip()
    return pattern

# pattern generator fun
def pattern_generator(pattern: str):
    idx = 0
    n = len(pattern)
    while True:
        yield pattern[idx], idx
        idx = (idx + 1) % n

# Rock class
class Rock:

    def __init__(self, rock_type: str, x_offset: int, y_offset: int = 2):

        # rock tye init ("-", "+", "L", "I", "o")
        if rock_type == '-':
            self.positions = [[x, y_offset] for x in range(x_offset, x_offset + 4)]
        elif rock_type == '+':
            self.positions = [[x_offset + 1, y_offset + 2], [x_offset, y_offset + 1], [x_offset + 1, y_offset + 1],
            [x_offset + 2, y_offset + 1], [x_offset + 1, y_offset]]
        elif rock_type == 'L':
            self.positions = [[x_offset + 2, y_offset + 2], [x_offset + 2, y_offset + 1]] \
                + [[x, y_offset] for x in range(x_offset, x_offset + 3)]
        elif rock_type == 'I':
            self.positions = [[x_offset, y] for y in range(y_offset + 3, y_offset - 1, -1)]
        elif rock_type == 'o':
            self.positions = [[x_offset, y_offset + 1], [x_offset + 1, y_offset + 1], [x_offset, y_offset],
            [x_offset + 1, y_offset]]
        else: 
            raise NotImplementedError(f'{rock_type} is low key not recognisable.')

        # rock type
        self.type = rock_type

    # fun to return the position in tuple
    def get_positions(self) -> list[tuple[int]]:
        return [tuple(position) for position in self.positions]

    # fun to move the rock
    def move(self, dx, dy):
        for coords in self.positions: 
            coords[0] += dx
            coords[1] += dy

# Cave class
class Cave:
    
    def __init__(self, wind_pattern: str, rock_pattern: str = '-+LIo', width = 7):
        self.width = width
        self.occupied = set()
        self.highetst = -1
        self.falling_rock = None
        self.wind_generator = pattern_generator(wind_pattern)
        self.rock_type_generator = pattern_generator(rock_pattern)
        self.rock_counter = 0
        self.state_cache = collections.defaultdict(list) # store the state here for cycle detection

    # spawn rocks down the cave
    def _spawn_rock(self):
        self.falling_rock = Rock(next(self.rock_type_generator)[0], 2, self.highetst + 4)

    # place rocks
    def _place_rock(self, rock: Rock):
        # rock position
        positions = rock.get_positions()

        # latest cave height
        self.highetst = max(self.highetst, *[position[1] for position in positions])

        # latest occupied rocks
        self.occupied.update(positions)

        # rock counter up 
        self.rock_counter += 1

        # no more falling rock
        self.falling_rock = None

    # has the rock colllided in the cave
    def _check_rock_collision(self, rock: Rock):
        return any(self._check_collision(*position) for position in rock.get_positions())

    def _check_collision(self, x: int, y:int):
        return x < 0 or x > self.width - 1 or y < 0 or (x, y) in self.occupied

    # check if a row is completely blocked in the tetris style
    def highest_row_blocked(self):
        return all((x, self.highest) in self.occupied for x in range(0, self.width))

    # fun to detect cycles
    def detect_cycles(self) -> tuple[tuple[int, int], int, int]:
        cycles = []

        for value in self.state_cache.values():
            if len(value) > 1:
                cycles = value
                break

        if not cycles:
            return (-1, -1), -1, -1

        return cycles[0], cycles[1][0] - cycles[0][0], cycles[1][1] - cycles[0][1]

    # get wind direction from strings
    @staticmethod
    def _get_direction_from_string(direction: str, inverse: bool = False):

        # translate the direction if inverse
        if inverse: 
            if direction == '>':
                direction = '<'
            elif direction == '<':
                direction = '>'
            elif direction == 'v':
                direction = '^'
            elif direction == '^':
                direction = 'v'
            else:
                raise NotImplementedError

        # go through the cases
        if direction == '>':
            dx = 1
            dy = 0
        elif direction == '<':
            dx = -1
            dy = 0
        elif direction == 'v':
            dx = 0
            dy = -1
        elif direction == '^':
            dx = 0
            dy = 1
        else: 
            raise NotImplementedError

        return dx, dy

    def _get_surface_profile(self):
        # the first element blocking each column
        profile = []
        for x in range(0, self.width):
            y = self.highetst
            while not self._check_collision(x, y):
                y -= 1
            profile.append(self.highetst - y)
        return tuple(profile)

    # implement the steps
    def step(self):
        # get the current wind
        wind, wind_idx = next(self.wind_generator)

        # init placed
        placed = False

        # do we have a falling rock or should one start fallling
        if not self.falling_rock:
            # spawn rock
            self._spawn_rock()
            # create unique key of the state for cycle detection
            key = (*self._get_surface_profile(), wind_idx, self.falling_rock.type)
            # append the key
            self.state_cache[key].append((self.rock_counter, self.highetst))


        # apply some wind
        self.falling_rock.move(*self._get_direction_from_string(wind))

        # has the rock collided?
        if self._check_rock_collision(self.falling_rock):
            self.falling_rock.move(*self._get_direction_from_string(wind, inverse = True))

        # apply rock fall motion
        self.falling_rock.move(*self._get_direction_from_string('v'))

        # check whether the rock collided in this move
        if self._check_rock_collision(self.falling_rock):
            self.falling_rock.move(*self._get_direction_from_string('v', inverse = True))
            self._place_rock(self.falling_rock)
            placed = True

        return placed

def part1(rocks = 2022):
    pattern = read_input()
    cave = Cave(pattern)
    while cave.rock_counter < rocks:
        cave.step()
    print(f'Part 1: the height after 2022 rocks is: ' + str(cave.highetst + 1))

def part2(target = 1000000000000):
    pattern = read_input()
    cave = Cave(pattern)

    # make these rocks fall to detect cycles
    rocks = 3000
    heights = []
    while cave.rock_counter < rocks:
        if cave.step():
            heights.append(cave.highetst)

    # detect cycles
    (cycle_start, highest_start), cycle_size, height_per_cycle = cave.detect_cycles()
    assert cycle_size != -1, 'No cycle...'
    
    # height before the first cycle
    result = heights[cycle_start] 

    # how many times do cycles get repeated
    cycle_number, rest = divmod(target - cycle_start, cycle_size)

    # compute the final result
    result += cycle_number * height_per_cycle + (heights[cycle_start + rest] - heights[cycle_start])
    print(f'Part 2: the height after 1000000000000 rocks is: ' + str(result))

if __name__ == '__main__':
    part1()
    part2()

# day 18

from collections import deque

Cubes = []
with open('input.txt', 'r') as file:
    for ll in file:
        Line = ll.strip()
        x, y, z = list(map(int, Line.split(",")))
        coord_tuple = (x, y, z)
        Cubes.append(coord_tuple)

CubesSet = set(Cubes)

AdjCoords = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, -1), (0, -1, 0), (-1, 0, 0)]

total_sa = 0 # init surface area
for x, y, z in Cubes:
    default_sa = 6
    for dx, dy, dz in AdjCoords:
        nx, ny, nz = x + dx, y + dy, z + dz
        if (nx, ny, nz) in CubesSet: # if there is the neighbour droplet
            default_sa -= 1
    total_sa += default_sa

print('Part 1: the total surface area is: ' + str(total_sa))

# let us figure out the boundaries
minx, miny, minz = 50, 50, 50
maxx, maxy, maxz = 0, 0, 0
for x, y, z in Cubes: 
    if x > maxx:
        maxx = x
    if x < minx:
        minx = x
    if y > maxy:
        maxy = y
    if y < miny:
        miny = y
    if z > maxz:
        maxz = z
    if z < minz:
        minz = z

# add the first boundary cube
Cube1 = (minx - 1, miny - 1, minz - 1)
ExtCubes = []
ExtCubesSet = set()
ExtCubesSet.add(Cube1)

QueueCubes = deque()
QueueCubes.append(Cube1)
while QueueCubes:
    # get next cube in queue
    NextCube = QueueCubes.popleft()
    x, y, z = NextCube

    # add it as an external cube
    ExtCubes.append(NextCube)

    # test the "boundariness" of the cubes
    for dx, dy, dz in AdjCoords:
        nx, ny, nz = x + dx, y + dy, z + dz
        if nx < minx - 1 or nx > maxx + 1 or ny < miny - 1 or ny > maxy + 1 or nz < minz - 1 or nz > maxz + 1:
            continue
        ThisCube = (nx, ny, nz)
        if ThisCube in CubesSet or ThisCube in ExtCubesSet:
            continue
        QueueCubes.append(ThisCube)
        ExtCubesSet.add(ThisCube)

total_ext_sa = 0 # init total external surface area
for x, y, z in ExtCubes:
    default_sa = 0
    for dx, dy, dz in AdjCoords:
        nx, ny, nz = x + dx, y + dy, z + dz
        if (nx, ny, nz) in CubesSet: # if there is a cube here
            default_sa += 1
    total_ext_sa += default_sa

print('Part 2: the total exterior surface area is: ' + str(total_ext_sa))

# day 19

import re

with open('input.txt', 'r') as file:
    data = file.readlines()
    bps = [bp.strip() for bp in data] # blueprints

# for every line extract integers
bp_int = list()
for bp in bps:
    ints = [int(ii) for ii in re.findall(r'\d+', bp)]
    bp_int.append(ints)

def qass(state): # artificial quality assessment 
    minutes, (robots, inventory, mined) = state
    return 1000 * mined[3] + 100 * mined[2] + 10 * mined[1] + mined[0]

def bfs(costs, robots, num_minutes, top_queue = 30000):
    queue = list()
    queue.append((0, (robots, (0, 0, 0, 0), (0, 0, 0, 0))))
    max_geodes_mined = 0
    depth = 0

    while queue: 
        # go through the queue 
        minutes, (robots, old_inventory, mined) = queue.pop(0)

        if minutes > depth:
            queue.sort(key = qass, reverse = True)
            queue = queue[:top_queue]
            depth = minutes

        if minutes == num_minutes:
            max_geodes_mined = max(max_geodes_mined, mined[3])
            continue 

        # robots go to work
        new_inventory = tuple([old_inventory[ii] + robots[ii] for ii in range(4)])
        new_mined = tuple([mined[ii] + robots[ii] for ii in range(4)])

        # if we were not to build a robot
        queue.append((minutes + 1, (robots, new_inventory, new_mined)))

        # if we were to build a robot
        for ii in range(4): # for each type of robot
            cost_robot = costs[ii]

            # can we afford this guy??
            if all([old_inventory[jj] >= cost_robot[jj] for jj in range(4)]):
                new_robots = list(robots)
                new_robots[ii] += 1
                new_robots = tuple(new_robots)

                # money well spent
                new_inventory_state = tuple([new_inventory[jj] - cost_robot[jj] for jj in range(4)])
                queue.append((minutes + 1, (new_robots, new_inventory_state, new_mined)))

    return max_geodes_mined

max_minutes = 24 
sum_quality = 0
for bpid, cost_ore_robot, cost_clay_robot, ob_ore, obs_clay, geode_ore, geode_ob in bp_int:
    cost_per_robot = [
        (cost_ore_robot, 0, 0, 0),
        (cost_clay_robot, 0, 0, 0),
        (ob_ore, obs_clay, 0, 0),
        (geode_ore, 0, geode_ob, 0)
    ]
    num_mined = bfs(cost_per_robot, (1, 0, 0, 0), max_minutes, top_queue = 1000)

    sum_quality += num_mined * bpid

print("Part 1: the quality level is: " + str(sum_quality))

max_minutes = 32 
product_geodes = 1
for bpid, cost_ore_robot, cost_clay_robot, ob_ore, obs_clay, geode_ore, geode_ob in bp_int[:3]:
    cost_per_robot = [
        (cost_ore_robot, 0, 0, 0),
        (cost_clay_robot, 0, 0, 0),
        (ob_ore, obs_clay, 0, 0),
        (geode_ore, 0, geode_ob, 0)
    ]
    num_mined = bfs(cost_per_robot, (1, 0, 0, 0), max_minutes, top_queue = 10000)
    product_geodes *= num_mined

print("Part 2: the answer is: " + str(product_geodes))

# Day 20

from collections import deque

with open('input.txt', 'r') as file:
    data = file.read().splitlines()

data_content = deque([*map(lambda n: int(n), data)])
index_data_content = deque(range(0, length := len(data_content)))

for idx in range(length):
    position = index_data_content.index(idx)
    for deq in [data_content, index_data_content]:
        deq.rotate(position * -1)
        local_value = deq.popleft()
        if deq == data_content: current_value = local_value
        deq.rotate(current_value * -1)
        deq.appendleft(local_value)

zero = data_content.index(0)
hint1, hint2, hint3 = (
    data_content[(zero + 1000) % (len(data_content))],
    data_content[(zero + 2000) % (len(data_content))],
    data_content[(zero + 3000) % (len(data_content))]
)

print('Part 1: the sum of the three numbers that form the grove coordinates is: ' + str(sum([hint1, hint2, hint3])))

data_content = deque([*map(lambda n: int(n) * 811589153, data)])
index_data_content = deque(range(0, length := len(data_content)))

for _ in range(10):
    for idx in range(length):
        position = index_data_content.index(idx)
        for deq in [data_content, index_data_content]:
            deq.rotate(position * -1)
            local_value = deq.popleft()
            if deq == data_content: current_value = local_value
            deq.rotate(current_value * -1)
            deq.appendleft(local_value)

zero = data_content.index(0)
hint1, hint2, hint3 = (
    data_content[(zero + 1000) % (len(data_content))],
    data_content[(zero + 2000) % (len(data_content))],
    data_content[(zero + 3000) % (len(data_content))]
)

print('Part 2: the sum of the three numbers that form the grove coordinates is: ' + str(sum([hint1, hint2, hint3])))

# Day 21

with open('input.txt', 'r') as file:
    data = file.read().splitlines()

# Part 1
monkeys = {} # store monkeys here
while 'root' not in monkeys: # until we get to root
    for ll in data:
        m = ll[:4]
        if len(ll) > 8: m1, m2 = ll[6:10], ll[13:]
        try:
            if '+' in ll: monkeys[m] = monkeys[m1] + monkeys[m2] 
            elif '-' in ll: monkeys[m] = monkeys[m1] - monkeys[m2] 
            elif '*' in ll: monkeys[m] = monkeys[m1] * monkeys[m2] 
            elif '/' in ll: monkeys[m] = monkeys[m1] // monkeys[m2] 
            else: monkeys[m] = int(ll[6:])
        except (KeyError, TypeError): pass

print('Part 1: root will yell: ' + str(monkeys['root']) + "!")

# Part 2
for ll in data: monkeys[ll[:4]] = ll[6:]
del monkeys['humn']
equation = monkeys.pop('root').replace("+", "=")
while any([k in equation for k in monkeys]):
    for m in monkeys:
        if m in equation:
            equation = equation.replace(m, '(' + monkeys[m] + ')')

equation = equation.replace('=', '- (') + ')'
c = eval(equation.replace('humn', '-1j'))
r2 = round(c.real / c.imag)

print('Part 2: I will need to yell: ' + str(r2) + " !")

# Day 22

from collections import defaultdict as dd

f = open("input.txt")

inp, seq = [z.split('\n') for z in f.read().split('\n\n')]

DX = [0, 1, 0, -1]
DY = [1, 0, -1, 0]

seq = seq[0]

x, y, d = 0, 0, 0
z = 0
side = 50

n = len(inp)
m = max([len(inp[i]) for i in range(n)])

for ii in range(n):
    while len(inp[ii]) < m:
        inp[ii] += ' '

k = len(seq)

while inp[x][y] == ' ':
    y += 1

# Part 1
while z < k:
    num = seq[z]
    while z + 1 < k and '0' <= seq[z + 1] <= '9':
        z += 1
        num += seq[z]
    z += 1

    num = int(num)
    mv = 0
    while mv < num:
        dx, dy = DX[d], DY[d]
        X = (x + dx) % n
        Y = (y + dy) % m
        if inp[X][Y] == '.':
            x = X
            y = Y
            mv += 1
        elif inp[X][Y] == ' ':
            x = X
            y = Y
        else:
            break

    while inp[x][y] == ' ':
        x = (x - dx) % n
        y = (y - dy) % m

    if z < k:
        assert seq[z] in ['L', 'R']
        if seq[z] == 'L':
            d = (d - 1) % 4
        else:
            d = (d + 1) % 4
        z += 1

print('Part 1: the final password is: ' + str((x + 1) * 1000 + (y + 1) * 4 + d))

nxt = [dd(tuple) for z in range(4)]
for ii in range(n):
    for jj in range(m):
        if inp[ii][jj] == ' ':
            continue
        for z in range(4):
            try:
                nxt[z][(ii, jj)] = (ii + DX[z], jj + DY[z], z)
            except:
                continue

            if ii == side - 1 and jj >= 2 * side and z == 1:
                nxt[1][(ii, jj)] = (jj - side, 2 * side - 1, 2)
            if side <= ii < 2 * side and jj == 2 * side - 1 and z == 0:
                nxt[0][(ii, jj)] = (side - 1, ii + side, 3)

            if ii < side and jj == 3 * side - 1 and z == 0:
                nxt[0][(ii, jj)] = (3 * side - 1 - ii, 2 * side - 1, 2)
            if 2 * side <= ii < 3 * side and jj == 2 * side - 1 and z == 0:
                nxt[0][(ii, jj)] = (3 * side - 1 - ii, 3 * side - 1, 2)

            if ii == 3 * side - 1 and side <= jj <= 2 * side - 1 and z == 1:
                nxt[1][(ii, jj)] = (jj + 2 * side, side - 1, 2)
            if 3 * side <= ii and jj == side - 1 and z == 0:
                nxt[0][(ii, jj)] = (3 * side - 1, ii - 2 * side, 3)

            if side <= ii < 2 * side and jj == side and z == 2:
                nxt[2][(ii, jj)] = (2 * side, ii - side, 1)
            if ii == 2 * side and jj < side and z == 3:
                nxt[3][(ii, jj)] = (jj + side, side, 0)

            if ii < side and jj == side and z == 2:
                nxt[2][(ii, jj)] = (3 * side - ii - 1, 0, 0)
            if 2 * side <= ii < 3 * side and jj == 0 and z == 2:
                nxt[2][(ii, jj)] = (3 * side - ii - 1, side, 0)

            if ii == 0 and 2 * side <= jj < 3 * side and z == 3:
                nxt[3][(ii, jj)] = (4 * side - 1, jj - 2 * side, 3)
            if ii == 4 * side - 1 and jj < side and z == 1:
                nxt[1][(ii, jj)] = (0, jj + 2 * side, 1)

            if ii == 0 and side <= jj < 2 * side and z == 3:
                nxt[3][(ii, jj)] = (jj + 2 * side, 0, 0)
            if 3 * side <= ii and jj == 0 and z == 2:
                nxt[2][(ii, jj)] = (0, ii - 2 * side, 1)

while inp[x][y] == ' ':
    y += 1

z = 0

while z < k:
    num = seq[z]
    while z + 1 < k and '0' <= seq[z + 1] <= '9':
        z += 1
        num += seq[z]
    z += 1

    num = int(num)

    mv = 0
    while mv < num:
        X, Y, D = nxt[d][(x, y)]
        if inp[X][Y] == '.':
            x = X
            y = Y
            d = D
            mv += 1
        else:
            break

    if z < k:
        assert seq[z] in ['L', 'R']
        if seq[z] == 'L':
            d = (d - 1) % 4
        else: 
            d = (d + 1) % 4
        z += 1

print('Part 2: the final password is: ' + str((x + 1) * 1000 + (y + 1) * 4 + d))
            
# Day 23

with open('input.txt', 'r') as file:
    elves = set()
    for ii, line in enumerate(file.readlines()):
        line = line.strip()
        for jj, c in enumerate(line):
            if c == '#':
                elves.add((jj, ii))

directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

round = 0
while True:
    round += 1
    proposals = {}
    for x, y in elves:
        tiles_around = [
            (x - 1, y - 1),
            (x, y - 1),
            (x + 1, y - 1),
            (x + 1, y),
            (x + 1, y + 1),
            (x, y + 1),
            (x - 1, y + 1),
            (x - 1, y)
        ]

        elf_can_move = False
        for px, py in tiles_around:
            if (px, py) in elves:
                for dx, dy in directions:
                    pos_to_check = []
                    propose_move = True
                    for px, py in tiles_around:
                        if (dx == 0 and py - y == dy) or (dy == 0 and px - x == dx):
                            if (px, py) in elves:
                                propose_move = False
                                break
                    if propose_move:
                        if (x + dx, y + dy) in proposals:
                            proposals[(x + dx, y + dy)] = None
                        else:
                            proposals[(x + dx, y + dy)] = (x, y)
                        break
                break

    if not proposals:
        break
    for (px, py), elf in proposals.items():
        if elf:
            elves.remove(elf)
            elves.add((px, py))

    min_x = min([x for x, y in elves])
    max_x = max([x for x, y in elves])
    min_y = min([y for x, y in elves])
    max_y = max([y for x, y in elves])

    directions = directions[1:] + directions[:1]

    if round == 10:
        print('Part 1: the number of empty ground tiles is: ' + str((max_x - min_x + 1) * (max_y - min_y + 1) - len(elves)))

print('Part 2: the number of the first round where no Elf moves is: ' + str(round))























    

















