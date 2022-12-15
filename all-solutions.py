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

        

















