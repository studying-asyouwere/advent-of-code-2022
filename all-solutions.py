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













