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