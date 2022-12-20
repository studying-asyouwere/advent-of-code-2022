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








