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

