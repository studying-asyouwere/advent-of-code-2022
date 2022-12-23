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
            








