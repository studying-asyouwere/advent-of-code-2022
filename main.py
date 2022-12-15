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

        





