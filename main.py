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





