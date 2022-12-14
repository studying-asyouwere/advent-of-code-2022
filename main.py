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

