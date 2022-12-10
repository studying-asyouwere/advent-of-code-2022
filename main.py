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





