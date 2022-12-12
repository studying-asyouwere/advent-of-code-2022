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


