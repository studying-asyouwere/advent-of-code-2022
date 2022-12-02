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







