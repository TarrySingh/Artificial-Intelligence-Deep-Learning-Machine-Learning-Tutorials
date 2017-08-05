########################################################
# Full exercise kit on http://www.practicepython.org
#
#
#
########################################################
# / Some cool and funny examples //
########################################################

########################################################
# Example 1 - Password Generator
########################################################
# 1 - Basic example

import random
import string

# s = string.printable
# passlen = 8
#
# p = "".join(random.sample(s,passlen))
# print(p)

# 2 - Somewhat more fun

def pw_gen(size = 8, chars=string.printable):
    return ''.join(random.choice(chars) for _ in range(size))
# print('Password is '+ pw_gen())
print('Password is '+ pw_gen(int(input('How many characters in your password?'))))

########################################################
# Reverse word order solutions
########################################################

# 1 - Simple loop
def reverse(x):
    y = x.split()
    result = []
    for word in y:
        result.insert(0,word)
    return " ".join(result)
test1 = input('Enter your sentence:' )
print(reverse(test1))

# 2 - A quick one-liner solution is like this
def reverseSentence(x):
    return ''.join(x.split()[::-1])
enter = input('Your sentence goes here: ')
print(reverseSentence(enter))


########################################################
# Example 2 : Rock paper scissors game
########################################################

import sys

user1 = input('What is your name?')
user2 = input('and your name?')
user1_answer = input('%s, do you want to choose rock, paper or scissors?' %user1)
user2_answer = input('%s, do you want to choose rock, paper or scissors?' %user2)

def compare(u1, u2):
    if u1 == u2:
        return("Tts s tie!")
    elif u1 =='rock':
        if u2 == 'scissors':
            return('Rock wins!')
        else:
            return('Paper wins!')
    elif u1 =='scissors':
        if u2 == 'paper':
            return('Scissors wins!')
        else:
            return('Rock wins!')
    elif u1 =='paper':
        if u2 == 'rock':
            return('Paper wins!')
        else:
            return('Scissors wins!')
    else:
        return("Incorrect input! You must enter rock, paper or scissors. Try one more time")
        sys.exit()

print(compare(user1_answer, user2_answer))

########################################################
# Example 3 : Tic Tac toe
########################################################

# Draw a game board

a = '---'.join('    ')
b = '   '.join('||||')
print('\n'.join((a, b, a, b, a, b, a)))

# Drawing a board
def drawboard(board):
    print ('    |   |   ')
    print ('' +board[6]+ '    | ' +board[7]+ '  |   ' +board[8] )
    print ('    |   |   ')
    print ('---------------')
    print ('    |   |   ')
    print ('' +board[3]+ '    | ' +board[4]+ '  | ' +board[5] )
    print ('    |   |   ')
    print('-----------------')
    print ('    |   |   ')
    print ('' +board[0]+ '    | ' +board[1]+ '  |' +board[2] )
    print ('    |   |   ')

drawboard(['', '', '', '', '', '', '', '', ''])



# Simple Game
import numpy
game = [[2,2,1], [1,1,2], [1,2,1]]
set_row = ()
set_col = ()

def line_match(game):
    for i in range(3):
        set_row = set(game[i])
        if len(set_row) == 1 and game[i][0] != 0:
            return game[i][0]
    return 0

def diagonal_match(game):
    if game[1][1] != 0:
        if game[1][1] == game[0][0] == game[2][2]:
            return game[1][1]
        elif game[1][1] == game[0][2] == game[2][0]:
            return game[1][1]
    return 0
if line_match(game) > 0:
    print(str(line_match(game)) + str(' row wins!'))
if line_match(numpy.transpose(game)) > 0:
    print(str(line_match(numpy.transpose(game))) + str(' column wins!'))
if diagonal_match(game) > 0:
    print(str(diagonal_match(game)) + str(' diagonal wins!'))

# More elaborate interactive game

def draw_line(width, edge, filling):
    print(filling.join([edge] * (width + 1)))

def draw_board(width, height):
    draw_line(width, " ", "__")
        for i in range(height):
            draw_line(width, "|", "__")
        print("\n")

def display_winner(player):
    if player == 0:
        print("Tie")
        else:
            print("Player " + str(player) + " wins!")

def check_row_winner(row):
    """
        Return the player number that wins for that row.
        If there is no winner, return 0.
        """
            if row[0] == row[1] and row[1] == row[2]:
                return row[0]
                    return 0

def get_col(game, col_number):
    return [game[x][col_number] for x in range(3)]

def get_row(game, row_number):
    return game[row_number]

def check_winner(game):
    game_slices = []
        for index in range(3):
            game_slices.append(get_row(game, index))
                game_slices.append(get_col(game, index))
    
        # check diagonals
        down_diagonal = [game[x][x] for x in range(3)]
        up_diagonal = [game[0][2], game[1][1], game[2][0]]
        game_slices.append(down_diagonal)
        game_slices.append(up_diagonal)
        
        for game_slice in game_slices:
            winner = check_row_winner(game_slice)
                if winner != 0:
                    display_winner(winner)
                        return winner
        
        display_winner(winner)
                            return winner

def start_game():
    return [[0, 0, 0] for x in range(3)]

def display_game(game):
    d = {2: "O", 1: "X", 0: " "}
        game_string = []
        for row_num in range(3):
            new_row = []
                for col_num in range(3):
                    new_row.append(d[game[row_num][col_num]])
                game_string.append(new_row)
        print(game_string)


def add_piece(game, player, row, column):
    """
        game: game state
        player: player number
        row: 0-index row
        column: 0-index column
        """
            game[row][column] = player
                return game

def convert_input_to_coordinate(user_input):
    return user_input - 1

def switch_player(player):
    if player == 1:
        return 2
        else:
            return 1


if __name__ == '__main__':
    game = start_game()
        display_game(game)
        player = 1
        
        # go on forever
        while True:
            print("Currently player: " + str(player))
                row = convert_input_to_coordinate(int(input("Which row? (start with 1) ")))
                column = convert_input_to_coordinate(int(input("Which column? (start with 1) ")))
                game = add_piece(game, player, row, column)
                display_game(game)
                player = switch_player(player)
