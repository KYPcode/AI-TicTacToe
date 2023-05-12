from simpleFunctions import *
import numpy as np

class Game:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.gameboard = []
        self.winner = ""

    def initializeGame(self):
        self.gameboard = [
            2, 7, 6,
            9, 5, 1,
            4, 3, 8
        ]

    def findEmptyPosition(self):
        emptyListIndices = []
        for idx, value in enumerate(self.gameboard):
            if type(value) == int:
                emptyListIndices.append(idx)
        return emptyListIndices

    def addPosition(self, position):
        self.gameboard[position] = self.current_player
        if self.current_player == "Red":
            self.player1.stateAppend(str(self.gameboard))
        else:
            self.player2.stateAppend(str(self.gameboard))
        win_token = self.checkWin()
        tie_token = self.checkDraw()
        if self.current_player == "Red":
            self.current_player = "Blue"
        else:
            self.current_player = "Red"
        return win_token, tie_token

    def checkWin(self):
        win_token = False
        clean_gameboard = [
            2, 7, 6,
            9, 5, 1,
            4, 3, 8
        ]
        index_of_pieces = find_indices_item_to_find(self.gameboard, self.current_player)
        for index, position in enumerate(self.gameboard):
            if index in index_of_pieces:
                pass
            else:
                clean_gameboard[index] = 0
        dimension = len(np.reshape(clean_gameboard, (3, 3))[0])
        sum_list = []
        sum_list.extend([sum(lines) for lines in np.reshape(clean_gameboard, (3, 3))])
        for col in range(dimension):
            sum_list.append(sum(row[col] for row in np.reshape(clean_gameboard, (3, 3))))
        result1 = 0
        for i in range(0, dimension):
            result1 += np.reshape(clean_gameboard, (3, 3))[i][i]
        sum_list.append(result1)
        result2 = 0
        for i in range(dimension - 1, -1, -1):
            result2 += np.reshape(clean_gameboard, (3, 3))[i][i]
        sum_list.append(result2)
        for sum_elem in sum_list:
            if sum_elem == 15:
                win_token = True
                self.winner = self.current_player
                break
        return win_token

    def checkDraw(self):
        draw_token = True
        for value in self.gameboard:
            if type(value) != str:
                draw_token = False
                break
        if draw_token == True and self.checkWin() == True:
            draw_token = False
        return draw_token

    def launchGame(self, iterations, verbose):
        for iter in range(iterations):
            print("Iteration", iter, "/", iterations)
            self.initializeGame()
            self.current_player = "Red"
            win_token, tie_token = False, False
            while win_token == False and tie_token == False:
                empty_pos = self.findEmptyPosition()
                if self.current_player == "Red":
                    position = self.player1.nextPosition(self.gameboard, empty_pos)
                else:
                    position = self.player2.nextPosition(self.gameboard, empty_pos)
                win_token, tie_token = self.addPosition(position)
            if self.winner == "Red":
                self.player1.backpropagationReward(1)
                self.player2.backpropagationReward(0)
            elif self.winner == "Blue":
                self.player1.backpropagationReward(0)
                self.player2.backpropagationReward(1)
            self.player1.resetStatesList()
            self.player2.resetStatesList()
            if win_token == True:
                if verbose == True:
                    print("Winner is %s" % self.winner)
                    self.displayVisibleGameboard()
            if win_token == False and tie_token == True:
                if verbose == True:
                    print("Tie")
                    self.displayVisibleGameboard()
            # Reset winner
            self.winner = ""
        self.player1.savePolicy()
        self.player2.savePolicy()

        return

    def displayVisibleGameboard(self):
        visible_gameboard = []
        for index, value in enumerate(self.gameboard):
            if type(value) == int:
                visible_gameboard.append(index)
            else:
                visible_gameboard.append(value)
        print(np.reshape(visible_gameboard, (3, 3)))

    def testAgainstHuman(self):
        self.initializeGame()
        self.current_player = "Red"
        win_token, tie_token = False, False
        while win_token == False and tie_token == False:
            empty_pos = self.findEmptyPosition()
            self.displayVisibleGameboard()
            if self.current_player == "Red":
                position = self.player1.nextPosition(self.gameboard, empty_pos)
            # Human must be player 2
            else:
                position = self.player2.nextPosition()
            win_token, tie_token = self.addPosition(position)
        self.displayVisibleGameboard()
        if win_token == True:
            print("Winner is %s" % self.winner)
        if win_token == False and tie_token == True:
            print("Draw")
