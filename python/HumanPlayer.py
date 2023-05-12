class HumanPlayer():
    def __init__(self, player_name):
        self.player_name = player_name

    def nextPosition(self):
        try:
            position = int(input("Human choose a position (position number):"))
            if position > 8:
                print("Out of the board.")
                position = int(input("Human choose another position (position number):"))
        except:
            print("Bug, choose again.")
            position = int(input("Human choose another position (position number):"))
            if position > 8:
                print("Out of the board.")
                position = int(input("Human choose another position (position number):"))
        return position

    def backpropagationReward(self, reward):
        pass

    def resetStatesList(self):
        pass

    def stateAppend(self, gameboard_new):
        pass

    def savePolicy(self):
        pass
