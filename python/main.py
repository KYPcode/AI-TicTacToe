from HumanPlayer import HumanPlayer
from PlayerAgent import PlayerAgent
from Game import Game

if __name__ == "__main__":
    # Training phase
    player1 = PlayerAgent(0.3, 0.4, 0.9, "Red")
    player2 = PlayerAgent(0.1, 0.4, 0.9, "Blue")
    Game(player1, player2).launchGame(10000, False)
    # Testing phase
    player1 = PlayerAgent(0, 0, 0, "Red")
    player2 = HumanPlayer("Blue")
    player1.loadPolicy("policy_Red.json")
    Game(player1, player2).testAgainstHuman()