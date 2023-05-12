# AI-TicTacToe

## How reinforcement AIs work?

Reinforcement AIs are very often used in the context of games but not only! Indeed, their most topical use for me is in autonomous cars which would use, among other things, this kind of AI to follow a "controlled" driving.

For this project, the method used is classic Q-learning which, according to my research, seems to be one of the most accessible methods to use.

## Q-Learning

Q-learning allows the agent to determine the best action to take in a given environment using a Q-function that associates with each state and action pair a value that represents the expected future cumulative reward of taking that action in that state.

The agent then uses this function to choose the action that maximizes the value of Q-function in the current state. Over time, the agent updates the Q-function using real observations of the environment to refine its decisions and maximize the cumulative reward obtained.

## A few reminders about Tic Tac Toe

For those who have never played Tic Tac Toe in class or anywhere else, here are some simple explanations. 

It is a board game for two players played on a 3x3 grid. Players take turns to place their mark, either an "X" or an "O" on an empty square of the grid. The goal is to place three of their marks in a horizontal, vertical or diagonal line before the other player. If the grid is filled and no player has managed to place three marks, it is a draw.



______________________

## Defining the environment
______________________

One of the three keystones of reinforcement algorithms is the environment and its states in which our agent will evolve. To do that, I have decided to implement this in a python class that I will call ```Game``` with the ```__init__``` method preparing the ground for the different variables we will need.

```python
def __init__(self, player1, player2):
   self.player1 = player1
   self.player2 = player2
   self.initializeGame()
   self.current_player = "Red"
```


The first thing to do when the Tic Tac Toe game starts is to initialize a board. The ```InitializeGame``` function will generate a special 3x3 matrix, which is a **magic square** here, which will be used to check the victory condition. 

``` python
def initializeGame(self):
   self.gameboard = [
       2, 7, 6,
       9, 5, 1,
       4, 3, 8
   ]
```

**Note:** Magic square notion will be introduced in a next part.

Once the board is ready, each player must choose a position for which the position will be replaced by their player name i.e. Red or Blue. To make sure that the choice of position to place a piece is correct, it is necessary to first check which positions are available.

``` python
def findEmptyPosition(self):
   indices = []
   for idx, value in enumerate(self.gameboard):
       if type(value) == int:
           indices.append(idx)
   return indices
```

Knowing that each position where a player will have placed a piece will be marked by a character string linked to its name, it is enough simply to recover the indexes of the list where the value is an integer.

As soon as we know which positions are available, we will have to send this list of indices back to the player's class so that he can choose the position he wants from this list.

```python
def addPosition(self,position):
   self.gameboard[position] = self.current_player
   if self.current_player == "Red":
       self.current_player = "Blue"
   else:
       self.current_player = "Red"
```

Then the addPosition function will get the position chosen by the current player (```self.current_player```) and change the ```self.gameboard``` variable to add the player's piece.

To finish this ```Game``` class, we need to know when the game stops. To perform this, two functions are implemented: 

- ```checkWin``` : This function will play on the fact that the gameboard is a magic square and therefore all its diagonals, rows and columns have the same sum (in this case 15). The function will therefore simply have to make sure that none of its rows/diagonals/columns are filled by one player by checking the sums.
- ```checkDraw``` : TicTacToe will often end in a draw and so a dedicated function will be essential.

So let's start with the simplest one: 
``` python
def checkDraw(self):
   draw_token = True
   for value in self.gameboard:
       if type(value) != str:
           draw_token = False
           break
   if draw_token == True:
       print("Draw!")
   return draw_token
```

This function will scan the entire board, checking that there are only strings. If this is the case, then all the boxes are filled by the player and therefore there is a draw. Otherwise, if only one of the boxes is an integer (since player pieces are represented by their name in string), then the loop is stopped and the function returns False.

```python
def checkWin(self):
   win_token = False
   clean_gameboard = [
       2, 7, 6,
       9, 5, 1,
       4, 3, 8
   ]
   index_of_pieces = self.findEmptyPosition(self.gameboard, self.player_name)
   for index, position in enumerate(self.gameboard):
       if index in index_of_pieces:
           pass
       else:
           clean_gameboard[index] = 0
   # Transform the gameboard by replacing the not string case by 0 for red gameboard and blue gameboard
   dimension = len(np.reshape(clean_gameboard, (3, 3))[0])
   sum_list = []
   # Sum lines
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
   if 15 in sum_list:
       print("Winner is %s" % self.player_name)
       win_token = True
   return win_token
```

This function is based on the mathematical concept of magic squares. The idea is to replace all the values on the board where the player has not placed any piece by 0 and then to replace all the other values where he has played a piece by the initial values of the magic square. Thus, if the player has filled a row/diagonal/column, then in the sum_list, there will be at least one 15 and so the game must stop because he has won.


Operation of the ```checkWin()``` function applied to the player "Red" on a winning case.

**Note**: We need a small external function to perform the step of transforming the player positions into a list of magic square numbers to perform the calculation of sums. 

```python
def find_indices_item_to_find(list_to_check, item_to_find):
   indices = []
   for idx, value in enumerate(list_to_check):
       if value == item_to_find:
           indices.append(idx)
   return indices
```

From here, we have a working game environment where all we need to do is make a few changes and add the ```LaunchGame``` function later once the ```PlayerAgent``` class is ready.
_______________

## Definition of the agent
_______________

Now that the environment is ready, we need an agent that will choose the positions and learn from its mistakes thanks to the Q-learning algorithm. For this, I used a class called ```PlayerAgent``` which takes as input all the parameters necessary to calculate the different values of the Q-value function.

```python
def __init__(self, learning_rate: 0.1, eps: 0.4, gamma: 0.9):
   self.qvalue_dic = {}
   self.states_list = []
   self.learning_rate = learning_rate
   self.exploratory_rate = eps
   self.gamma = gamma
```

The first thing this player will have to do is one of the most important things in our game: choosing his position with two ways: 

- **The exploratory response**: The initial definition of an epsilon value tells the algorithm that eps% of the choices will be made randomly. These responses are essential for the training phase as they simply allow new combinations to be experimented with and therefore the rewards for each decision to evolve. Without it, there is simply no improvement in the policy.

- **The response with maximum Q-value**: This response could be defined simply as a "considered" response based on Q-value values already defined by past experience. The idea here is to choose the decision that offers the maximum reward.

These two responses are included in the ```nextPosition()``` function: 

```python
def qValueDictChecking(self, dictionnary, futur_state):
   if dictionnary.get(futur_state) is None:
       q_value = 0
   else:
       q_value = dictionnary.get(futur_state)
   return q_value

def nextPosition(self, gameboard, empty_cells_index):
   # Exploratory
   if random.uniform(0, 1) < self.exploratory_eps:
       next_position = np.random.choice(empty_cells_index)
   # Max Q-value
   else:
       qmax = - 100000000000000
       for position in empty_cells_index:
           futur_state = gameboard.copy()
           futur_state[position] = self.player_name
           futur_state_str = str(futur_state)
           # Verify the value of this state and if None create it
           qvalue = self.qValueDictChecking(self.qvalues_dic,futur_state_str)
           if qvalue > qmax:
               qmax = qvalue
               next_position = position
   return next_position
```

The call of this function in the ```LaunchGame()``` function of the Game class will return the position chosen by the agent.

Now, the agent can choose a position but he still has to be able to take into account the consequences to improve himself. For that, he receives a reward according to the principle of the Q-value according to the outcome of the game which will be null in case of a draw or a defeat and which will be 1 in case of a victory.

For this, at the end of each game, the ```backpropagationReward()``` function will apply the theoretical Q-value improvement formulas in the dictionary ```self.qvalues_dic```.

```python
def backpropagationReward(self, reward):
   for state in reversed(self.states_list):
       if self.qvalues_dic.get(state) is None:
           self.qvalues_dic[state] = 0
       self.qvalues_dic[state] = self.qvalues_dic[state] + self.learning_rate * (self.gamma * reward - self.qvalues_dic[state])
       reward = self.qvalues_dic[state]
```

So we have everything! Now all we need is a function in the ```Game``` class that runs x number of training games to improve the Q-value dictionary.

```python
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
       if verbose == True:
           if win_token == True:
               print("Winner is %s" % self.winner)
           if win_token == False and tie_token == True:
               print("Tie")
           self.displayVisibleGameboard()
       # Reset winner
       self.winner = ""
   self.player1.savePolicy()
   self.player2.savePolicy()
```

We are therefore able to train two agents against each other and recover the generated policy with the following function: 
```python
def savePolicy(self):
   with open('policy_' + str(self.player_name) + ".json", "w") as outfile:
       json.dump(self.qvalues_dic,outfile, indent = 4)
```

