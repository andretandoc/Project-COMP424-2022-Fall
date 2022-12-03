# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
from copy import deepcopy
from random import randint
from math import log, sqrt
from datetime import datetime, timedelta

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y) of where your player is at that moment
        - adv_pos: a tuple of (x, y) of where the other player is at that moment
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # Defining possible moves
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1)) # possible moves: up, right, down, left
        steps = randint(0, max_step + 1) # within max steps, samples with random for number of steps to take

        # Moving
        # General strategy:
        # Do something completely random, if you can reach the adversary, move there
        for _ in range(steps):

            r, c = my_pos

            # if adversary is in range, move there
            if adv_pos in range(my_pos):
                dir = randint(0,4)
                # m_r, m_c = 
                # my_pos =
            else:
                dir = randint(0,4) # chooses a random direction
                m_r, m_c = moves[dir] # 0 = up, 1 = right, 2 = down, 3 = left
                my_pos =  (r + m_r, c + m_c) # update our player position by 1, goes in one of the 4 directions only once
                # current x + move x(-1, 0, 1) and current y + move y(0, 1, -1). this is what you're returning.

            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos: # while a wall or adversary blocks this move
                k += 1
                if k > 300:
                    break
            
                r, c = my_pos
                dir = randint(0,4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c) # this is what you're updating and returning. (where you end up)

            if k > 300:
                my_pos = ori_pos
                break

        # Putting Barrier
        # once we are at our destination, choose a direction to place the barrier in
        # if the adversary is next to you, place it in that direction
        dir = randint(0,4) # put it anywhere
        r, c = my_pos # mapping my_pos(0) and my_pos(1) to r and c so that we can perform a check
        
        while chess_board[r, c, dir]: # if there's already a barrier/wall there,
            dir = randint(0,4) # keep trying to put it down anywhere randomly

        return my_pos, dir

        # dummy return
        #return my_pos, self.dir_map["u"]

class MonteCarlo:
    """
    MCTS algorithm
    """
    def __init__(self, chess_board, my_pos, adv_pos, max_step, **kwargs):
        """
        Initializing
        """
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step

        seconds = kwargs.get('time', 1.99)
        self.calculation_time = datetime.timedelta(seconds=seconds)

        # self.preprocessing = True 

        self.max_moves = kwargs.get('max_moves', 100)

    def get_play(self):
        """
        Calculate best move for the current game state
        and returns my_pos, dir as ((x,y), dir)
        """
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            self.run_sim()

    def run_sim(self):
        """
        Runs simulations
        """
        chess_board = deepcopy(self.chess_board)
        my_pos = self.my_pos
        adv_pos = self.adv_pos
        max_step = self.max_step

        expand = True
        for t in range(self.max_moves): # run simulation until maximum amount of moves is reached
            legal_moves = self.get_moves(chess_board, my_pos, adv_pos, max_step)
            # moves_states = [(play, (play, num)) for play in moves] # moves and states in one list

    def get_moves(self, chess_board, my_pos, adv_pos, max_step):
        """
        returns a list of random moves from sampling
        """
        moves = []
        for i in range(max_step):
            random_move = self.get_random_move(chess_board, my_pos, adv_pos, max_step)
            moves.add(random_move)

        return moves

    def get_random_move(chess_board, my_pos, adv_pos, max_step):
        """
        returns a random move
        """
        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = randint(0, max_step + 1)

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = randint(0, 4)

        return my_pos, dir