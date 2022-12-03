# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
from copy import deepcopy
from random import randint, choice
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
        self.monte_carlo = None # initialize
        self.autoplay = True 

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
        if self.monte_carlo is None: 
            self.monte_carlo = MonteCarlo(chess_board, my_pos, adv_pos, max_step)

        move = self.monte_carlo.get_play(chess_board, my_pos, adv_pos, max_step)

        #my_pos = move[0]
        #dir = move[1]

        return move #my_pos, dir

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

        sec = kwargs.get('time', 1.95)
        self.calculation_time = timedelta(seconds=sec)

        self.preprocessing = True 

        self.max_moves = kwargs.get('max_moves', 50)

        self.C = kwargs.get("C", 1.4)

        self.wins = {}
        self.plays = {}
        self.movecount = 0 

    def get_play(self, board, my_pos, adv_pos, max_step):
        """
        Calculate best move for the current game state
        and returns move as ((x,y), dir)
        """
        self.chess_board = board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        player = True
        self.movecount += 1

        legal_moves = self.get_moves(self.chess_board, self.my_pos, self.adv_pos, self.max_step) # get list of legal random moves

        begin = datetime.utcnow()

        if self.preprocessing:
            time = timedelta(seconds = 29.95)
            self.preprocessing = False
        else:
            time = self.calculation_time

        while datetime.utcnow() - begin < time:
            self.run_sim()

        moves_states = [(play, (play, self.movecount)) for play in legal_moves] # moves and states in one list

        move, S = moves_states[0]
        percent = self.wins.get((player, S), 0)/self.plays.get((player, S), 1)

        for p, S in moves_states[1:]:
            if self.wins.get((player, S), 0)/self.plays.get((player, S), 1) > percent:
                move = p
                percent = self.wins.get((player, S), 0)/self.plays.get((player, S), 1)
        
        self.movecount += 1
        return move

    def run_sim(self):
        """
        Runs simulations
        """
        board = deepcopy(self.chess_board)
        pos = self.my_pos
        adv = self.adv_pos
        max_step = self.max_step
        visited_states = set() # to store already visited states in the simulation
        player = True
        plays, wins = self.plays, self.wins
        winner = None # initialize to None at first
        movecount = self.movecount  

        expand = True
        for t in range(self.max_moves): # run simulation until maximum amount of moves is reached
            legal_moves = self.get_moves(board, pos, adv, max_step)
            moves_states = [(play, (play, movecount)) for play in legal_moves] # moves and states in one list

            if all(plays.get((player, S)) for p, S in moves_states):
                log_total = log(sum(plays[(player, S)] for  p, S in moves_states))

                value = 0
                play, state = moves_states[0]

                for p, S in moves_states[1:]:
                    v = (wins[(player, S)]/plays[(player, S)])+self.C*sqrt(log_total/plays[(player, S)])
                    if v > value:
                        value = v
                        play = p
                        state = S
            
            else:
                play, state = choice(moves_states)

            move, dir = play
            r, c = move
            board = self.apply_move(board, r, c, dir)

            if expand and (player, state) not in self.plays:
                expand = False
                self.plays[(player, state)] = 0
                self.wins[(player, state)] = 0

            visited_states.add((player, state))

            win, score = self.check_endgame(board, pos, adv)
            if win:
                if score == 1:
                    winner = player
                else:
                    winner = not player
                break

            pos = adv
            adv = move
            player = not player
            movecount += 1
        
        for player, state in visited_states:
            if (player, state) not in self.plays:
                continue
            self.plays[(player, state)] += 1
            if winner is not None:
                if player == winner:
                    self.wins[(player, state)] += 1

    def check_endgame(self, chess_board, my_pos, adv_pos):
            """
            Check if the game ends and compute the current score of the agents.

            Returns
            -------
            is_endgame : bool
                Whether the game ends.
            player_1_score : int
                The score of player 1.
            player_2_score : int
                The score of player 2.
            """
            board_size = chess_board.shape[0]
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

            # Union-Find
            father = dict()
            for r in range(board_size):
                for c in range(board_size):
                    father[(r, c)] = (r, c)

            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]

            def union(pos1, pos2):
                father[pos1] = pos2

            for r in range(board_size):
                for c in range(board_size):
                    for dir, move in enumerate(
                        moves[1:3]
                    ):  # Only check down and right
                        if chess_board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)

            for r in range(board_size):
                for c in range(board_size):
                    find((r, c))
            p0_r = find(tuple(my_pos))
            p1_r = find(tuple(adv_pos))
            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)
            if p0_r == p1_r:
                return False, 0
            if p0_score == p1_score:
                return True, 0
            return True, max(0, (p0_score-p1_score)/abs(p0_score-p1_score))
    
    def apply_move(self, chess_board, r, c, dir):
        """
        returns position from a simulation move on the board
        """
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposite_dir = {0: 2, 1: 3, 2: 0, 3: 1}
        # 0 = up, 1 = right, 2 = down, 3 = left

        board = deepcopy(chess_board)

        board[r, c, dir] = True

        move = moves[dir]
        board[r + move[0], c + move[1], opposite_dir[dir]] = True
        return board

    def get_moves(self, chess_board, my_pos, adv_pos, max_step):
        """
        returns a list of random moves from sampling
        """
        moves = set()
        for i in range(max_step*4):
            moves.add(self.get_random_move(chess_board, my_pos, adv_pos, max_step))

        return list(moves)

    def get_random_move(self, chess_board, my_pos, adv_pos, max_step):
        """
        returns a random move
        """
        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = randint(0, max_step)

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = randint(0, 3)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = randint(0, 3)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = randint(0, 3)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = randint(0, 3)

        return my_pos, dir