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
        while self.monte_carlo is None: 
            self.monte_carlo = MonteCarlo(chess_board, my_pos, adv_pos, max_step)
        play = self.monte_carlo.get_play(chess_board, my_pos, adv_pos, max_step)

        my_pos = play[0]
        dir = play[1]

        return my_pos, dir

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

        self.compute_time = timedelta(seconds=(kwargs.get('time', 1.90))) # time allowed for each move
        self.preprocessing = True # set to False once setup is over
        self.max_moves = kwargs.get('max_moves', 100) # maximum moves allowed in simulations

        self.C = kwargs.get("C", 1.3) # set constant in UCT

        self.wins = {} # wins from simulations
        self.plays = {} # plays from simulations
        self.playcount = 0 

    def get_play(self, chessboard, my_pos, adv_pos, max_step):
        """
        Calculate best play for the current game state by running simulations during alloted time
        and returns play as a tuple ((x,y), dir)
        """
        self.chess_board = chessboard
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        player = True 
        self.playcount += 1
        self.max_depth = 0

        begin = datetime.utcnow()
        if self.preprocessing: # if first turn of the game, we have more time to setup
            time = timedelta(seconds = 29.90) 
            self.preprocessing = False
        else:
            time = self.compute_time

        while datetime.utcnow() - begin < time: # during the alloted time
            self.run_sim() # run simulation

        # sample a list of legal random moves from our current position on the board
        legal_moves = self.get_moves(self.chess_board, self.my_pos, self.adv_pos, self.max_step)

        play_states = [(move, (move, self.playcount)) for move in legal_moves]
        play, state = play_states[0]

        # calculate best play from simulation using wins/total plays
        win_percentage = self.wins.get((player, state), 0)/self.plays.get((player, state), 1)
        for p, state in play_states[1:]:
            if self.wins.get((player, state), 0)/self.plays.get((player, state), 1) > win_percentage:
                # update best play and wr%
                play = p
                win_percentage = self.wins.get((player, state), 0)/self.plays.get((player, state), 1)
        
        self.playcount += 1 # +1 turn

        # for debugging purposes    
        # print("Max depth reached:", self.max_depth)
        # print("Time taken:", datetime.utcnow() - begin)

        return play

    def run_sim(self):
        """
        Simulates moves for as long as time allows or until
        maximum moves parameter is reached
        """
        chessboard = deepcopy(self.chess_board) # to avoid modifying original board
        plyr_pos = self.my_pos
        enemy_pos = self.adv_pos
        max_step = self.max_step

        visited_states = set() # to store already visited states in the simulation
        player = True # our turn first
        winner = None # no winner at first
        wins = self.wins # total wins
        plays = self.plays # nb times visited
        playcount = self.playcount

        node_expansion = True
        
        # selection
        for t in range(self.max_moves): # run simulation until maximum amount of moves is reached
            legal_moves = self.get_moves(chessboard, plyr_pos, enemy_pos, max_step)
            play_states = [(move, (move, playcount)) for move in legal_moves]

            # Tree traversal
            if all(plays.get((player, S)) for m, S in play_states): # if current state is not a leaf node
                move, state = play_states[0]
                value = 0
                
                # use UCB1 formula to decide which node to visit
                log_calc = log(sum(plays[(player, S)] for  m, S in play_states))
                for m, S in play_states[1:]:
                    v = (wins[(player, S)]/plays[(player, S)])+self.C*sqrt(log_calc/plays[(player, S)])

                    if v > value:
                        value = v
                        move = m
                        state = S
            
            # leaf node
            else:
                move, state = choice(play_states) # rollout

            play, dir = move
            r, c = play
            chessboard = self.apply_move(chessboard, r, c, dir) # update state of chessboard from simulated move

            # expansion
            if node_expansion and (player, state) not in self.plays: 
                node_expansion = False # expand 1 state
                self.plays[(player, state)] = 0
                self.wins[(player, state)] = 0
                # if t > self.max_depth: 
                    # self.max_depth = t

            visited_states.add((player, state)) # mark as visited

            win, score = self.check_endgame(chessboard, plyr_pos, enemy_pos)
            if win:
                if score == 1:
                    winner = player
                else:
                    winner = not player
                break
            
            # adversary's turn 
            plyr_pos = enemy_pos
            player = not player # switch to adversary
            enemy_pos = play # adversary makes his play
            playcount += 1
        
        # update/back-propagation
        # after all simulations are finished, we update the total wins and plays of our simulation
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
        returns board state from a simulated move
        """
        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Opposite Directions
        opposites = {0: 2, 1: 3, 2: 0, 3: 1} 

        board = deepcopy(chess_board) # copy current state of the board
        # simulate barrier on one side
        board[r, c, dir] = True
        move = moves[dir]

        # simulate barrier on opposite side
        board[r + move[0], c + move[1], opposites[dir]] = True

        return board # return state of the board

    def get_moves(self, chess_board, my_pos, adv_pos, max_step):
        """
        returns a list of random moves from sampling
        """
        moves = set()
        for i in range(max_step*4): # times 4
            moves.add(self.get_random_move(chess_board, my_pos, adv_pos, max_step))

        return list(moves)

    def get_random_move(self, chess_board, my_pos, adv_pos, max_step):
        """
        returns a random move as a tuple ((x,y),dir)
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