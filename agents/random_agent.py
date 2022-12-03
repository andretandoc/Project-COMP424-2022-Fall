import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent

# Important: you should register your agent with a name
@register_agent("random_agent")
class RandomAgent(Agent):
    """
    Example of an agent which takes random decisions
    """

    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = "RandomAgent"
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        # Moves (Up, Right, Down, Left)
        # defining our possible moves
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1)) # possible moves: up down, left right, stay in place
        steps = np.random.randint(0, max_step + 1) # within max steps, samples with numpy random for steps to take

        # Random Walk
        # general strategy, in this case it just moves randomly
        # loop does 1 step at a time until steps taken is reached, and adjusts for when a barrier/adversary is in the way
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4) # chooses a random direction
            m_r, m_c = moves[dir] # 0 = up, 1 = right, 2 = down, 3 = left
            my_pos =  (r + m_r, c + m_c) # update our player position by 1, goes in one of the 4 directions only once
            # current x + move x(-1, 0, 1) and current y + move y(0, 1, -1)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos: # while a wall or adversary blocks this move
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4) # it just moves in a random direction again
                m_r, m_c = moves[dir] 
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        # once the loop is finished, choose a direction to place the barrier in
        dir = np.random.randint(0, 4) # put it down in a random direction
        r, c = my_pos
        while chess_board[r, c, dir]: # if there's already a barrier/wall in the way
            dir = np.random.randint(0, 4) # keep trying in random directions around you until you can put one down

        return my_pos, dir # return player position(x,y) and direction facing(up, right, down, left)
