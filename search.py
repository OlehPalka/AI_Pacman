# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from inspect import stack
from operator import rshift
from xml.etree.ElementTree import iselement
import util
check = True


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    global check
    res_stack = []
    result = []

    visited = []
    dfs_help(problem, problem.getStartState(), res_stack, visited)
    while res_stack != []:
        val = res_stack.pop()
        result.append(val)
    check = True
    return result[::-1]


def dfs_help(problem, cur_state, res_stack, visited, dir=False):
    global check
    visited.append(cur_state)
    if dir != False:
        res_stack.append(dir)

    if problem.isGoalState(cur_state):
        check = False
        return

    for i in problem.getSuccessors(cur_state):
        if i[0] not in visited:
            dfs_help(problem, i[0], res_stack, visited, i[1])
            if check != True:
                return
    if check:
        res_stack.pop()

    if problem.isGoalState(cur_state):
        check = False
        return

    return


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    parents = dict()
    visited = set()
    cur_state = problem.getStartState()

    queue.push(cur_state)

    while queue.isEmpty() is not True:
        cur_state = queue.pop()
        if problem.isGoalState(cur_state):
            break
        visited.add(cur_state)

        for i in problem.getSuccessors(cur_state):
            if i[0] not in visited and i[0] not in parents:
                queue.push(i[0])
                parents[i[0]] = (cur_state, i[1])

    result = []

    while cur_state != problem.getStartState():
        result.append(parents[cur_state][1])
        cur_state = parents[cur_state][0]
    return result[::-1]


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    prior_q = util.PriorityQueue()
    visited = set()

    cur_state = problem.getStartState()
    prior_q.push([cur_state, [], 0], 0)

    while prior_q.isEmpty() != True:
        cur_state = prior_q.pop()

        if cur_state[0] not in visited:
            visited.add(cur_state[0])
            if problem.isGoalState(cur_state[0]):
                return cur_state[1]

            for i in problem.getSuccessors(cur_state[0]):
                if i[0] not in visited:
                    prior_q.push([i[0], cur_state[1] + [i[1]], i[2] +
                                  cur_state[2]], i[2] + cur_state[2])


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    prior_q = util.PriorityQueue()
    visited = set()
    cur_state = problem.getStartState()
    h_dist = heuristic(cur_state, problem)
    prior_q.push([cur_state, [], 0], [h_dist, h_dist])

    while prior_q.isEmpty() != True:
        cur_state = prior_q.pop()

        if cur_state[0] not in visited:
            visited.add(cur_state[0])
            if problem.isGoalState(cur_state[0]):
                return cur_state[1]

            for i in problem.getSuccessors(cur_state[0]):
                if i[0] not in visited:
                    h_dist = heuristic(i[0], problem)
                    prior_q.push([i[0], cur_state[1] + [i[1]], i[2] +
                                  cur_state[2]], [i[2] + cur_state[2] + h_dist, h_dist])


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
