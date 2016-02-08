#!/usr/bin/env python
from __future__ import print_function
import argparse
import datetime as dt
import gc
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import random
import sys

import astar


PDB = {
    (1, 1): 2,
    (1, 0): 3,
    (0, 1): 3,
    (2, 0): 2,
    (0, 2): 2,
    (3, 1): 2,
    (1, 3): 2,
    (2, 2): 4,
    (3, 0): 3,
    (0, 3): 3,
    (3, 2): 3,
    (2, 3): 3,
}


def main():
    args = parseArgs(sys.argv)
    # print(args)
    '''
    k = KnightNode(np.array([1, 1]), None, np.array([3, 2]))
    for s in k.successors():
        print('s.xy: {}'.format(s.xy))
        print('s.sXy: {}'.format(s.sXy))
        print('s.isGoal(): {}\n'.format(s.isGoal()))
    print('Solution: {}'.format(astar.search(k)))
    k = KnightNode(np.array([-3, 3]), None, np.array([-1, 1]))
    print('Solution: {}'.format(astar.search(k)))
    e = np.array([1, 1])
    k = KnightNode(np.array([0, 0]), None, e)
    print('{} solution: {}'.format(e, astar.search(k)))
    e = np.array([0, 1])
    k = KnightNode(np.array([0, 0]), None, e)
    print('{} solution: {}'.format(e, astar.search(k)))
    e = np.array([0, 2])
    k = KnightNode(np.array([0, 0]), None, e)
    print('{} solution: {}'.format(e, astar.search(k)))
    e = np.array([2, 2])
    k = KnightNode(np.array([0, 0]), None, e)
    print('{} solution: {}'.format(e, astar.search(k)))
    e = np.array([3, 1])
    k = KnightNode(np.array([0, 0]), None, e)
    print('{} solution: {}'.format(e, astar.search(k)))
    e = np.array([3, 3])
    k = KnightNode(np.array([0, 0]), None, e)
    print('{} solution: {}'.format(e, astar.search(k)))
    e = np.array([10, 9])
    k = KnightNode(np.array([0, 0]), None, e)
    print('{} solution: {}'.format(e, astar.search(k)))
    '''

    '''
    e = np.array([3, 0])
    k = KnightNode(np.array([0, 0]), None, e, H[0])
    print('{} solution: {}'.format(e, astar.search(k)))
    e = np.array([3, 2])
    k = KnightNode(np.array([0, 0]), None, e, H[0])
    print('{} solution: {}'.format(e, astar.search(k)))
    sys.exit(0)
    '''

    random.seed(args.seed)
    solLength = np.zeros(args.iter)
    expanded = np.zeros(args.iter)
    compTime = np.zeros(args.iter)
    for i in xrange(args.iter):
        goal = np.array(
            [
                random.randrange(args.max*-1, args.max+1, 1),
                random.randrange(args.max*-1, args.max+1, 1),
            ]
        )
        k = KnightNode(np.array([0, 0]), None, None, goal, H[args.heuristic])
        print(goal)
        start = dt.datetime.now()
        path, e = astar.search(k)
        compTime[i] = (dt.datetime.now()-start).total_seconds()
        solLength[i] = path.shape[0]
        expanded[i] = e
        k = None
        gc.collect()
    print('solLength: {}'.format(solLength))
    print('expanded: {}'.format(expanded))
    print('compTime: {}'.format(compTime))
    plot(args.heuristic, solLength, expanded, compTime, args.max, args.iter,
         args.seed, args.dir)


def plot(h, solLength, expanded, compTime, mx, it, seed, outDir):
    plt.close('all')
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(solLength, expanded)
    ax1.set_xlim(xmin=0)
    ax1.set_xlabel('Solution length')
    ax1.set_ylim(ymin=0)
    ax1.set_ylabel('Nodes expanded')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(solLength, compTime)
    ax2.set_xlim(xmin=0)
    ax2.set_xlabel('Solution length')
    ax2.set_ylim(ymin=0)
    ax2.set_ylabel('Compute time (s)')
    plt.grid(False)
    plt.tight_layout(True)
    plt.savefig(
        os.path.join(
            outDir,
            'p5_h_{}_max_{}_iter_{}_seed_{}.eps'.format(
                h,
                mx,
                it,
                seed,
            )
        ),
        dpi=800,
    )


def v0(xy, goal):
    return 0.


def v1(xy, goal):
    return np.abs(np.max(goal-xy)/2)


def v2(xy, goal):
    maxDif = np.abs(np.max(goal-xy))
    minDif = np.abs(np.min(goal-xy))
    return maxDif-minDif+2*int(math.floor((maxDif-2*(maxDif-minDif))/3.))


def v3(xy, goal):
    return np.sum(np.abs(goal-xy))/3


def v4(xy, goal):
    absDist = np.abs(goal-xy)
    h = PDB.get(tuple(absDist))
    if h is not None:
        return h
    return np.sum(absDist)/3


H = [v0, v1, v2, v3, v4]


class KnightNode(astar.Node):
    moves = np.array(
        [
            [2, 1],
            [2, -1],
            [-2, 1],
            [-2, -1],
            [1, 2],
            [1, -2],
            [-1, 2],
            [-1, -2],
        ]
    )

    def __init__(self, xy, ppth, pg, goal, heuristic):
        self.xy = xy
        self.goal = goal
        self.cost = 0 if pg is None else pg+1
        self.pth = (np.array([self.xy]) if ppth is None
                    else np.concatenate(
                        (ppth, np.array([self.xy])),
                        axis=0)
                    )
        successors = self.xy + KnightNode.moves
        self.sXy = KnightNode.pruneSuccesors(successors, self.pth)
        self.heuristic = heuristic

    def destroy(self):
        self.xy = None
        self.goal = None
        self.cost = None
        self.pth = None
        self.sXy = None
        self.heuristic = None

    def successors(self):
        return [KnightNode(xy, self.pth, self.cost, self.goal, self.heuristic)
                for xy in self.sXy]

    def isGoal(self):
        return np.array_equal(self.xy, self.goal)

    def g(self):
        return self.cost

    def h(self):
        return self.heuristic(self.xy, self.goal)

    def path(self):
        return self.pth

    @staticmethod
    def pruneSuccesors(sXy, pth):
        '''
        Removes any coordinates already visited in path.

        TODO: prune paths that do not go in the correct direction or
        redundantly go in the correct direction when far away from the goal.
        '''
        # Set differnce code found here
        # http://stackoverflow.com/questions/11903083/find-the-set-difference-between-two-large-arrays-matrices-in-python
        sXy_rows = sXy.view([('', sXy.dtype)]*sXy.shape[1])
        pth_rows = pth.view([('', pth.dtype)]*pth.shape[1])
        return np.setdiff1d(sXy_rows, pth_rows).view(sXy.dtype).reshape(
            -1, sXy.shape[1])


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description=('Solve knight movement using A* and plot results. '
                     'Written in Python 2.7.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d', '--dir',
        default='../doc/report/fig/',
        help='Directory to place results.',
    )
    parser.add_argument(
        '-s', '--seed',
        default=0,
        help='Seed for random number generator.',
    )
    parser.add_argument(
        '-m', '--max',
        default=20,
        type=int,
        help='Maximum solution coordinate distance from origin.',
    )
    parser.add_argument(
        '-i', '--iter',
        default=10,
        type=int,
        help='Number of iterations.',
    )
    parser.add_argument(
        '-H', '--heuristic',
        default=len(H)-1,
        type=int,
        help='Heuristic function in {}.'.format(range(len(H))),
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
