#!/usr/bin/env python
from __future__ import print_function
import argparse
import datetime as dt
import gc
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

import astar


def main():
    args = parseArgs(sys.argv)

    np.random.seed(args.seed)
    solLength = np.zeros(args.iter)
    expanded = np.zeros(args.iter)
    compTime = np.zeros(args.iter)
    for i in xrange(args.iter):
        visitedList = [0]
        G = pd.DataFrame(np.random.rand(args.cities, 2), columns=['x', 'y'])
        cost = 0
        t = TspNode(visitedList, G, cost, H[args.heuristic])
        start = dt.datetime.now()
        path, e = astar.search(t)
        compTime[i] = (dt.datetime.now()-start).total_seconds()
        solLength[i] = path.shape[0]
        expanded[i] = e
        G = None
        t = None
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


def primMstWeight(G):
    G['key'] = np.inf
    # set tree root arbitrarily
    G.loc[G.index[0], 'key'] = 0
    notTreeIdxs = G.index
    while len(notTreeIdxs) > 1:
        # select the min weighted vertex
        minIdx = G.loc[notTreeIdxs, 'key'].idxmin()
        # remove min weighted vertex from set of vertices not in tree
        notTreeIdxs = notTreeIdxs.drop(minIdx)
        # compute the distances to all remainig vertices
        dif = G.loc[notTreeIdxs, 'x':'y']-G.loc[minIdx, 'x':'y']
        dist = np.linalg.norm(dif, axis=1)
        # update distances with possible new minimums
        G.loc[notTreeIdxs, 'key'] = np.min(
            [G.loc[notTreeIdxs, 'key'].values, dist], axis=0)
    # print(G)
    return(G['key'].sum())


H = [primMstWeight]


class TspNode(astar.Node):

    def __init__(self, visitedList, graph, cost, heuristic):
        self.vl = visitedList
        self.G = graph
        self.cost = cost
        self.heuristic = heuristic
        # df[~df.index.isin([2,1])]

    def destroy(self):
        self.state = None
        self.cost = None
        # self.pth = None
        # self.sXy = None
        self.heuristic = None

    def successors(self):
        pass

    def isGoal(self):
        pass

    def g(self):
        return self.cost

    def h(self):
        return self.heuristic(self.state)

    def path(self):
        return self.pth

    @staticmethod
    def pruneSuccesors(sXy, pth):
        '''
        Removes any coordinates already visited in path.
        '''
        pass


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description=('Solve TSP using A* and plot results. '
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
        type=int,
        help='Seed for random number generator.',
    )
    parser.add_argument(
        '-c', '--cities',
        default=20,
        type=int,
        help='Number of cities on tour.',
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
