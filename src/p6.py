#!/usr/bin/env python
from __future__ import print_function
import argparse
import datetime as dt
from itertools import product
import gc
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.spatial.distance as spd
import sys

import astar


def main():
    args = parseArgs(sys.argv)

    np.random.seed(args.seed)
    # solLength = np.zeros(args.iter)
    cityCnts = np.zeros(args.iter)
    expanded = np.zeros(args.iter)
    compTime = np.zeros(args.iter)
    cityCnt = args.cities
    for i in xrange(args.iter):
        if args.random:
            cityCnt = np.random.randint(1, high=args.cities+1)
        print('i: {}'.format(i))
        print('cityCnt: {}'.format(cityCnt))
        visitedList = [0]
        # G = pd.DataFrame(np.random.rand(cityCnt, 2), columns=['x', 'y'])
        G = generateProblem(cityCnt)
        # print(G)
        cost = 0
        t = TspNode(visitedList, G, cost, H[args.heuristic])
        start = dt.datetime.now()
        path, e = astar.search(t)
        # print('solution path: {}'.format(path))
        compTime[i] = (dt.datetime.now()-start).total_seconds()
        print('compTime: {}'.format(compTime[i]))
        # solLength[i] = path.shape[0]
        expanded[i] = e
        cityCnts[i] = cityCnt
        G = None
        t = None
        gc.collect()
    # print('solLength: {}'.format(solLength))
    print('cityCnts: {}'.format(cityCnts))
    print('expanded: {}'.format(expanded))
    print('compTime: {}'.format(compTime))
    plot(args.heuristic, cityCnts, args.random, expanded, compTime, args.cities,
         args.iter, args.seed, args.dir)


def generateProblem(cityCnt):
    '''
    Generate a cityCnt by 2 matrix where each entry is in [0,1).
    '''
    return pd.DataFrame(np.random.rand(cityCnt, 2), columns=['x', 'y'])


def plot(h, cityCnts, rand, expanded, compTime, cities, it, seed, outDir):
    plt.close('all')
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(cityCnts, expanded)
    ax1.set_xlim(xmin=0)
    ax1.set_xlabel('Number of cities')
    ax1.set_ylim(ymin=0)
    ax1.set_ylabel('Nodes expanded')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(cityCnts, compTime)
    ax2.set_xlim(xmin=0)
    ax2.set_xlabel('Number of cities')
    ax2.set_ylim(ymin=0)
    ax2.set_ylabel('Compute time (s)')
    plt.grid(False)
    plt.tight_layout(True)
    plt.savefig(
        os.path.join(
            outDir,
            'p6_h_{}_cities_{}_rand_{}_iter_{}_seed_{}.eps'.format(
                h,
                cities,
                rand,
                it,
                seed,
            )
        ),
        dpi=800,
    )


def localSearch(G):
    path = range(G.shape[0])+[0]
    swaps = range(1, G.shape[0])
    swaps = [pair for pair in product(swaps, repeat=2) if pair[0] != pair[1]]
    d = spd.squareform(spd.pdist(G.values))
    print(swaps)
    tourLength = sum([d[path[i], path[i+1]]
                      for i in xrange(len(path)-1)])
    while True:
        change = False
        print('while iter')
        for swap in swaps:
            print('for iter')
            path[swap[0]], path[swap[1]] = path[swap[1]], path[swap[0]]
            newTourLength = sum([d[path[i], path[i+1]]
                                 for i in xrange(len(path)-1)])
            if newTourLength < tourLength:
                tourLength = newTourLength
                change = True
                break
            else:
                path[swap[0]], path[swap[1]] = path[swap[1]], path[swap[0]]
        if not change:
            return path


def primMstWeight(G, visitedList):
    origNotTreeIdxs = G[~G.index.isin(visitedList[1:])].index
    if len(origNotTreeIdxs) == 0:
        return 0.
    notTreeIdxs = origNotTreeIdxs.copy()
    # G['key'] = np.inf
    G.loc[notTreeIdxs, 'key'] = np.inf
    # set tree root arbitrarily
    # G.loc[G.index[0], 'key'] = 0
    G.loc[notTreeIdxs[0], 'key'] = 0
    # notTreeIdxs = G.index
    # print('notTreeIdxs: {}'.format(notTreeIdxs))
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
    # weight = G['key'].sum()
    weight = G.loc[origNotTreeIdxs, 'key'].sum()
    # print('G in prim: {}'.format(G.loc[origNotTreeIdxs]))
    # print('MST weight: {}'.format(weight))
    return weight


def zero(G, visitedList):
    return 0.


H = [primMstWeight, zero]


class TspNode(astar.Node):

    def __init__(self, visitedList, graph, cost, heuristic):
        self.vl = visitedList
        self.G = graph
        self.cost = cost
        self.heuristic = heuristic
        # df[~df.index.isin([2,1])]

    def destroy(self):
        self.vl = None
        self.G = None
        self.cost = None
        self.heuristic = None

    def successors(self):
        notVisitedIdxs = self.G[~self.G.index.isin(self.vl)].index
        # compute the distances from current city to all remaining cities
        dif = (self.G.loc[notVisitedIdxs, 'x':'y'] -
               self.G.loc[self.vl[-1], 'x':'y'])
        dist = np.linalg.norm(dif, axis=1)
        if len(notVisitedIdxs) > 1:
            ss = [
                TspNode(
                    self.vl+[notVisitedIdxs[i]],
                    self.G,
                    self.cost+dist[i],
                    self.heuristic,
                ) for i in range(len(notVisitedIdxs))
            ]
        else:
            # compute the tour completion node
            ss = [
                TspNode(
                    self.vl+[notVisitedIdxs[i]]+self.vl[:1],
                    self.G,
                    # add the distance to the first city
                    self.cost+dist[i]+np.linalg.norm(
                        (self.G.loc[self.vl[0], 'x':'y'] -
                         self.G.loc[notVisitedIdxs[i], 'x':'y'])
                    ),
                    self.heuristic,
                ) for i in range(len(notVisitedIdxs))
            ]
        # print('Successors of {}:'.format(self.path()))
        '''
        for s in ss:
            print('\t{}'.format(s.path()))
        '''
        return ss

    def isGoal(self):
        '''
        Return true if all cities have been visited and the first visited is
        the last visited.
        '''
        return (np.alltrue(self.G.index.isin(self.vl)) and
                self.vl[0] == self.vl[-1])

    def g(self):
        return self.cost

    def h(self):
        '''
        Return the MST weight for cities not yet visited plus the first.
        '''
        # return self.heuristic(self.G[~self.G.index.isin(self.vl[1:])])
        return self.heuristic(self.G, self.vl)

    def path(self):
        return self.G.loc[self.vl, 'x':'y']


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
        '-r', '--random',
        action='store_true',
        help='Randomize number of cities.',
    )
    parser.add_argument(
        '-i', '--iter',
        default=10,
        type=int,
        help='Number of iterations.',
    )
    parser.add_argument(
        '-H', '--heuristic',
        default=0,
        type=int,
        help='Heuristic function in {}.'.format(range(len(H))),
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
