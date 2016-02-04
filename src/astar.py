from __future__ import print_function
import heapq


def search(start):
    frontier = PriorityQueue()
    frontier.push(start, start.f())
    expanded = 0
    while True:
        node = frontier.pop()
        if node is None:
            return None, expanded
        if node.isGoal():
            return node.path(), expanded
        expanded += 1
        for s in node.successors():
            frontier.push(s, s.f())


class PriorityQueue:
    def __init__(self):
        self.h = []

    def push(self, n, priority):
        heapq.heappush(self.h, (priority, n))

    def pop(self):
        return None if len(self.h) == 0 else heapq.heappop(self.h)[1]

    def empty(self):
        return len(self.h) == 0


class Node:
    def parent(self):
        pass

    def successors(self):
        pass

    def isGoal(self):
        pass

    def f(self):
        return self.g()+self.h()

    def g(self):
        pass

    def h(self):
        pass

    def path(self):
        pass
