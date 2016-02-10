# Homework 1
## Building
### Dependencies
Programs are dependent upon Python 2.7.11, pandas 0.17.1, matplotlib 1.5.1, and numpy 1.10.4.
## Running Knight Search
```
python ./p5.py -h
usage: p5.py [-h] [-d DIR] [-s SEED] [-m MAX] [-i ITER] [-H HEURISTIC]

Solve knight movement using A* and plot results. Written in Python 2.7.

optional arguments:
	-h, --help            show this help message and exit
	-d DIR, --dir DIR     Directory to place results. (default:
							../doc/report/fig/)
	-s SEED, --seed SEED  Seed for random number generator. (default: 0)
	-m MAX, --max MAX     Maximum solution coordinate distance from origin.
							(default: 20)
	-i ITER, --iter ITER  Number of iterations. (default: 10)
	-H HEURISTIC, --heuristic HEURISTIC
							Heuristic function in [0, 1, 2, 3, 4]. (default: 4)
```
### Example
`python ./p5.py`
## Running TSP Search
```
python ./p6.py -h
usage: p6.py [-h] [-d DIR] [-s SEED] [-c CITIES] [-r] [-l] [-i ITER]
             [-H HEURISTIC]

Solve TSP using A* and plot results. Written in Python 2.7.

optional arguments:
	-h, --help            show this help message and exit
	-d DIR, --dir DIR     Directory to place results. (default:
							../doc/report/fig/)
	-s SEED, --seed SEED  Seed for random number generator. (default: 0)
	-c CITIES, --cities CITIES
							Number of cities on tour. (default: 20)
	-r, --random          Randomize number of cities. (default: False)
	-l, --local           Compare A* to local search. (default: False)
	-i ITER, --iter ITER  Number of iterations. (default: 10)
	-H HEURISTIC, --heuristic HEURISTIC
							Heuristic function in [0, 1]. (default: 0)
```
### Example
`python ./p6.py -r`
