# Maze Runner

Maze Runner is a research output for some code for directing robots to solve
simple mazes.

Maze Runner is an example of how do design a planification (i.e., plan-building)
for fields such as x-ray physics and in areas such as experimental design
for x-ray beamlines.

This code uses generator coroutines, a very sophisticated approach in Python.
You will notice that maze solvers involve the use of [PEP-380 Delegation to a
Subgenerator](https://peps.python.org/pep380) (also known as `yield from`.)

This code simulates a robot that operates like an X-Ray beamline instrument:
- the robot receives messages
- the robot replies with information about it state
- you design a plan for the robot to solve the maze by passing messages back and forth

## Example for Use

```bash
PYTHONPATH=$PWD python examples/solver.py --maze mazes/linear0.mz
```
