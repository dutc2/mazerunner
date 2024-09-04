FROM python:3.12

COPY requirements.txt /tmp
RUN  env -C /tmp python -m pip install -r requirements.txt

COPY . /mazerunner

CMD env -C /mazerunner PYTHONPATH=/mazerunner python3 examples/solver.py -v --maze mazes/linear1.mz
