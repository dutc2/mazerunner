from time import sleep
from logging import getLogger, basicConfig, INFO, DEBUG
from random import Random

from mazerunner import Request, Response, connection, run_agent, parser

def turn_until_free(tick, *, random_state=None):
    rnd = random_state if random_state is not None else Random()
    yield rnd.choice([Request.TurnLeft, Request.TurnRight])()
    while not isinstance(resp := (yield Request.FrontSensor()), Response.NoWall):
        sleep(tick)
    yield Request.StopTurn()

def move_until_stopped(tick, *, random_state=None):
    rnd = random_state if random_state is not None else Random()
    yield Request.Move()
    while True:
        if isinstance(resp := (yield Request.FrontSensor()), Response.Wall):
            break
        if isinstance(resp := (yield Request.ExitSensor()), Response.Exit):
            break
        sleep(tick)
    yield Request.StopMove()

def simplesolver(tick, *, random_state=None):
    rnd = random_state if random_state is not None else Random()
    while not isinstance(resp := (yield Request.ExitSensor()), Response.Exit):
        yield from turn_until_free(tick=tick)
        yield from move_until_stopped(tick=tick)
        sleep(tick)

if __name__ == '__main__':
    args = parser.parse_args()
    logger = getLogger(__name__)
    basicConfig(level={0: INFO, 1: DEBUG}.get(args.verbose, INFO))

    agent_kwargs= {'host': args.host, 'port': args.port, 'maze': args.maze, 'seed': args.seed, 'errors': args.errors, 'tick': args.tick}
    run_agent(**agent_kwargs)

    with connection(host=args.host, port=args.port) as send:
        resp = send(req := Request.Test())
        logger.info('Request → Response: %16r → %r', req, resp)

        ci = simplesolver(tick=1, random_state=Random(0))

        resp = None
        while True:
            try:
                req = ci.send(resp)
            except StopIteration:
                break
            resp = send(req)
            logger.info('Request → Response: %16r → %r', req, resp)

        resp = send(req := Request.ExitSensor())
        logger.info('Request → Response: %16r → %r', req, resp)
