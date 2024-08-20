from mazerunner import Request, Response, connection
from time import sleep
from random import Random

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
