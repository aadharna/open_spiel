import datetime
import socket
import time

import fastapi
import open_spiel.python.algorithms.alpha_zero_mpg.services.actor as actor
import open_spiel.python.algorithms.alpha_zero_mpg.multiprocess.queue as queue_lib

import os

from open_spiel.python.algorithms.alpha_zero_mpg.multiprocess import orchestrator
from open_spiel.python.utils import spawn
import open_spiel.python.distributed.http.common as common
import open_spiel.python.distributed.http.dto.health as health_dto
import open_spiel.python.distributed.http.dto.model as model_dto


class HttpActorFactory:
    def __init__(self, working_directory):
        self.working_directory = working_directory
        pass

    def __call__(self, config, game, queue, hostname, num):
        act = actor.MultiProcActor(config, num, name=f"actor",log_directory=self.working_directory)
        return act.start(queue=queue, game=game)


class ActorApp(common.AlphaZeroService):

    def start(self):
        config = self.config
        game = self.game
        if isinstance(config.services.actors.instances, int):
            self.actors = [
                spawn.Process(self.actor_factory, kwargs=
                    {"game": game,
                     "config": config,
                     "num": i,
                     "hostname": socket.gethostname()
                     })
                for i in range(config.services.actors.instances)
            ]
            pass

        else:
            raise ValueError("As of now, only int is supported for actors")

        self.orchestrator = orchestrator.ActorsGrpcOrchestrator(config=config, actors=self.actors,
                                                                max_game_length=game.max_game_length(),
                                                                working_directory=self.working_directory)
        self.last_heartbeats = health_dto.NodeHeartbeatResponse(threads=[], timestamp=0, hostname=socket.gethostname())
        for thread_id in range(len(self.actors)):
            self.last_heartbeats.threads.append(health_dto.ThreadHeartbeatResponse(timestamp=0, thread_id=thread_id))

        self.orchestrator.add_on_heartbeat(self.update_heartbeat)
        self.model_path = None
        self._started = True

    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(service_type="actor", path=os.path.join(current_dir, "config.yml"))
        self.actor_factory = HttpActorFactory(self.working_directory)
        self.actors = []
        config = self.config
        self._started = False
        if not config.services.wait_for_discovery:
            self.start()

    pass

    @property
    def started(self):
        return self._started

    def update_heartbeat(self, thread_id, msg):
        self.last_heartbeats.threads[thread_id].timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def request_heartbeat(self):
        self.orchestrator.heartbeat()


app = ActorApp()


@app.get("/config")
def get_config():
    return app.config


@app.post("/start")
def start(request: fastapi.Request):
    return discovery(request)


@app.get("/stop")
def stop():
    app.orchestrator.stop()


@app.get("/health")
def health() -> health_dto.NodeHeartbeatResponse:
    if not app.started:
        return health_dto.NodeHeartbeatResponse(threads=[], timestamp=0, hostname=socket.gethostname())
    app.request_heartbeat()
    time.sleep(app.config.services.heartbeat_response)
    return app.last_heartbeats


@app.get("/heartbeat")
def heartbeat():
    return health()


@app.get("/stats")
def stats():
    return NotImplementedError()


@app.post("/model")
def update_model(model: model_dto.ModelPath):
    app.orchestrator.broadcast(queue_lib.QueueMessage(queue_lib.MessageTypes.QUEUE_MESSAGE, model.path))


@app.get("/discovery")
def discovery(request: fastapi.Request):
    host = request.client.host
    if not app.started:
        if app.config.replay_buffer.implementation.address == "auto":
            app.config.replay_buffer.implementation.address = host
        app.start()
        return {"status": "started"}
    else:
        return {"status": "already started"}
