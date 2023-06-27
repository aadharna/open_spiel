import datetime
import socket
import time
import fastapi
import open_spiel.python.algorithms.alpha_zero_mpg.services.evaluator as evaluator
import open_spiel.python.algorithms.alpha_zero_mpg.multiprocess.queue as queue_lib

import os

from open_spiel.python.algorithms.alpha_zero_mpg.multiprocess import orchestrator
from open_spiel.python.utils import spawn

import open_spiel.python.distributed.http.common as common
import open_spiel.python.distributed.http.dto.health as health_dto
import open_spiel.python.distributed.http.dto.model as model_dto


class HttpEvaluatorFactory:
    def __init__(self, working_directory):
        self.working_directory = working_directory
        pass

    def __call__(self, config, game, queue, hostname, num):
        evaluate = evaluator.MultiProcEvaluator(config, num, name=f"evaluator",
                                                opponent=config.services.evaluators.opponent,
                                                log_directory=self.working_directory)
        return evaluate.start(queue=queue, game=game)


class EvaluatorApp(common.AlphaZeroService):

    def start(self):
        config = self.config
        game = self.game
        if isinstance(config.services.evaluators.instances, int):
            self.evaluators = [
                spawn.Process(self.evaluator_factory,
                              kwargs=
                              {"game": game,
                               "config": config,
                               "num": i,
                               "hostname": socket.gethostname()
                               })
                for i in range(config.services.evaluators.instances)
            ]
            pass

        else:
            raise ValueError("As of now, only int is supported for evaluators")

        self.orchestrator = orchestrator.EvaluatorOrchestrator(config=config, evaluators=self.evaluators,
                                                               max_game_length=game.max_game_length(),
                                                               working_directory=self.working_directory)
        self.last_heartbeats = health_dto.NodeHeartbeatResponse(threads=[], timestamp=0, hostname=socket.gethostname())
        for thread_id in range(len(self.evaluators)):
            self.last_heartbeats.threads.append(health_dto.ThreadHeartbeatResponse(timestamp=0, thread_id=thread_id))

        self.orchestrator.add_on_heartbeat(self.update_heartbeat)
        self.model_path = None
        self._started = True

    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(service_type="evaluator", path=os.path.join(current_dir, "config.yml"))
        self.evaluator_factory = HttpEvaluatorFactory(self.working_directory)
        self.evaluators = []
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
        return self.orchestrator.heartbeat()


app = EvaluatorApp()


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
    port = request.client.port
    if not app.started:
        if app.config.replay_buffer.implementation.address == "auto":
            app.config.replay_buffer.implementation.address = host
        print(f"Starting actor on {host}:{port}")
        app.start()
        return {"status": "started"}
    else:
        return {"status": "already started"}
