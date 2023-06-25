import abc
import datetime
import http
import os
import socket
import sys
import threading
from typing import Union, List

import fastapi
import open_spiel.python.algorithms.alpha_zero_mpg.services.actor as actor
import open_spiel.python.algorithms.alpha_zero_mpg.dto as mpg_dto
import open_spiel.python.algorithms.alpha_zero_mpg.utils as utils
import reverb
import tensorflow as tf
import requests
from open_spiel.python.distributed.http.common import format_date

from open_spiel.python.algorithms.alpha_zero_mpg.services import learner
import open_spiel.python.distributed.http.common as common
import pyspiel
import open_spiel.python.distributed.http.learner.service as service_node


class LearnerApp(common.AlphaZeroService):

    def __init__(self):
        self.evaluators_cluster = None
        self.actors_cluster = None
        self.cluster = None
        current_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(service_type="learner", path=os.path.join(current_dir, "config.yml"))
        if self.config.replay_buffer.implementation.type != "grpc":
            raise ValueError("Only reverb replay buffer is supported on server side")

        if self.config.replay_buffer.implementation.address == "auto":
            self.config.replay_buffer.implementation.address = socket.gethostname()

        table = reverb.Table(
            name=self.config.replay_buffer.implementation.table,
            sampler=utils.expand_arguments(
                utils.get_reverb_selector(self.config.replay_buffer.implementation.sampler.name)),
            remover=utils.expand_arguments(
                utils.get_reverb_selector(self.config.replay_buffer.implementation.remover.name)),
            max_size=self.config.replay_buffer.buffer_size,
            rate_limiter=reverb.rate_limiters.MinSize(self.config.replay_buffer.implementation.min_size),
            signature=utils.get_reverb_signature()
        )
        self.reverb_server = reverb.Server(tables=[table], port=self.config.replay_buffer.implementation.port)
        self.replay_buffer = mpg_dto.GrpcReplayBuffer(config=self.config.replay_buffer.implementation,
                                                      batch_size=self.config.training.batch_size,
                                                      padding=self.config.training.padding,
                                                      timeout=self.config.services.timeout)

        self.actors = []
        self.evaluators = []
        self._started = False
        if not self.config.services.wait_for_discovery:
            self.start()

    pass

    def start(self):
        self.discovery()
        response = self.cluster.start()
        self.learner = learner.Learner(config=self.config, replay_buffer=self.replay_buffer,
                                       model_broadcaster=self.broadcast_model,
                                       log_directory=self.working_directory)
        self.learner_thread = threading.Thread(target=self.learner.start, args=(self.game,), daemon=True)
        self.learner_thread.start()
        print(f"Started learner on Thread")
        self._started = True
        return response

    def discovery(self):
        self.actors = []
        self.evaluators = []
        for actor_hostname in os.listdir(os.path.join(app.services_path, "actor")):
            app.actors.append(service_node.ServiceNode(config=self.config, hostname=actor_hostname,
                                                       port=self.config.services.actors.port))
        self.actors_cluster = service_node.ServiceCluster(self.actors, name="actor")
        for evaluator_hostname in os.listdir(os.path.join(self.services_path, "evaluator")):
            self.evaluators.append(service_node.ServiceNode(config=self.config, hostname=evaluator_hostname,
                                                            port=self.config.services.evaluators.port))
        self.evaluators_cluster = service_node.ServiceCluster(self.evaluators, name="evaluator")
        self.cluster = service_node.ServiceCluster([self.actors_cluster, self.evaluators_cluster], name="cluster")
        return self.cluster.json()

    @property
    def started(self):
        return self._started

    def broadcast_model(self, model_path):
        for node in self.actors_cluster.children():
            requests.post(f"http://{node.identifier}:{self.config.services.actors.port}/model",
                          json={"path": model_path},
                          timeout=self.config.services.timeout)
        for node in self.evaluators_cluster.children():
            requests.post(f"http://{node.identifier}:{self.config.services.evaluators.port}/model",
                          json={"path": model_path},
                          timeout=self.config.services.timeout)


app = LearnerApp()


@app.get("/config")
def get_config():
    return app.config


@app.get("/start")
def start():
    return app.start()


@app.post("/start")
def start_with_config(config):
    app.config = config
    return start()


@app.get("/stop")
def stop():
    raise NotImplementedError()


@app.get("/health")
def health():
    return True


@app.get("/heartbeat")
def heartbeat():
    return health()


@app.get("/stats")
def stats():
    return NotImplementedError()


@app.get("/update")
def update():
    return NotImplementedError()


@app.get("/discovery")
def discovery():
    return app.discovery()
