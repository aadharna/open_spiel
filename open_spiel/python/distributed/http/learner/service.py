import abc
import datetime
import http
from typing import List, Union

from open_spiel.python.distributed.http.common import format_date
from open_spiel.python.distributed.http.dto import health as health_dto
import requests


class Service(abc.ABC):
    @abc.abstractmethod
    def heartbeat(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def update_model(self,path):
        pass

    @abc.abstractmethod
    def get_model(self):
        pass

    @abc.abstractmethod
    def start(self):
        pass

    @property
    @abc.abstractmethod
    def identifier(self):
        pass

    @abc.abstractmethod
    def json(self):
        pass

    @abc.abstractmethod
    def health(self):
        pass


class ServiceNode(Service):
    def __init__(self,config,hostname:str,port:int):
        self.hostname = hostname
        self.port = port
        self.config=config
        self.timeout=config.services.timeout

    def heartbeat(self):
        response=requests.get(f"http://{self.hostname}:{self.port}/heartbeat",timeout=self.timeout)
        if response.status_code != http.HTTPStatus.OK:
            return health_dto.NodeHeartbeatResponse(timestamp=format_date(datetime.datetime.now()),hostname=self.hostname,threads=None,
                                                    status=response.status_code)
        health_dto.NodeHeartbeatResponse(**response.json(),status=response.status_code)

    def update_model(self,path):
        response=requests.post(f"http://{self.hostname}:{self.port}/model",json={"path":path},timeout=self.timeout)
        if response.status_code != http.HTTPStatus.OK:
            raise ValueError(f"Could not update model on {self.hostname}:{self.port}")

    def get_model(self):
        response=requests.get(f"http://{self.hostname}:{self.port}/model",timeout=1)
        if response.status_code != http.HTTPStatus.OK:
            raise ValueError(f"Could not get model from {self.hostname}:{self.port}")
        return response.json()

    def stop(self):
        response=requests.post(f"http://{self.hostname}:{self.port}/stop",timeout=self.timeout)
        if response.status_code != http.HTTPStatus.OK:
            raise ValueError(f"Could not stop {self.hostname}:{self.port}")
        return response.json()

    def start(self):
        response=requests.post(f"http://{self.hostname}:{self.port}/start",timeout=self.timeout)
        if response.status_code != http.HTTPStatus.OK:
            raise ValueError(f"Could not start {self.hostname}:{self.port}. Status code: {response.status_code}")
        return response.json()

    @property
    def identifier(self):
        return self.hostname

    def json(self):
        return {
            "hostname":self.hostname,
            "port":self.port
        }

    def health(self):
        response=requests.get(f"http://{self.hostname}:{self.port}/health",timeout=self.timeout)
        if response.status_code != http.HTTPStatus.OK:
            return health_dto.NodeHealthResponse(timestamp=format_date(datetime.datetime.now()),hostname=self.hostname,threads=None,
                                                    status=response.status_code)
        health_dto.NodeHealthResponse(**response.json(),status=response.status_code)

class ServiceCluster(Service):
    def __init__(self,services:List[Service],name:str):
        self.services=services
        self.name=name
        self.services_map={service.identifier:service for service in services}

    def heartbeat(self):
        return [service.heartbeat() for service in self.services]

    def stop(self):
        return [service.stop() for service in self.services]

    def update_model(self,path):
        return [service.update_model(path) for service in self.services]

    def get_model(self):
        return [service.get_model() for service in self.services]

    def get_services(self):
        return self.services_map.keys()

    def start(self):
        return [service.start() for service in self.services]

    @property
    def identifier(self):
        return self.name

    def json(self):
        return {
            "name":self.name,
            "services":[service.json() for service in self.services]
        }

    def children(self):
        yield from self.services

    def health(self):
        return [service.health() for service in self.services]
