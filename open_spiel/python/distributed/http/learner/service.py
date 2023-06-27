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

    @abc.abstractmethod
    def post(self,endpoint:str,data):
        pass


    @abc.abstractmethod
    def get(self,endpoint:str):
        pass

    @abc.abstractmethod
    def put(self,endpoint:str,data):
        pass

    @abc.abstractmethod
    def delete(self,endpoint:str):
        pass

    @abc.abstractmethod
    def patch(self,endpoint:str,data):
        pass

    @abc.abstractmethod
    def options(self,endpoint:str):
        pass

    @abc.abstractmethod
    def head(self,endpoint:str):
        pass



class ServiceNode(Service):
    def __init__(self,config,hostname:str,port:int,*,skip_dead:bool=False):
        self.hostname = hostname
        self.port = port
        self.config=config
        self.timeout=config.services.timeout
        self._alive=True
        self.skip_dead=skip_dead

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
        if response is None or response.status_code != http.HTTPStatus.OK:
            raise ValueError(f"Could not get model from {self.hostname}:{self.port}")
        return response.json()

    def stop(self):
        response=self.post("/stop",{})
        if response is None:
            return {"status":"dead"}
        elif response.status_code != http.HTTPStatus.OK:
            raise ValueError(f"Could not stop {self.hostname}:{self.port}")
        return response.json()

    def start(self):
        response=self.post("/start",{})
        if response is None:
            return {"status":"dead","name":self.hostname}
        elif response.status_code != http.HTTPStatus.OK:
            raise ValueError(f"Could not start {self.hostname}:{self.port}. Status code: {response.status_code}")
        json_response=response.json()
        json_response["name"]=self.hostname
        return json_response

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

    def post(self,endpoint,data):
        if endpoint.startswith("/"):
            endpoint=endpoint[1:]
        if self.skip_dead and not self._alive:
            return None
        try:
            response=requests.post(f"http://{self.hostname}:{self.port}/{endpoint}",json=data,timeout=self.timeout)
            self._alive=True
        except requests.exceptions.ConnectionError as e:
            self._alive=False
            return
        return response

    def get(self,endpoint):
        if endpoint.startswith("/"):
            endpoint=endpoint[1:]
        if self.skip_dead and not self._alive:
            return None
        try:
            response=requests.get(f"http://{self.hostname}:{self.port}/{endpoint}",timeout=self.timeout)
            self._alive=True
        except requests.exceptions.ConnectionError as e:
            self._alive=False
            return
        return response

    def put(self,endpoint,data):
        if endpoint.startswith("/"):
            endpoint=endpoint[1:]
        if self.skip_dead and not self._alive:
            return None
        try:
            response=requests.put(f"http://{self.hostname}:{self.port}/{endpoint}",json=data,timeout=self.timeout)
            self._alive=True
        except requests.exceptions.ConnectionError as e:
            self._alive=False
            return
        return response

    def delete(self,endpoint):
        if endpoint.startswith("/"):
            endpoint=endpoint[1:]
        if self.skip_dead and not self._alive:
            return None
        try:
            response=requests.delete(f"http://{self.hostname}:{self.port}/{endpoint}",timeout=self.timeout)
            self._alive=True
        except:
            self._alive=False
            return
        return response

    def patch(self,endpoint,data):
        if endpoint.startswith("/"):
            endpoint=endpoint[1:]
        if self.skip_dead and not self._alive:
            return None
        try:
            response=requests.patch(f"http://{self.hostname}:{self.port}/{endpoint}",json=data,timeout=self.timeout)
            self._alive=True
        except requests.exceptions.ConnectionError as e:
            self._alive=False
            return
        return response

    def options(self,endpoint):
        if endpoint.startswith("/"):
            endpoint=endpoint[1:]
        if self.skip_dead and not self._alive:
            return None
        try:
            response=requests.options(f"http://{self.hostname}:{self.port}/{endpoint}",timeout=self.timeout)
            self._alive=True
        except requests.exceptions.ConnectionError as e:
            self._alive=False
            return
        return response

    def head(self,endpoint):
        if endpoint.startswith("/"):
            endpoint=endpoint[1:]
        if self.skip_dead and not self._alive:
            return None
        try:
            response=requests.head(f"http://{self.hostname}:{self.port}/{endpoint}",timeout=self.timeout)
            self._alive=True
        except requests.exceptions.ConnectionError as e:
            self._alive=False
            return
        return response


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
        json_responses={}
        json_responses["name"]=self.name
        json_responses["services"]=[service.start() for service in self.services]
        return json_responses

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
        json_response={}
        json_response["name"]=self.name
        json_response["services"]=[service.health() for service in self.services]
        return json_response

    def post(self,endpoint,data):
        json_response={}
        json_response["name"]=self.name
        json_response["services"]=[service.post(endpoint,data) for service in self.services]
        return json_response

    def get(self,endpoint):
        json_response={}
        json_response["name"]=self.name
        json_response["services"]=[service.get(endpoint) for service in self.services]
        return json_response

    def put(self,endpoint,data):
        json_response={}
        json_response["name"]=self.name
        json_response["services"]=[service.put(endpoint,data) for service in self.services]
        return json_response

    def delete(self,endpoint):
        json_response={}
        json_response["name"]=self.name
        json_response["services"]=[service.delete(endpoint) for service in self.services]

    def patch(self,endpoint,data):
        json_response={}
        json_response["name"]=self.name
        json_response["services"]=[service.patch(endpoint,data) for service in self.services]
        return json_response

    def options(self,endpoint):
        json_response={}
        json_response["name"]=self.name
        json_response["services"]=[service.options(endpoint) for service in self.services]
        return json_response

    def head(self,endpoint):
        json_response={}
        json_response["name"]=self.name
        json_response["services"]=[service.head(endpoint) for service in self.services]
        return json_response
