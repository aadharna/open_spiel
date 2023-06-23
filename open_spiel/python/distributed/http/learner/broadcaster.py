import open_spiel.python.algorithms.alpha_zero_mpg.utils as mpg_utils
import open_spiel.python.algorithms.alpha_zero_mpg.dto as mpg_dto
import requests


class HttpBroadcaster(mpg_dto.Broadcaster):
    def __init__(self, config, hosts):
        self.config = config
        self.timeout = self.config.http.timeout
        self.hosts = hosts

    def broadcast(self, path):
        for host in self.hosts:
            requests.post(host, path, timeout=self.timeout)
        pass
