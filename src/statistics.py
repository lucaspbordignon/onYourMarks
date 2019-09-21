from timer import Timer
from collectors.cpu_collector import CPUCollector
from collectors.mem_collector import MemCollector


class Statistics():
    def __init__(self, model_name):
        self._collectors = {
            'timer': Timer(model_name),
            'cpu': CPUCollector(model_name),
            'memory': MemCollector(model_name)
        }

    def start(self):
        self._collectors['timer'].start()

    def end(self):
        self._collectors['timer'].end()
        self._collectors['memory'].collect()

    @property
    def timer(self):
        return self._collectors['timer']
