import csv
from timer import Timer
from collectors.cpu_collector import CPUCollector
from collectors.mem_collector import MemCollector


class Statistics():
    def __init__(self, model_name):
        self._name = model_name
        self._collectors = {
            'timer': Timer(model_name),
            'cpu': CPUCollector(model_name),
            'memory': MemCollector(model_name)
        }

    def start(self):
        self._collectors['timer'].start()

    def end(self):
        self._collectors['timer'].end()
        self._collectors['cpu'].collect()
        self._collectors['memory'].collect()

    def export(self, base_path='./'):
        for metric in self._collectors:
            fields = self._collectors[metric].fields
            checkpoints = self._collectors[metric].checkpoints
            filename = base_path + self._name + '_' + metric + '.csv'

            with open(filename, 'w') as file:
                writer = csv.DictWriter(file, fieldnames=fields)

                writer.writeheader()
                list(map(lambda x: writer.writerow(x), checkpoints))

    @property
    def timer(self):
        return self._collectors['timer']
