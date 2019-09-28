import platform
import psutil


class MemCollector():
    '''
        Class used to measure Memory usage of the tested models. It stores
        a checkjoints list, containing all the information about the tracked
        memory usage.

        Worth notice that all metrics stored all measured in bytes.
    '''
    def __init__(self, name):
        self._process = psutil.Process()
        self._name = name
        self._checkpoints = []
        self._current = 0

    def collect(self):
        full_info = self._process.memory_full_info()

        self.update_checkpoint({
            'percentage': self._process.memory_percent(),
            'process_total': full_info.uss,
            'ram_total': full_info.rss,
        }, override=True)

        if (platform.system() == 'Linux'):
            self.update_checkpoint({'swap_total': full_info.swap})

        self._current += 1

    def update_checkpoint(self, data, override=False):
        if (override):
            self._checkpoints.append(data)

        current_data = self.checkpoint

        self._checkpoints[self._current - 1] = {**current_data, **data}

    @property
    def checkpoint(self):
        return self._checkpoints[self._current - 1]

    @property
    def checkpoints(self):
        return self._checkpoints

    @property
    def fields(self):
        fields = ['percentage', 'process_total', 'ram_total']

        if (platform.system() == 'Linux'):
            fields.append('swap_total')

        return fields
