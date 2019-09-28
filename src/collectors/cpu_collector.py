import platform
import psutil


class CPUCollector():
    '''
        Class used to measure CPU performance of the tested models. It stores
        a checkpoints list, containing all the information about the tracked
        CPU usage.
    '''
    def __init__(self, name):
        self._name = name
        self._checkpoints = []
        self._current = 0

    def collect(self):
        if (platform.system() == 'Linux'):
            temperatures = psutil.sensors_temperatures()

            self.update_checkpoint({
                'gpu_temperature': temperatures['GPU-therm'][0].current,
                'cpu_temperature': temperatures['MCPU-therm'][0].current,
            }, override=True)

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
        return ['temperature']
