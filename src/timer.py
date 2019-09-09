import time


class Timer():
    '''
        Class used to measure performance of the tested models. It stores
        a checkpoints list, containing all the information about the tracked
        times. Each item has the following format:
         - started_formatted: Human ready format to when the timer started;
         - started: Unix timestamp measured in nanoseconds;
         - ended: Unix timestamp measured in nanoseconds;
         - elapsed: Unix timestamp of the difference measured in nanoseconds;
    '''
    def __init__(self, name):
        self._name = name
        self._checkpoints = []
        self._current = 0

    def start(self):
        self.update_checkpoint({
            'started_formatted': time.ctime(),
            'started': time.time()
        }, override=True)

    def end(self):
        now = time.time()

        self._current += 1

        self.update_checkpoint({
            'ended': now,
            'elapsed': now - self.checkpoint['started']
        })

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
