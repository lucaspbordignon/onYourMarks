class FPSCollector():
    '''
        Class used to measure FPS performance of the tested models. It stores
        a checkpoints list, containing the amount of predictions on the
        given input.
    '''
    def __init__(self, name):
        self._name = name
        self._checkpoints = []
        self._current = 0

    def collect(self, output):
        predictions = output[1]

        if (self._name == 'yolo_v3_coco'):
            predictions = len(predictions)
        else:
            predictions = predictions[0]

        self.update_checkpoint({'predictions': predictions}, override=True)

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
        return ['predictions']
