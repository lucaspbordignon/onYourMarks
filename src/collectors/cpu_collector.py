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
