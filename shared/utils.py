# shared/utils.py
class Observable:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self, event_type, data=None):
        for obs in self._observers:
            obs.update(event_type, data)