#!/usr/bin/env python3

# TODO: add base callback class

class History:
    """
    Callback that records events into 'History' object
    """
    def __init__(self):
        self.epoch = []
        self.history = {}

    def reset(self):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
