import pandas as pd

# EE file inte main use s- closeness kandethenda time window select cheyan anu

class TimeWindow:
    def __init__(self, window_size):
        self.window_size = pd.Timedelta(minutes=window_size)
        self.window_start = None

    def start(self, timestamp):
        self.window_start = timestamp

    def is_complete(self, timestamp):
        if self.window_start is None:
            return False

        return timestamp - self.window_start >= self.window_size

    def reset(self, timestamp):
        self.window_start = timestamp