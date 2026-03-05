import csv, os


class Logger:
    """writes training stats to a csv so I can plot them later"""

    def __init__(self, path, fields):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=fields)
        self.writer.writeheader()

    def log(self, row):
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()
