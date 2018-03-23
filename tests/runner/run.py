import sys
import os
import json
import subprocess

class TestCase:
    def __init__(self, source, expected):
        self.source = source
        self.expected = expected

    def run(self):
        # TODO
        pass

target_dir = sys.argv[1]
print("Running tests in directory " + target_dir)

with open(os.path.join(target_dir, "config.json")) as f:
    config = json.loads(f.read())

# TODO
print(config)
