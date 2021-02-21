import json
import os, sys

class Jsoner():
    def __init__(self, arr):
        self.arr = arr

    def writer(self):

        json_path = 'config.json'

        _dict = {}

        with open(json_path, 'rb') as f:

            params = json.load(f)
            params['data']['columns'] = self.arr

            params['model']['layser'][0]['input_dim'] = len(self.arr)

            _dict = params

        f.close()

        with open(json_path, 'w') as r:

            json.dump(_dict, r)

        r.close()