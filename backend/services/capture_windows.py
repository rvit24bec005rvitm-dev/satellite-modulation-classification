"""Small job-local LRU of read-only mappings; windows are always bounded copies."""
from collections import OrderedDict

import numpy as np

from backend.services import workspace_store as store


class CaptureWindows:
    def __init__(self, capacity=8):
        if not 1 <= capacity <= 8:
            raise ValueError('A worker may retain at most eight read-only capture mappings.')
        self.capacity = capacity
        self.mappings = OrderedDict()

    def window(self, capture_id, start):
        capture_id = store.identifier(capture_id)
        with store.LOCK:
            if capture_id in self.mappings:
                mapped, metadata = self.mappings.pop(capture_id)
                self.mappings[capture_id] = (mapped, metadata)
            else:
                metadata = store.get('capture', capture_id)
                mapped = np.memmap(store.file_path(capture_id, 'iq'), dtype='<f4', mode='r',
                                   shape=(metadata['sampleCount'], 2))
                self.mappings[capture_id] = (mapped, metadata)
                if len(self.mappings) > self.capacity:
                    _, (old, _) = self.mappings.popitem(last=False)
                    old._mmap.close()
            if start < 0 or start + 128 > metadata['sampleCount']:
                raise ValueError('Requested model window lies outside the stored capture.')
            return np.array(mapped[start:start + 128], copy=True), metadata

    def close(self):
        for mapped, _ in self.mappings.values():
            mapped._mmap.close()
        self.mappings.clear()
