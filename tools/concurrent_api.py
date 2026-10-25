"""Two API workers, one spending guard, one request-start rate limiter.

Each worker owns its ledger/cache; the main pipeline alone saves predictions.
Reuse the existing API client, prompts, complete-response checks and retries.
"""
import copy
import queue
import threading
import time
from contextlib import contextmanager

import reader
from storage import fingerprint


@contextmanager
def network_pacing(interval):
    """Process-local transport wrapper; keep >= interval seconds between all starts."""
    original = reader.urlopen
    lock = threading.Lock()
    last_start = [None]

    def paced_urlopen(*args, **kwargs):
        with lock:
            if last_start[0] is not None:
                delay = interval - (time.perf_counter() - last_start[0])
                if delay > 0:
                    time.sleep(delay)
            last_start[0] = time.perf_counter()
        # Release the gate before waiting on the server: two calls may overlap.
        return original(*args, **kwargs)

    reader.urlopen = paced_urlopen
    try:
        yield
    finally:
        reader.urlopen = original


class TwoWorkerClient:
    def __init__(self, config, role, run, tokenizer, allow_api):
        self.allow_api = allow_api
        self.config = config[role]
        self.guard = reader.CostGuard(run / 'spending_reservations.json', tokenizer, config['cost_reserve_cny'])
        self.available = queue.Queue()
        self.offline = []
        self.key_locks = {}
        self.key_lock = threading.Lock()
        if allow_api:
            reader.load_keys()
        for index in range(2):
            folder = run / 'api' / role / f'worker_{index + 1}'
            self.offline.append(reader.CachedChatClient(self.config, folder, offline=True))
            worker_config = copy.deepcopy(self.config)
            # Split the existing aggregate attempt cap; never double it.
            worker_config['max_api_attempts'] = self.config['max_api_attempts'] // 2
            self.available.put(reader.CachedChatClient(worker_config, folder, offline=not allow_api))

    def complete(self, messages, max_tokens, json_mode=False):
        identity = fingerprint([self.config, messages, max_tokens, json_mode])
        with self.key_lock:
            lock = self.key_locks.setdefault(identity, threading.Lock())
        with lock:  # Duplicate concurrent inputs share one actual request.
            for cached in self.offline:
                try:
                    return cached.complete(messages, max_tokens, json_mode)
                except RuntimeError as error:
                    if 'Offline replay is missing' not in str(error):
                        raise
            if not self.allow_api:
                raise RuntimeError('Offline replay is missing a cached response. No API call was made.')
            worker = self.available.get()
            try:
                reservation = self.guard.reserve(self.config['model'], messages, max_tokens)
                before = len(worker.ledger)
                try:
                    return worker.complete(messages, max_tokens, json_mode)
                finally:
                    self.guard.settle(reservation, worker.ledger[before:])
            finally:
                self.available.put(worker)


def bounded_results(function, items):
    """At most two tasks in flight; stop scheduling on error and save finished peers.

Yields (input, result) in completion order. Callers write files in this main
thread and restore sample order before computing metrics.
"""
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    iterator = iter(items)
    failure = None
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {}
        for _ in range(2):
            item = next(iterator, None)
            if item is not None:
                pending[pool.submit(function, item)] = item
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Inspect the entire completed batch before scheduling any new work.
            success = []
            for future in done:
                item = pending.pop(future)
                try:
                    success.append((item, future.result()))
                except Exception as error:
                    failure = failure or error
            for item, result in success:
                yield item, result
            if failure is None:
                while len(pending) < 2:
                    item = next(iterator, None)
                    if item is None:
                        break
                    pending[pool.submit(function, item)] = item
        if failure is not None:
            raise failure
