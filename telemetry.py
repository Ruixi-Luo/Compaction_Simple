"""Main-thread progress: elapsed time and provider-reported token usage, never API calls."""
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from storage import read_json, write_json
from tools.monitor_progress import read_shared_json


def now():
    return datetime.now(timezone.utc).isoformat()


def api_usage(run):
    """Sum attempt ledgers once; cached reads add no tokens; missing usage stays unknown."""
    totals = {}
    for role in ('summary', 'reader'):
        value = dict(attempts=0, received=0, failed=0, in_flight=0,
                     unknown_usage=0, input_tokens=0, output_tokens=0,
                     cached_input_tokens=0, attempt_seconds=0.0)
        for path in (run/'api'/role).rglob('api_attempts.json'):
            for item in read_shared_json(path):
                value['attempts'] += 1
                status = item['status']
                value['received' if status == 'received' else 'in_flight' if status == 'started' else 'failed'] += 1
                usage = item.get('usage') or {}
                if usage:
                    value['input_tokens'] += usage.get('prompt_tokens', 0)
                    value['output_tokens'] += usage.get('completion_tokens', 0)
                    value['cached_input_tokens'] += (usage.get('prompt_cache_hit_tokens',
                        (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)) or 0)
                elif status != 'started':
                    value['unknown_usage'] += 1
                value['attempt_seconds'] += item.get('elapsed_seconds', 0.0)
        totals[role] = value
    spending = run/'spending_reservations.json'
    totals['accounted_cny_including_reservations'] = sum(
        x['accounted_cny'] for x in read_shared_json(spending)) if spending.exists() else 0.0
    return totals


class Progress:
    """One invocation is a session; pauses between invocations are not execution time."""
    def __init__(self, run, command):
        self.run, self.path = run, run/'telemetry.json'
        self.data = read_json(self.path) if self.path.exists() else {'sessions': []}
        self.start = time.perf_counter()
        self.session = {'started_utc': now(), 'command': command, 'status': 'running',
                        'elapsed_seconds': 0.0, 'stages': [], 'progress': {}}
        # An interrupted invocation retains its last measured duration, not a fabricated end time.
        for previous in self.data['sessions']:
            if previous['status'] == 'running':
                previous['status'] = 'interrupted_duration_lower_bound'
        self.data['sessions'].append(self.session)
        self.current = None
        self.save()

    def save(self):
        self.session['elapsed_seconds'] = time.perf_counter() - self.start
        if self.current:
            self.current['elapsed_seconds'] = time.perf_counter() - self.stage_start
        self.data['updated_utc'] = now()
        self.data['execution_seconds'] = sum(s['elapsed_seconds'] for s in self.data['sessions'])
        self.data['api_usage'] = api_usage(self.run)
        write_json(self.path, self.data)

    def update(self, stage, dataset, done, total, detail=''):
        self.session['progress'][f'{stage}/{dataset}'] = {'done': done, 'total': total}
        self.save()
        usage = self.data['api_usage']
        incoming = sum(usage[r]['input_tokens'] for r in ('summary', 'reader'))
        outgoing = sum(usage[r]['output_tokens'] for r in ('summary', 'reader'))
        print(f'{stage} {dataset} {done}/{total} {detail} | elapsed={self.session["elapsed_seconds"]/60:.1f} min'
              f' | API tokens in/out={incoming:,}/{outgoing:,}', flush=True)

    @contextmanager
    def stage(self, name):
        self.current = {'name': name, 'started_utc': now(), 'elapsed_seconds': 0.0, 'status': 'running'}
        self.session['stages'].append(self.current)
        self.stage_start = time.perf_counter()
        self.save()
        try:
            yield
            self.current['status'] = 'complete'
        except BaseException:
            self.current['status'] = 'failed_or_interrupted'
            raise
        finally:
            self.current['ended_utc'] = now()
            self.save()
            self.current = None

    def finish(self, status):
        self.session.update(status=status, ended_utc=now())
        self.save()
