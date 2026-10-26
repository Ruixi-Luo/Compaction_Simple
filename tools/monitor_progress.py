"""Read-only Windows monitoring; open live files with FILE_SHARE_DELETE.

Run this helper instead of PowerShell Get-Content on live experiment files.
It does not import the pipeline, create clients, or write any files.
"""
import ctypes
import json
import os
from pathlib import Path
from datetime import datetime


def open_shared_read(path):
    if os.name != 'nt':
        return open(path, 'rb')
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    # GENERIC_READ; SHARE_READ | SHARE_WRITE | SHARE_DELETE; OPEN_EXISTING.
    handle = create(str(Path(path).resolve()), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except Exception:
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle(handle)
        raise
    return os.fdopen(descriptor, 'rb')


def read_shared_json(path):
    with open_shared_read(path) as file:
        raw = file.read()
    # Close the handle before decoding/parsing; some Windows volumes still block
    # replacement while a shared-delete handle is open. Writers retry briefly.
    return json.loads(raw.decode('utf-8-sig'))


def main():
    root = Path(__file__).resolve().parents[1]
    status = {'now': datetime.now().astimezone().isoformat(), 'runs': {}}
    queue = root / '.cache/qwen_summary_queue_state.json'
    if queue.exists() and (root/'runs/once100_qwen_summary_20261025').exists():
        status['queue'] = read_shared_json(queue)
    for run in sorted((root/'runs').iterdir()):
        if not run.is_dir():
            continue
        name = run.name
        if not run.exists():
            continue
        entry = {'report_exists': (run / 'Report.html').exists(), 'datasets': {}, 'api': {}}
        if (run / '.run.lock').exists():
            entry['lock'] = read_shared_json(run / '.run.lock')
        for dataset in ('hotpotqa', 'qasper'):
            counts = {}
            for file in (run / dataset / 'compressions').glob('*.json'):
                method = read_shared_json(file)['request']['method']
                counts[method] = counts.get(method, 0) + 1
            predictions = run / dataset / 'predictions.json'
            entry['datasets'][dataset] = {
                'compressions': counts,
                'predicted_questions': len(read_shared_json(predictions)) if predictions.exists() else 0,
                'metrics_exists': (run / dataset / 'metrics.json').exists(),
            }
        for role in ('summary', 'reader'):
            paths = sorted((run / 'api' / role).rglob('api_attempts.json'))
            if paths:
                ledger = [item for path in paths for item in read_shared_json(path)]
                ledger.sort(key=lambda item: item['started_utc'])
                counts = {}
                for item in ledger:
                    counts[item['status']] = counts.get(item['status'], 0) + 1
                entry['api'][role] = {'worker_ledgers': len(paths), 'statuses': counts,
                    'last': {key: ledger[-1].get(key) for key in ('status', 'started_utc', 'error_type')} if ledger else None}
        spending = run / 'spending_reservations.json'
        if spending.exists():
            entry['accounted_cny'] = sum(item['accounted_cny'] for item in read_shared_json(spending))
        completion = run / 'completion.json'
        if completion.exists():
            entry['completion'] = read_shared_json(completion)
        if (run/'telemetry.json').exists():
            entry['telemetry'] = read_shared_json(run/'telemetry.json')
        status['runs'][name] = entry
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
