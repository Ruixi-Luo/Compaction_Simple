"""本项目文件读写；共享资源只读，结果和临时文件写在新项目。"""
import hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def fingerprint(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, value):
    """先写临时文件，再替换；意外退出不会留下半个 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    # Windows readers (editors, antivirus) can briefly block replacement.
    # Retry only the disk operation, never repeat a successful model request.
    for attempt in range(12):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 11:
                raise  # Keep the complete .tmp file for recovery.
            time.sleep(min(.1 * 2**attempt, 1.0))

RESOURCES=read_json(ROOT/'resources.json')
DATA_ROOT=Path(RESOURCES['data_root'])

def download(url,path,expected_sha256=None,offline=False):
    """沿用数据适配器接口，但这里只读取已存在的共享文件，绝不修改旧缓存。"""
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Shared resource missing: {path}; update resources.json.')
    if expected_sha256 and file_hash(path)!=expected_sha256:
        raise ValueError('Shared resource checksum mismatch: '+str(path))
    return path
