"""只读共享模型；在新项目设置临时缓存与 CUDA。"""
import os,tempfile
from pathlib import Path
from storage import ROOT, DATA_ROOT, RESOURCES, read_json

def configure_cache():
    for key,folder in {'HF_HOME':'huggingface','TIKTOKEN_CACHE_DIR':'tiktoken','TORCH_HOME':'torch','TEMP':'tmp','TMP':'tmp'}.items():
        path=ROOT/'.cache'/folder;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    os.environ['NLTK_DATA']=RESOURCES['nltk_data']
    tempfile.tempdir=os.environ['TEMP']
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')

def model_directory(method):
    spec=read_json(ROOT/'models.lock.json')[method]
    folder=DATA_ROOT/spec['directory'].removeprefix('data/')
    if not (folder/'model.safetensors').exists():raise FileNotFoundError(folder)
    return folder

def verify_assets(methods):
    # 固定模型文件清单；缺失/大小不符就停止，不触发下载。
    for method in methods:
        folder=model_directory(method)
        for name,entry in read_json(ROOT/'models.lock.json')[method]['files'].items():
            p=folder/name
            if not p.is_file() or p.stat().st_size!=entry['bytes']:raise ValueError('Model file mismatch: '+str(p))

def prepare_torch(device='cpu'):
    configure_cache()
    import torch
    if device not in {'cpu', 'cuda'}:
        raise ValueError('compression_device must be cpu or cuda.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Install CUDA PyTorch with setup-local.ps1 -Device cuda.')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch
