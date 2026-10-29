import datetime, json, os, platform, random
import numpy as np, yaml

DIRS = {'interim': 'data/interim', 'splits': 'data/splits',
        'cache': 'data/cache', 'runs': 'runs', 'results': 'results', 'audit': 'results/audit',
        'figures': 'results/figures', 'gates': 'gates'}

def set_seeds(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def environment():
    env = {'python': platform.python_version(), 'numpy': np.__version__,
           'time': datetime.datetime.now().isoformat(timespec='seconds')}
    try:
        import torch
        env.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        env['torch'] = None
    return env

def load(path, seed=None, verbose=True):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    root = os.path.dirname(os.path.abspath(path))
    cfg['paths'] = {'root': root, 'raw': cfg['data']['raw_dir'], 'local_data': cfg['data']['local_dir'],
                    **{k: os.path.join(root, v) for k, v in DIRS.items()}}
    cfg['seed'] = cfg['seeds'][0] if seed is None else seed
    set_seeds(cfg['seed'])
    if verbose:
        e = environment()
        print(f"Root {root} | seed {cfg['seed']} | Python {e['python']} | torch {e.get('torch')} | GPU {e.get('gpu') or 'none (CPU)'}")
    return cfg

def freeze(cfg, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.frozen.yaml'), 'w') as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if k != 'paths'}, f, sort_keys=False)
    with open(os.path.join(run_dir, 'environment.json'), 'w') as f:
        json.dump(environment(), f, indent=2)
