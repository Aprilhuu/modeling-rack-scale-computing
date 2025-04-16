from pathlib import Path
from pytimeloop.looptree.run import run_looptree

architecture = 'Configs/gpu.yaml'
workload = 'Configs/two_fc.workload.yaml'
mapping = 'Configs/data_parallel_fc.mapping.yaml'

bindings = {
    0: 'MainMemory',
    1: 'GlobalBuffer',
    2: 'MACC'
}

stats = run_looptree(
    Path('configs'),
    [architecture, workload, mapping],
    Path('tmp'),
    bindings,
    call_accelergy=True
)
print('Latency:', stats.latency)
print('Energy:')