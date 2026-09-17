"""Native multi-GPU models need separate processes, not loader threads."""
import os
from pathlib import Path

from scripts import qwen3_prealign as prealign
from scripts.qwen3_hf_backend import Qwen3HFSettings


class ProcessBackend:
    last_language = 'Chinese'

    def __init__(self, settings, require_asr):
        self.device = settings.device
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        return '你'

    def align(self, audio, text, **kwargs):
        self.calls += 1
        return [dict(unit='你', start_s=0, end_s=.1, pid=os.getpid(),
                     device=self.device, call=self.calls)]

    def close(self):
        pass


def process_backend(settings, **kwargs):
    return ProcessBackend(settings, **kwargs)


def test_each_gpu_has_a_persistent_child_process(monkeypatch):
    monkeypatch.setattr(prealign, 'load_backend', process_backend)
    inputs = [Path(f'{n:03d}.wav') for n in range(16)]
    devices = [f'cuda:{n}' for n in range(8)]
    settings = Qwen3HFSettings(Path('asr'), Path('aligner'), batch_size=2)
    results = list(prealign.infer_qwen_items(inputs, {}, 'fallback', settings, devices=devices))
    assert [path for path, _ in results] == inputs
    rows = [result[-1][0] for _, result in results]
    assert len({r['pid'] for r in rows}) == 8
    assert all(r['pid'] != os.getpid() for r in rows)
    for n, device in enumerate(devices):
        owned = [r for r in rows if r['device'] == device]
        assert len(owned) == 2 and len({r['pid'] for r in owned}) == 1
        assert [r['call'] for r in owned] == [1, 2]
