from src.training.callbacks import base
from torch import save, onnx
import typing
import pathlib
import os

class SnapshotCallback(base.BaseCallback):
    """
    Callback for tracking training epoch snapshots.
    """
    def __init__(self, 
        snapshot_ext: typing.Literal['onnx', 'pth', 'pt'],
        save_every: int,
        log_dir: typing.Union[str, pathlib.Path]
    ):
        super(SnapshotCallback, self).__init__(log_dir=log_dir)
        self.snapshot_ext = snapshot_ext
        self.snapshot_dir = log_dir
        self.save_every = save_every

    def on_epoch_end(self, **kwargs):

        global_step = kwargs.get("global_step")
        snapshot = kwargs.get("snapshot_info")
        
        if global_step % self.save_every == 0:
            if self.snapshot_ext in ('pt', 'pth'):
                save(
                    obj=snapshot,
                    f=os.path.join(self.snapshot_dir, 'epoch_%s.pt' % global_step)
                )
            if self.snapshot_ext == 'onnx':
                onnx.export(
                    model=snapshot['network'],
                    args=snapshot['test_input'],
                    f=os.path.join(self.snapshot_dir, 'epoch_%s.onnx' % global_step)
                )
