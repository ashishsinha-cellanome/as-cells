import pytorch_lightning as pl
import os
import shutil


class BackupToNASCallback(pl.Callback):
    """
    Copies the 'last.ckpt' and best checkpoints from fast scratch
    to slow NAS storage at the end of every epoch.
    """

    def __init__(self, backup_dir: str):
        super().__init__()
        self.backup_dir = backup_dir
        os.makedirs(self.backup_dir, exist_ok=True)
        print(f"✓ Backup Callback active. Mirroring checkpoints to: {self.backup_dir}")

    def on_train_epoch_end(self, trainer, pl_module):
        # 1. Copy 'last.ckpt' if it exists
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback.last_model_path and os.path.exists(
            checkpoint_callback.last_model_path
        ):
            try:
                # Copy with metadata preservation
                shutil.copy2(
                    checkpoint_callback.last_model_path,
                    os.path.join(self.backup_dir, "last.ckpt"),
                )
            except Exception as e:
                print(f"Warning: Failed to backup last.ckpt: {e}")

        # 2. Copy the current best model if it exists
        if checkpoint_callback.best_model_path and os.path.exists(
            checkpoint_callback.best_model_path
        ):
            try:
                # Get the filename (e.g., 'epoch=05-val_map=0.45.ckpt')
                best_filename = os.path.basename(checkpoint_callback.best_model_path)
                dest_path = os.path.join(self.backup_dir, best_filename)

                # Only copy if we haven't already (saves bandwidth)
                if not os.path.exists(dest_path):
                    shutil.copy2(checkpoint_callback.best_model_path, dest_path)
                    print(f"✓ Backed up best model to NAS: {best_filename}")
            except Exception as e:
                print(f"Warning: Failed to backup best model: {e}")
