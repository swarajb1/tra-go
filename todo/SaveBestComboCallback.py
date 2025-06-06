import tensorflow as tf


class SaveBestComboCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, monitor="loss+val_loss", mode="min", save_best_only=True):
        super().__init__()
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = float("inf") if mode == "min" else -float("inf")
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        combo = loss + val_loss  # Example combination function: simple sum

        if self.mode == "min" and combo < self.best:
            self.best = combo
            if self.save_best_only:
                self.best_weights = self.model.get_weights()
            else:
                self.model.save(self.save_path.format(epoch=epoch, loss=loss, val_loss=val_loss))
        elif self.mode == "max" and combo > self.best:
            self.best = combo
            if self.save_best_only:
                self.best_weights = self.model.get_weights()
            else:
                self.model.save(self.save_path.format(epoch=epoch, loss=loss, val_loss=val_loss))

    def on_train_end(self, logs=None):
        if self.save_best_only and self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            self.model.save(self.save_path)


# Usage
model_checkpoint = SaveBestComboCallback(save_path="model_best_combo.h5")


how_can_i_remove_password_protection_from_a_pdf_file_using_python
