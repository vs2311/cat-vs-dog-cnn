"""Train the classifier.   Usage: python -m src.train"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from . import config
from .data import load_train, load_val
from .model import build_model


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


def plot_history(history, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key, title in zip(ax, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        a.plot(history[key], label="train")
        a.plot(history["val_" + key], label="val")
        a.set_title(title)
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.OUTPUTS_DIR.mkdir(exist_ok=True)

    train_ds, val_ds = load_train(), load_val()
    model, base = build_model()
    early = keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True, monitor="val_loss"
    )

    # Stage 1: train the new head only
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=config.HEAD_EPOCHS, callbacks=[early])

    # Stage 2: fine-tune the top layers with a small learning rate
    base.trainable = True
    for layer in base.layers[: -config.FINE_TUNE_LAYERS]:
        layer.trainable = False
    compile_model(model, 1e-5)
    h2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=config.FINE_TUNE_EPOCHS, callbacks=[early])

    history = {k: h1.history[k] + h2.history[k] for k in h1.history}
    plot_history(history, config.OUTPUTS_DIR / "training_curves.png")

    model.save(config.MODEL_PATH)
    print(f"Saved model -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
