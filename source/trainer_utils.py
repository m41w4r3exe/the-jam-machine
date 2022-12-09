import matplotlib.pyplot as plt
import numpy as np


def get_history(trainer):
    history = trainer.state.log_history
    train_history = []
    valid_history = []
    for h in history:
        if len(h.items()) == 4:
            train_history.append([h["epoch"], h["step"], h["loss"], h["learning_rate"]])
        elif len(h.items()) == 6:
            valid_history.append([h["epoch"], h["step"], h["eval_loss"]])

    history = {
        "train_epoch": np.array(train_history).T[0],
        "train_loss": np.array(train_history).T[2],
        "learning_rate": np.array(train_history).T[3],
        "valid_epoch": np.array(valid_history).T[0],
        "valid_loss": np.array(valid_history).T[2],
    }

    return history


def plot_history(history, model_path, hf_repo):
    plt.subplots(figsize=(10, 10))
    plt.subplot(211)
    plt.plot(history["train_epoch"], history["learning_rate"], color="black")
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
    plt.title(hf_repo)

    plt.subplot(212)
    plt.plot(
        history["train_epoch"],
        history["train_loss"],
        color="purple",
        label="training loss",
    )
    plt.plot(
        history["valid_epoch"],
        history["valid_loss"],
        color="orange",
        label="validation loss",
    )
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()
    plt.savefig(f"{model_path}/training_history.png")
    plt.close()
