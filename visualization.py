# Import matplotlib -> visulization
import matplotlib.pyplot as plt

def visualize_results(history, epochs):
    fig, axs = plt.subplots(2)
    epochs_list = list(range(1, epochs+1))

    axs[0].plot(epochs_list, history["train_acc"], 'g-', label="train_accuracy")
    axs[0].plot(epochs_list, history["test_acc"], 'b-', label="test_accuracy")
    axs[0].set_title("Accuracy of the Model", fontweight="bold")
    axs[0].set_xlabel("Number of Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xticks(epochs_list)
    axs[0].legend(loc="best")

    axs[1].plot(epochs_list, history["train_loss"], 'r-', label="train_loss")
    axs[1].plot(epochs_list, history["test_loss"], 'm-', label="test_loss")
    axs[1].set_title("Loss of the Model", fontweight="bold")
    axs[1].set_xlabel("Number of Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].set_xticks(epochs_list)
    axs[1].legend(loc="best")
    plt.tight_layout()
    plt.show()