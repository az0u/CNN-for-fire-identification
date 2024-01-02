from cnn_model import LeNet
from data import load_data
import torch
import time
from visualization import visualize_results

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10
# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the data loader (see data.py folder)
loaders = load_data(BATCH_SIZE)
train_loader = loaders["train_loader"]
test_loader = loaders["test_loader"]

# calculate steps per epoch for training and testing set
train_steps = len(train_loader.dataset) // BATCH_SIZE # number of batches that we split the training dataset into 
test_steps = len(test_loader.dataset) // BATCH_SIZE # number of batches that we split the test dataset into

# initialize the LeNet model 
print("[INFO] Initializing the LeNet model...")
model = LeNet(
	numChannels=3,
	classes=len(train_loader.dataset.classes)
).to(device)

# initialize our optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR)
lossFn = torch.nn.NLLLoss()

# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc": [],
	"test_loss": [],
	"test_acc": []
}

# measure how long training is going to take
print("[INFO] Training the network...")
start_time = time.time()


# loop over the passes (epochs)
for i in range(0, EPOCHS):
    # set the model in training mode 
    model.train()

    # initialize the total training and testing loss 
    totalTrainLoss = 0
    totalTestLoss = 0

    # initialize the number of correct predictions in the training and testing step
    train_correct = 0
    test_correct = 0

    # loop over the training set 
    for (x, y) in train_loader: 
        #send input to the device 
        (x, y) = (x.to(device), y.to(device))

        # perform a forward pass + calculate training loss
        pred = model(x)
        loss = lossFn(pred, y)

        # zero out the gradients, perform the backpropagation step and updaye the weights 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add the loss to the total training loss so far and calculate the number of correct predictions 
        totalTrainLoss += loss 
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # switch off autograd for evaluation 
    # ^ (to help reduce memory consumption and make computations faster)
    with torch.no_grad(): 
        # set the model in evaluation mode 
        model.eval()

        # loop over the testing set 
        for (x, y) in test_loader: 
            #send input to the device 
            (x, y) = (x.to(device), y.to(device))

            # perform a forward pass + calculate test loss
            pred = model(x)
            loss = lossFn(pred, y)

            # add the loss to the total testing loss so far 
            totalTestLoss += loss 
            # calculate the number of correct predictions 
            test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # moving on to some statistical estimations
    # calculate the averadge training and testing loss 
    avgTrainLoss = totalTrainLoss / train_steps
    avgTestLoss = totalTestLoss / test_steps

    # calculate the trainind and testing accuracy 
    train_correct = train_correct / len(train_loader.dataset)
    test_correct = test_correct / len(test_loader.dataset)

    # update training history 
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(train_correct)
    H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
    H["test_acc"].append(test_correct)

    # print the model training and testing infrormation 
    print("[INFO] EPOCH: {}/{}". format(i + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, train_correct))
    print("Testing loss: {:.6f}, Testing accuracy: {:.4f}".format(avgTestLoss, test_correct))

    # measure and show how much time training took
    end_time = time.time()
    print("[INFO] Total amount of time it took to train the model: {:.2f}s".format(end_time - start_time))


visualize_results(history=H, epochs=EPOCHS)