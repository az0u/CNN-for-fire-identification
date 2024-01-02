from torchvision import datasets, transforms
import torch

def load_data(batchsize):
    transform = transforms.Compose([
        # make them smaller so the model can proccess them easier
        transforms.Resize((32, 32)), 
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(r'./fires/forest_fire/Training and Validation', transform=transform)
    test_dataset = datasets.ImageFolder(r'./fires/forest_fire/Testing', transform=transform)

    # data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

    return {
        "train_loader": train_loader, 
        "test_loader": test_loader
    }