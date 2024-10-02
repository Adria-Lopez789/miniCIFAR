import argparse
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding="same")
        self.conv2 = nn.Conv2d(32, 64, 3, padding="same")
        self.conv3 = nn.Conv2d(64, 128, 3, padding="same")
        self.conv4 = nn.Conv2d(128, 128, 3, padding="same")
        self.conv5 = nn.Conv2d(128, 256, 3, padding="same")
        self.conv6 = nn.Conv2d(256, 256, 3, padding="same")
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.35)
        self.fc1 = nn.Linear(256*4*4, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)#Convolution matrix with a 3x3 kernel
        x = F.relu(x)#Eliminate all negative values (introduces non-linearity)
        x = self.conv2(x)#Convolution matrix with a 3x3 kernel
        x = F.relu(x)#Eliminate all negative values (introduces non-linearity)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(x)#Eliminate all negative values (introduces non-linearity)
        x = self.conv4(x)#Convolution matrix with a 3x3 kernel
        x = F.relu(x)#Eliminate all negative values (introduces non-linearity)
        x = self.pool(x)
        
        x = self.conv5(x)
        x = F.relu(x)#Eliminate all negative values (introduces non-linearity)
        x = self.conv6(x)#Convolution matrix with a 3x3 kernel
        x = F.relu(x)#Eliminate all negative values (introduces non-linearity)
        x = self.pool(x)
        
        x = self.dropout(x)#Remove part of the data for the purpose of eliminating fake relationships (overfitting)
        
        x = torch.flatten(x, 1)#Turn all images into a long array
        x = self.fc1(x)#Reduce the size of the array from 64*72*72 to 128
        x = F.relu(x)#Eliminate all negative values (introduces non-linearity)
        x = self.fc2(x)#Reduce the size of the array from 64*72*72 to 128
        x = F.relu(x)#Eliminate all negative values (introduces non-linearity)
        x = self.fc3(x)#Reduce the 128 sized-array to the 10 possibilities of output
        output = F.log_softmax(x, dim=1)#Turn the nn output into probabilities and then turn them into logarithmic, since we are using nll_loss
        return output



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_samples = 0
    correctness_before_optim = 0
    correctness_after_optim = 0
    for batch_id, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)#move the data tensor and target label to the CPU/GPU
        total_samples += args.batch_size
        
        batch_output_before_optim = model(data)#Compute the model output from the input data(forward)
        batch_loss_before_optim = nn.functional.nll_loss(batch_output_before_optim, target)#Compute the loss (difference between the target and the output)
        prediction_before_optim = batch_output_before_optim.argmax(dim=1, keepdim=True)
        correctness_before_optim += prediction_before_optim.eq(target.view_as(prediction_before_optim)).sum().item()#Add to correct whenever the most likely prediction is the same as the ground truth
        wandb.log({
            'batch_loss_before_optim': batch_loss_before_optim.item(),
            'accuracy_before_optimization': correctness_before_optim / ((batch_id + 1) * len(data))
            })
        optimizer.zero_grad()#Reset the gradients of all optimized tensors for the current batch
        batch_loss_before_optim.backward()
        optimizer.step()
        
        batch_output_after_optim = model(data)#Compute the model output from the input data(forward) after the optimization is done
        batch_loss_after_optim = nn.functional.nll_loss(batch_output_after_optim, target)#Compute the loss (difference between the target and the output)
        prediction_after_optim = batch_output_after_optim.argmax(dim=1, keepdim=True)
        correctness_after_optim += prediction_after_optim.eq(target.view_as(prediction_after_optim)).sum().item()#Add to correct whenever the most likely prediction is the same as the ground truth
        wandb.log({
            'batch_loss_after_optim': batch_loss_after_optim.item(),
            'accuracy_after_optimization': correctness_after_optim / ((batch_id + 1) * len(data))
            })
        if batch_id % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_id * len(data), len(train_loader.dataset),
        100. * batch_id / len(train_loader), batch_loss_after_optim.item()))




def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    
    wandb.login(key='6f35bcdd0108305865b662cb61e4572421e36d7a', relogin=True)
    wandb.init(project="pytorch-intro")
    wandb.config.update(args, allow_val_change=True)
    wandb.log
    

    torch.manual_seed(args.seed)


    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('hola')
    else:
        device = torch.device('cpu')
        print('dw')


    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) #Normalize the tensor so the nn works better with means and standard deviations
        ])
    train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('../data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
        main()
