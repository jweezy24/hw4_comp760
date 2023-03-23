import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Define the neural network architecture
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim=1)
        return x

def train():
    net = DNN()
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    
    epochs = 50
    losses_train = []
    losses_test = []
    error_test=[]
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        ave_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            outputs = net(inputs.view(-1, 784)).to(device)
            outputs = outputs.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            ave_loss += loss.item()
            if i % 1000 == 499:    
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
        
        ave_loss/= i
        losses_train.append(ave_loss)


        ave_loss = 0
        with torch.no_grad():
            total = 0
            correct = 0
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                
            
                outputs = net(images.view(-1, 784))
                outputs = outputs.to(device)
                loss = criterion(outputs, labels)
                ave_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
            ave_loss/= total
            losses_test.append(ave_loss)
            error_test.append(1 - (correct/total))

    plt.title("Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(epochs),losses_train,label="Training")
    plt.legend()
    plt.savefig("learning_curve_pytorch.pdf")

    plt.clf()

    plt.title("Test Errors as a Function of Epochs ")
    plt.xlabel("Epochs")
    plt.ylabel("Errors")
    plt.plot(range(epochs),error_test)
    plt.savefig("error_rates_pytorch.pdf")

if __name__ == "__main__":
    train()
