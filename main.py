import argparse
import os
import os.path
import torch
import torch.nn.functional as F
import torch.optim as optim
import model as m
from torch.autograd import Variable
from dataset import FashionAI

import matplotlib
import pickle
import copy

import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# Training settings
parser = argparse.ArgumentParser(description='FashionAI')
parser.add_argument('--model', type=str, default='resnet34', metavar='M',
                    help='model name')
parser.add_argument('--attribute', type=str, default='coat_length_labels', metavar='A',
                    help='fashion attribute (default: coat_length_labels)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0, metavar='M',
                    help='SGD momentum (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--ci', action='store_true', default=False,
                    help='running CI')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
print("Loading trainset")
trainset = FashionAI('./', attribute=args.attribute, split=0.8, ci=args.ci, data_type='train', reset=False)
print("Loading testset")
testset = FashionAI('./', attribute=args.attribute, split=0.8, ci=args.ci, data_type='test', reset=trainset.reset)
print("Creating train loader")
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
print("Test loader")
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

if args.ci:
    args.model = 'ci'

print("Loading a model for training")
model = m.create_model(args.model, FashionAI.AttrKey[args.attribute])
print("Loading save folder")
save_folder = os.path.join(os.path.expanduser('.'), 'save', args.attribute, args.model)
print("Check point folder check")
if os.path.exists(os.path.join(save_folder, args.model + '_checkpoint.pth')):
    start_epoch = torch.load(os.path.join(save_folder, args.model + '_checkpoint.pth'))
    model.load_state_dict(torch.load(os.path.join(save_folder, args.model + '_' + str(start_epoch) + '.pth')))
else:
    start_epoch = 0

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # torch.save(model.state_dict(), os.path.join(save_folder, args.model + '_' + str(epoch) + '.pth'))
    # torch.save(epoch, os.path.join(save_folder, args.model + '_checkpoint.pth'))
    train_loss /= len(train_loader.dataset)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return {'loss': train_loss, 'accuracy': 100. *correct/ len(train_loader.dataset)}

best_accuracy = 0

def test():
    global best_accuracy
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = data, target
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    current_accuracy = 100. * correct / len(test_loader.dataset)
    if best_accuracy > current_accuracy:
        print("Saving model current "+str(current_accuracy)+" "+"last best "+str(best_accuracy))
        best_accuracy = current_accuracy
        torch.save(model.state_dict(), os.path.join(save_folder, args.model + '_' + str(epoch) + '.pth'))
        torch.save(epoch, os.path.join(save_folder, args.model + '_checkpoint.pth'))
    return {'loss':test_loss, 'accuracy':100. *correct/ len(test_loader.dataset)}

def save_fig(name_fig, tight_layout=True):
    path = os.path.join("./", "images", name_fig + ".png")
    print("Saving figure", name_fig)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

train_loss = []
train_accuracy = []

test_loss = []
test_accuracy = []
print("Starting training")
for epoch in range(start_epoch + 1, args.epochs + 1):
    loss_acc = train(epoch)
    train_loss.append(copy.deepcopy(loss_acc.get('loss')))
    train_accuracy.append(copy.deepcopy(loss_acc.get('accuracy')))
    tloss_acc = test()
    test_loss.append(copy.deepcopy(tloss_acc.get('loss')))
    test_accuracy.append(copy.deepcopy(tloss_acc.get('accuracy')))
train_loss_acc = {'acc':train_accuracy, 'loss':train_loss}
test_loss_acc = {'acc':test_accuracy, 'loss':test_loss}
pickle.dump(train_loss_acc, open("train_metrics.p","wb"))
pickle.dump(test_loss_acc, open("test_metrics.p","wb"))
