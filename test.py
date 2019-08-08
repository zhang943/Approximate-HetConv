import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from vgg import vgg16_bn


def args_parse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
    parser.add_argument('--P', default=64, type=int, help='Part')
    args = parser.parse_args()

    return args


def main():
    args = args_parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False)

    # Model
    print('==> Building model..')
    net = vgg16_bn(p=args.P)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    checkpoint = torch.load('./model/vgg16_P{}.pth'.format(args.P))
    net.load_state_dict(checkpoint['net'])

    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(testloader, desc='Testing ', unit='batch') as loader:
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                loader.set_postfix(info='Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                        % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            loader.close()

    # acc = 100. * correct / total


if __name__ == '__main__':
    main()
