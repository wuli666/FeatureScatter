import argparse
import time
import numpy as np
import torchvision
from torch import optim
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import os
import torch
from attack_methods import Attack_FeaScatter
from dataset_d import generate_dataset
from models.MLP import MLP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

parser = argparse.ArgumentParser(description='PyTorch Adversarial Training with Automatic Noisy Labels Injection')
### Experimental setting ###
# add type keyword to registries


parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--adv_mode',
                    default='feature_scatter',
                    type=str,
                    help='adv_mode (feature_scatter)')
parser.add_argument('--model_dir', type=str, help='model path')
parser.add_argument('--init_model_pass',
                    default='-1',
                    type=str,
                    help='init model pass (-1: from scratch; K: checkpoint-K)')
parser.add_argument('--max_epoch',
                    default=200,
                    type=int,
                    help='max number of epochs')
parser.add_argument('--save_epochs', default=10, type=int, help='save period')
parser.add_argument('--decay_epoch1',
                    default=60,
                    type=int,
                    help='learning rate decay epoch one')
parser.add_argument('--decay_epoch2',
                    default=90,
                    type=int,
                    help='learning rate decay point two')
parser.add_argument('--decay_rate',
                    default=0.1,
                    type=float,
                    help='learning rate decay rate')
parser.add_argument('--batch_size_train',
                    default=100,
                    type=int,
                    help='batch size for training')
parser.add_argument('--batch_size_test',
                    default=128,
                    type=int,
                    help='batch size for training')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    help='momentum (1-tf.momentum)')
parser.add_argument('--weight_decay',
                    default=2e-4,
                    type=float,
                    help='weight decay')
parser.add_argument('--log_step', default=10, type=int, help='log_step')
parser.add_argument('--gpu', default=0, type=int, help='log_step')

# number of classes and image size will be updated below based on the dataset
parser.add_argument('--num_classes', default=3, type=int, help='num classes')
parser.add_argument('--image_size', default=32, type=int, help='image size')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')  # concat cascade
args = parser.parse_args()
print(args)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
])
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform_test)
classes = ('plane', 'car', 'bird')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# basic_net = MLP(32, 100, 3)
# basic_net = basic_net.to(device)

# config for feature scatter
config_feature_scatter = {
    'train': True,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 1,
    'step_size': 8.0 / 255 * 2,
    'random_start': True,
    'ls_factor': 0.5,
}

# net = Attack_FeaScatter(basic_net, config_feature_scatter)
# input_list = []

# print(target_list)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size_train,
                                          shuffle=True,
                                          num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
net = MLP(num_classes=3).to(device)

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# iterator = tqdm(trainloader, ncols=0, leave=False)
inputs_list = [[] for _ in range(args.num_classes)]
optimizer = optim.SGD(net.parameters(),
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)
# for batch_idx, (inputs, targets) in enumerate(iterator):
#
#     for i in range(args.num_classes):
#         target_list[i].append(targets[targets == i].detach())
#         inputs_list[i].append(inputs[targets == i].detach())
#
#
# for i in range(args.num_classes):
#     # print(len(target_list))
#     target_list[i] = torch.cat(target_list[i], dim=0).detach()[:100]
#     inputs_list[i] = torch.cat(inputs_list[i], dim=0).detach()[:100]
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
class_ids = [0, 1, 2]
dataset = generate_dataset(train_loader=trainloader, num_classes=args.num_classes, class_ids=class_ids, transform=None)
testset1 = generate_dataset(train_loader=testloader, num_classes=args.num_classes, class_ids=class_ids, transform=None)
trainloader1 = torch.utils.data.DataLoader(dataset,
                                           batch_size=args.batch_size_train,
                                           shuffle=True,
                                           num_workers=2)
testloader1 = torch.utils.data.DataLoader(dataset,
                                          batch_size=args.batch_size_train,
                                          shuffle=True,
                                          num_workers=2)

outputs_list = []
targetss_list = []


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader1):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)

        for i, j in zip(outputs, targets):
            outputs_list.append(i.detach().numpy())
            targetss_list.append(j.detach().numpy())

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    if epoch % args.save_epochs == 0 or epoch >= args.max_epoch - 2:
        print('Saving..')
        f_path = os.path.join(args.model_dir, ('checkpoint-%s' % epoch))
        state = {
            'net': net.state_dict(),
            # 'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        torch.save(state, f_path)

    if epoch >= 0:
        print('Saving latest @ epoch %s..' % (epoch))
        f_path = os.path.join(args.model_dir, 'latest')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        torch.save(state, f_path)


def test(epoch, net):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    iterator = tqdm(testloader1, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)

        pert_inputs = inputs.detach()

        outputs = net(pert_inputs)

        loss = criterion(outputs, targets)
        test_loss += loss.item()

        duration = time.time() - start_time

        _, predicted = outputs.max(1)
        for i, j in zip(outputs, predicted):
            outputs_list.append(i.detach().numpy())
            targetss_list.append(j.detach().numpy())
        print(targets)
        batch_size = targets.size(0)
        total += batch_size
        correct_num = predicted.eq(targets).sum().item()
        correct += correct_num
        iterator.set_description(
            str(predicted.eq(targets).sum().item() / targets.size(0)))

        if batch_idx % args.log_step == 0:
            print(
                "step %d, duration %.2f, test  acc %.2f, avg-acc %.2f, loss %.2f"
                % (batch_idx, duration, 100. * correct_num / batch_size,
                   100. * correct / total, test_loss / total))

    acc = 100. * correct / total
    print('Val acc:', acc)
def point_gene(point_list):
    x,y,z=[],[],[]

    for k in point_list:
        x.append(k[0])
        y.append(k[1])
        z.append(k[2])
    print(x)
    return x,y,z

def plot(x, y, z):




    # for j in y:
    #     x2.append(j[0])
    #     y2.append(j[1])
    #     z2.append(j[2])
    #
    # for k in z:
    #     x3.append(k[0])
    #     y3.append(k[1])
    #     z3.append(k[2])

    # z = 4 * np.tan(np.random.randint(10, size=(500))) + np.random.randint(100, size=(500))
    #
    # x = 4 * np.cos(z) + np.random.normal(size=500)
    # y = 4 * np.sin(z) + 4 * np.random.normal(size=500)

    # print(target)

    # Creating figure
    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection="3d")
    # Add x, and y gridlines for the figure
    # ax.grid(b=True, color='blue', linestyle='-.', linewidth=0.5, alpha=0.3)
    # Creating the color map for the plot
    # my_cmap = plt.get_cmap('hsv')

    # Creating the 3D plot
    j,k,h = point_gene(x)
    sctt = ax.scatter3D(j,k,h, c='green')
    #
    j,k,h= point_gene(y)
    sctt = ax.scatter3D(j,k,h, c='red',marker='^')
    #
    j,k,h = point_gene(z)
    sctt = ax.scatter3D(j,k,h, c='blue',marker='s')
    # plt.title("train")
    # ax.set_xlabel('X-axis', fontweight='bold')
    # ax.set_ylabel('Y-axis', fontweight='bold')
    # ax.set_zlabel('Z-axis', fontweight='bold')
    # fig.colorbar(sctt, ax = ax, shrink = 0.6, aspect = 5)
    # display the plot
    plt.savefig('./train.png')
    plt.show()



list_0 = []
list_1 = []
list_2 = []

list_target = []
for epoch in range(0, 1):
    train(epoch)

    # test(epoch,net)
    for j in range(len(targetss_list)):
        list_target.append(targetss_list[j].tolist())

    for i in range(len(outputs_list)):
        if list_target[i] == 0:
            list_0.append(outputs_list[i].tolist())
        if list_target[i] == 1:
            list_1.append(outputs_list[i].tolist())
        if list_target[i] == 2:
            list_2.append(outputs_list[i].tolist())
        # else:
        #     list_2.append(outputs_list[i].tolist())

    # for j in targetss_list:
    #     print(j)
    # list_x0.append(outputs_list[i][0])
    # list_y0.append(outputs_list[i][1])
    # list_z0.append(outputs_list[i][2])
    # for j in range(len(targetss_list)):
    #     list_target.append(targetss_list[j])
    # print(list_target)
# print(list_target[0].tolist())

plot(list_0, list_1, list_2)
