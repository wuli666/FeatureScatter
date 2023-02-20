import torchvision.transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from torchvision.datasets import MNIST

# torch.load('')
class TensorDatasetTransform(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
        super(TensorDatasetTransform, self).__init__()

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]
        return (image, label)

    def __len__(self):
        return len(self.images)
# class TensorcifarTransform(Dataset):
#     def __init__(self, images, labels, transform):
#         self.images = images
#         self.labels = labels
#         self.transform = transform
#         super(TensorcifarTransform, self).__init__()
#
#
#     def __getitem__(self, index):
#         image = self.images[index]
#         if self.transform is not None:
#             image = self.transform(image)
#
#         label = self.labels[index]
#         return (image, label)
#
#     def __len__(self):
#         return len(self.images)


def generate_dataset(train_loader, class_ids: tuple, num_classes, transform):
    class_ids = class_ids
    target_list = [[] for _ in range(num_classes)]
    inputs_list = [[] for _ in range(num_classes)]
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        for i in range(num_classes):
            for j in range(0,num_classes):
                target_list[i].append(targets[targets == class_ids[j]].detach())

                inputs_list[i].append(inputs[targets == class_ids[j]].detach())

    for i in range(num_classes):
        target_list[i] = torch.cat(target_list[i], dim=0).detach()[:100]
        inputs_list[i] = torch.cat(inputs_list[i], dim=0).detach()[:100]

    for inputs, targets in zip(inputs_list, target_list):

        return TensorDatasetTransform(inputs, targets, transform)

# images = torch.randn((100, 3, 32, 32))  # [N, C, H, W]
# labels = torch.randint(3, (100,))  # [N]


#
# my_transform = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# dataset = generate_dataset(None, (None,), transform=None,num_classes=3)
# data_loader = DataLoader(dataset, )
# for i in range(100):
#     iamge, label = dataset.__getitem__(i)
#     print(iamge.shape)
#     print(label)
