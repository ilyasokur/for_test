from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Каждый эпок проходит тренировочную и валидационную фазу
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # тренируем
            else:
                model.eval()  # делаем эваль модели

            current_loss = 0.0
            current_corrects = 0

            # здесь начинается обучение
            print('Iterating through data...')
            i = 0
            for inputs, labels in dataloaders[phase]:
                i += 1
                inputs = inputs.to(device)
                labels = labels.to(device)
                print('{} load...'.format(i))
                # Нам нужны нулевые гладиенты
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # если фаза тренинга оптимизируем
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # сохраняем статистику о loss
                current_loss += loss.item() * inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)

            epoch_loss = current_loss / dataset_sizes[phase]
            epoch_acc = current_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # если на валидационном сете улучшилась точность сохраняем
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    # показываем время
    time_since = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # загружаем лучшие веса модели
    model.load_state_dict(best_model_wts)
    return model

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([mean_nums])
    std = np.array([std_nums])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_handeled = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_handeled += 1
                ax = plt.subplot(num_images // 2, 2, images_handeled)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_handeled == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == '__main__':

    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    chosen_transforms = {'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_nums, std_nums)
    ]), 'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean_nums, std_nums)
    ]),
    }

    # директория с данными
    data_dir = 'NNimgs/'

    # создаем датасет из папки с файлами (train, val)
    chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
      chosen_transforms[x])
                      for x in ['train', 'val']}

    # делаем батч 4 делаем шафл чтобы раскидать рандомно данные соответственно идет создание дата лоудора
    dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size=4,
      shuffle=True, num_workers=4)
                  for x in ['train', 'val']}

    dataset_sizes = {x: len(chosen_datasets[x]) for x in ['train', 'val']}
    class_names = chosen_datasets['train'].classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # визуализируем картинки

    # берем данные чтобы визуализировать
    inputs, classes = next(iter(dataloaders['train']))

    # делаем грид
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    # берем заранее натренированную модель resnet

    res_mod = models.resnet34(pretrained=True)

    num_ftrs = res_mod.fc.in_features
    res_mod.fc = nn.Linear(num_ftrs, 2)

    for name, child in res_mod.named_children():
        print(name)

    res_mod = res_mod.to(device)
    criterion = nn.CrossEntropyLoss()

    # оптимизируем с помощью SGD
    optimizer_ft = optim.SGD(res_mod.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    base_model = train_model(res_mod, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=3)
    visualize_model(base_model)
    plt.show()

