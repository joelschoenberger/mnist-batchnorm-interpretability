from torchvision import datasets, transforms
from torch.utils.data import DataLoader

_MEAN, _STD = 0.1307, 0.3081 #Those are taken from the MNIST example project

def _augmentation():
    return transforms.RandomChoice([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

def get_dataloaders(batch_train=64, batch_test=1000, augment=False):
    tfm = [transforms.ToTensor(),
           transforms.Normalize((_MEAN,), (_STD,))]
    if augment:
        tfm.insert(0, _augmentation())
    transform = transforms.Compose(tfm)

    train_set = datasets.MNIST('data', train=True, download=True,
                               transform=transform)
    test_set  = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_train,
                              shuffle=True, num_workers=1, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_test,
                              shuffle=False, num_workers=1, pin_memory=True)
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_dataloaders()
    print(type(train_loader))
    print(type(test_loader))