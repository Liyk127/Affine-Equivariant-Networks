import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def prepare_data_loader(args):

    if args.dataset == "affNIST":
        from load_affNIST import train_data, test_data_aff
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_data_aff, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
    
    elif args.dataset in ["RS-MNIST", "RS-Fashion"]:
        data_root = "./data/{}/seed_{}".format(args.dataset, args.data_seed)
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.0607,), (0.2161,))
            ])
        train_dataset = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_root, 'test'), transform=transform)
        train_loader = DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
        test_loader = DataLoader(test_dataset,
                        batch_size=args.batch_size_test, shuffle=False, pin_memory=True, num_workers=2)
    
    elif args.dataset in ["Scale-MNIST", "Scale-Fashion"]:
        from load_scale_mnist import load_dataset, Dataset
        data_root = "./data/{}".format(args.dataset)
        listdict = load_dataset(data_root, 6, 50000)
        train_data = listdict[args.data_seed]['train_data']
        train_labels = listdict[args.data_seed]['train_label']
        test_data = listdict[args.data_seed]['test_data']
        test_labels = listdict[args.data_seed]['test_label']
        Data_train = Dataset(args.dataset, train_data, train_labels, transforms.ToTensor())
        Data_test = Dataset(args.dataset, test_data, test_labels, transforms.ToTensor())
        train_loader = DataLoader(Data_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(Data_test, batch_size=args.batch_size_test, shuffle=False, num_workers=2)

    return train_loader, test_loader