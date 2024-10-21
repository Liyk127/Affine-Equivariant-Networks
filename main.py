import os
import sys
import datetime
import argparse
import torch
from torch.optim import lr_scheduler
from utils import Logger, compute_param, train, test
from prepare_data_loader import prepare_data_loader
from InvarPDEs_Net_affine import InvarPDEs_Net_affine
from InvarPDEs_Net_RS import InvarPDEs_Net_RS
from InvarPDEs_Net_scale import InvarPDEs_Net_scale
from InvarLayer_affine import InvarLayer_affine_resnet32
from InvarLayer_RS import InvarLayer_RS_cnn
from InvarLayer_scale import InvarLayer_scale_cnn


parser = argparse.ArgumentParser(description='net')

parser.add_argument('--model', help='model name')
parser.add_argument('--dataset', help='dataset')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')
parser.add_argument('--batch_size', default=32, type=int, help='input batch size for training')
parser.add_argument('--batch_size_test', default=300, type=int, help='input batch size for testing ')
parser.add_argument('--epochs', default=90, type=int, help='number of epochs to train')

parser.add_argument('--channel', default=0, type=int, help='number of channel')
parser.add_argument('--hidden', default=0, type=int, help='number of hidden channel')
parser.add_argument('--iter', default=0, type=int, help='number of iterations')

parser.add_argument('--seed', default=0, type=int, help="random seed")
parser.add_argument('--data_seed', default=0, type=int, help="random seed for dataset")
parser.add_argument('--test_interval', default=10, type=int, help='test interval')
parser.add_argument('--log_interval', default=2, type=int, help='log interval(%)')
parser.add_argument('--log', default=1, type=int, help='log or not')

args = parser.parse_args()


os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



if args.model == "InvarPDEs_Net_scale":
    network = InvarPDEs_Net_scale(channel=args.channel, hidden_channel=args.hidden, iter=args.iter)
elif args.model == "InvarLayer_scale":
    network = InvarLayer_scale_cnn()
elif args.model == "InvarPDEs_Net_RS":
    network = InvarPDEs_Net_RS(channel=args.channel, hidden_channel=args.hidden, iter=args.iter)
elif args.model == "InvarLayer_RS":
    network = InvarLayer_RS_cnn()
elif args.model == "InvarPDEs_Net_affine":
    network = InvarPDEs_Net_affine(channel=args.channel, hidden_channel=args.hidden, iter=args.iter)
elif args.model == "InvarLayer_affine":
    network = InvarLayer_affine_resnet32()

network.cuda()
optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
train_loader, test_loader = prepare_data_loader(args)
log_interval = int(round(len(train_loader) / 100 * args.log_interval))


date_time = datetime.datetime.now().strftime('%m%d-%H%M%S')
path_name = "./results/checkpoints/" + date_time + "/"
file_name = "./results/{} {} {} {} {} {}".format(args.dataset, date_time, args.model,
                                            args.learning_rate, args.weight_decay, args.epochs)
if args.log:
    os.makedirs(path_name)
    sys.stdout = Logger(file_name)
print(args)
print("Number of parameters: ", compute_param(network))


for epoch in range(1, args.epochs + 1):
    train(epoch, network, train_loader, optimizer, scheduler, args.log, log_interval, path_name)
    scheduler.step()
    if epoch % args.test_interval == 0 or epoch == args.epochs:
        test(test_loader, network)