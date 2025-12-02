#!/usr/bin/env python

# pyre-unsafe

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import DataLoader
from PIL import Image

import pandas as pd
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import moco.builder
import moco.loader
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models

# import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms

import webdataset as wds

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--pretrain_dir", help="path to pretrain dataset")
parser.add_argument("--val_dir", help="path to validation datasets")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument("--num-pretrain", type=int, help="number of pretrain samples")
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument("--checkpoint-dir", type=str, help="path to checkpoint dir")
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# moco specific configs:
parser.add_argument(
    "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--moco-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)

# options for moco v2
parser.add_argument("--mlp", action="store_true", help="use mlp head")
parser.add_argument(
    "--aug-plus", action="store_true", help="use moco v2 data augmentation"
)
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")


class ImageDataset(Dataset):
    """Simple dataset for loading images"""

    def __init__(
        self, image_dir, image_list, labels=None, resolution=224, transform=None
    ):
        """
        Args:
            image_dir: Directory containing images
            image_list: List of image filenames
            labels: List of labels (optional, for train/val)
            resolution: Image resolution (96 for competition, 224 for DINO baseline)
        """
        self.image_dir = image_dir
        self.image_list = image_list
        self.labels = labels
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load and resize image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx], img_name

        return image, img_name


def img_collate_fn(batch, *, labels, filenames=False):
    """Custom collate function to handle PIL images"""
    if labels:
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([torch.tensor(item[1]) for item in batch])

        return images, labels

    else:
        images = torch.stack([item[0] for item in batch])
        if filenames:
            filenames = [item[1] for item in batch]
            return images, filenames
        else:
            return images


def main() -> None:
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def make_val_loaders(val_name, args, normalize, num_classes):
    val_root = os.path.join(args.val_dir, val_name, "data")

    train_df = pd.read_csv(os.path.join(val_root, "train_labels.csv"))
    val_df = pd.read_csv(os.path.join(val_root, "val_labels.csv"))
    test_df = pd.read_csv(os.path.join(val_root, "test_images.csv"))
    test_aug = transforms.Compose(
        [
            transforms.Resize(96),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            normalize,
            # transforms.Normalize(mean=cfg_data.ds_mean, std=cfg_data.ds_std),
        ]
    )

    train_aug = transforms.Compose(
        [
            transforms.Resize(96),
            transforms.RandomResizedCrop(
                96,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            normalize,
            # transforms.Normalize(mean=cfg_data.ds_mean, std=cfg_data.ds_std),
        ]
    )

    train_set = ImageDataset(
        train_df["filename"].tolist(),
        train_df["class_id"].tolist(),
        resolution=96,
        transform=train_aug,
    )

    val_set = ImageDataset(
        os.path.join(val_root, "val"),
        val_df["filename"].tolist(),
        val_df["class_id"].tolist(),
        resolution=96,
        transform=test_aug,
    )

    test_set = ImageDataset(
        os.path.join(val_root, "test"),
        test_df["filename"].tolist(),
        labels=None,
        resolution=96,
        transform=test_aug,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=(lambda x: (img_collate_fn(x, labels=True))),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=(lambda x: (img_collate_fn(x, labels=True))),
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=(lambda x: (img_collate_fn(x, labels=False, filenames=True))),
        pin_memory=True,
    )
    return (
        train_loader,
        val_loader,
        test_loader,
        num_classes,
    )


@torch.no_grad()
def extract_features(
    feat_net,
    loader,
    device,
):
    feats, labels = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        # y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            f = feat_net(x)  # N x D

        # cache to CPU:
        # f = F.normalize(f.float(), dim=1).cpu()
        feats.append(f.float().cpu())
        labels.append(y)

    feats = torch.cat(feats, dim=0)  # N x D
    labels = torch.cat(labels, dim=0)  # N
    return feats, labels


def lp_update_and_predict(
    val_feats,
    train_feats,
    train_labels,
    *,
    linear_probe,
    num_classes,
    epochs,
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=512,
    device,
):

    assert (
        torch.is_grad_enabled()
    ), "Linear probe update is running under no_grad/inference_mode!"

    optimizer = torch.optim.AdamW(
        linear_probe.parameters(), lr=lr, weight_decay=weight_decay
    )

    # optimizer = torch.optim.SGD(
    #     linear_probe.parameters(),
    #     1e-3,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay,
    # )
    # optimizer = torch.optim.SGD(linear_probe.parameters(), lr=0.1*(batch_size/256), momentum=0.9, nesterov=True, weight_decay=0.0)

    criterion = nn.CrossEntropyLoss()

    N = train_labels.shape[0]
    # update:
    linear_probe.train()
    for _ in range(epochs):
        order = torch.randperm(N)
        for start_idx in range(0, N, batch_size):
            idxs = order[start_idx : start_idx + batch_size]
            x = train_feats[idxs].to(device)
            y = train_labels[idxs].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = linear_probe(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # predict
    linear_probe.eval()
    preds = []
    with torch.no_grad():
        for start_idx in range(0, N, batch_size):
            x = val_feats[start_idx : start_idx + batch_size].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = linear_probe(x)

            pred = logits.argmax(1)
            preds.append(pred)

    return torch.cat(preds, dim=0)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args) -> None:
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim,
        args.moco_k,
        args.moco_m,
        args.moco_t,
        args.mlp,
    )
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, "train")
    pretrain_dir = args.pretrain_dir
    val_dir = args.val_dir
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(96, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [moco.loader.GaussianBlur([0.1, 2.0])],
                p=0.5,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(96, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     moco.loader.TwoCropsTransform(transforms.Compose(augmentation)),
    # )
    train_dataset = (
        (
            wds.WebDataset(
                f"{pretrain_dir}/shard-*.tar",
                handler=wds.reraise_exception,
                nodesplitter=wds.split_by_node if args.distributed else None,
                shardshuffle=1000,
            )
            .map(lambda s: {"jpg": s["jpg"]})
            .decode("torch")
            .to_tuple("jpg")
            .map(lambda x: x[0])
        )
        .shuffle(5000)
        .map(moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
        .batched(args.batch_size, partial=False)
    )

    train_loader = wds.WebLoader(
        train_dataset,
        batch_size=None,  # Handled in pipeline
        shuffle=False,  # Handled in pipeline
        num_workers=args.workers,
        pin_memory=True,
    ).with_epochs(args.num_pretrain // args.batch_size)

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=(train_sampler is None),
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     sampler=train_sampler,
    #     drop_last=True,
    # )

    val_names = ["cub200", "minet", "sun397"]
    val_loaders = {}
    val_num_classes = {"cub200": 200, "minet": 64, "sun397": 397}
    for val_name in val_names:
        val_loaders[val_name] = make_val_loaders(
            val_name, args, normalize, val_num_classes[val_name]
        )

    linear_probe = nn.Linear(model.encoder_q)

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename="{args.checkpoint_dir}/moco.{:04d}.pth.tar".format(epoch),
            )
            enc = model.encoder_q
            enc.eval()

            for val_name, (
                val_train_loader,
                val_val_loader,
                _,
                val_num_classes,
            ) in val_loaders.items():

                train_feats, train_labels = extract_features(
                    enc, val_train_loader, device=device
                )
                val_feats, val_labels = extract_features(
                    enc, val_val_loader, device=device
                )
                val_labels = val_labels.to(device)
                preds = lp_update_and_predict(
                    val_feats,
                    train_feats,
                    train_labels,
                    linear_probe=linear_probe,
                    num_classes=val_num_classes,
                    epochs=2,
                    batch_size=args.batch_size,
                    device=device,
                )
                lp_acc = (preds == val_labels).float().mean().cpu().item()

            enc.train()


def train(train_loader, model, criterion, optimizer, epoch, args) -> None:
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, images in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename: str = "checkpoint.pth.tar") -> None:
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix: str = "") -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args) -> None:
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
