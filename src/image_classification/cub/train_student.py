"""
# ImageNet Pytorch Boilerplate code from https://github.com/pytorch/examples/tree/main/imagenet
# Use this code to train CUB 200-2011 teacher models. All teacher models trained using standard 90-epoch receipe.

The additional changes included are:
    1) Mixed Precision Training
    2) Weights and Biases logging for experiment monitoring
"""

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision

# Import own modules here
from cub_dataset import Cub2011
from utils import KL_div_Loss, get_cub_trained_teacher_for_cub_student

# Import wandb
import wandb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CUB 200-2011 Knowledge Distillation Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--gpus', default=-1, type=int,
                    help='total gpus to use')
parser.add_argument('--output_dir', default='/results/', type=str,
                    help='Directory to store weights')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


# KD experiment arguments
parser.add_argument('--temperature', type=float, required=True,
                    help='Temperature for Distillation')
parser.add_argument('--teacher_weights_path', type=str, required=True,
                    help='Tar file path of teacher weights')
parser.add_argument('--teacher_alpha', type=float, required=True,
                    help='Should the teacher be pretrained?')


# Use automatic mixed precision
parser.add_argument('--use_amp', type=int, required=True,
                    help='Use automatic mixed precision?')


# wandb project name for experiment tracking
parser.add_argument('--exp_name', type=str, required=True,
                    help='Experiment name for wandb logging')
 
best_acc1 = 0
best_acc5 = 0

def main():
    args = parser.parse_args()
    args.use_amp = bool(args.use_amp)

    # Set environement variables for storing wandb outputs
    os.environ['WANDB_DIR'] = args.output_dir
    os.environ['WANDB_API_KEY'] = 'e7badad8ddfe0ffa2638056776cc3374cfedc648' # Should be kept private

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.gpus  == -1:
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = args.gpus
    
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


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # Create teacher model
    teacher, teacher_model_name = get_cub_trained_teacher_for_cub_student(args)
    
    # create model
    if args.pretrained:
        print("=> Using pretrained '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True) # Since CUB200-2011 is a very small dataset, we use transfer learning similar to Shen et.al.
        
        if args.arch in ['resnet18', 'resnet50']:
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 200)
        
        elif args.arch in ['mobilenet_v2']:
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(num_ftrs, 200)
        
        elif args.arch in ['squeezenet1_1']:
            # change the last Conv2D layer in case of squeezenet. there is no fc layer in the end.
            num_ftrs = 512
            #model.classifier._modules["1"] = nn.Conv2d(512, 200, kernel_size=(1, 1))
            #model.classifier[1].out_channels = 200

            in_chans = model.classifier[1].in_channels
            k = model.classifier[1].kernel_size
            s = model.classifier[1].stride
            model.classifier[1] = torch.nn.Conv2d(in_chans, 200, kernel_size=k, stride=s)
            #model.classifier[0] = torch.nn.Dropout(p=0.1, inplace=False)

            # because in forward pass, there is a view function call which depends on the final output class size.
            model.num_classes = 200
            print(model)
        
        elif args.arch in ['convnext_tiny']:
            num_ftrs = model.classifier[2].in_features
            model.classifier[2] = torch.nn.Linear(num_ftrs, 200)
            print(model)

        else:
            raise Exception    
    else:
        raise NotImplementedError # Transfer learning for student must be applied

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            teacher.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        else:
            model.cuda()
            teacher.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            teacher = torch.nn.parallel.DistributedDataParallel(teacher)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        teacher = teacher.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            teacher = torch.nn.DataParallel(teacher).cuda()

    # Define KL divergence loss
    criterion_train = KL_div_Loss(temperature=args.temperature).cuda(args.gpu)
    
    criterion = nn.CrossEntropyLoss().cuda(args.gpu) # Test criterion
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    # Create GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    print("automatic mixed precision set to", args.use_amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = torch.Tensor([best_acc1]).to(args.gpu)
            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.use_amp:
        cudnn.enabled = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # train_dataset = Cub2011(args.data, train=True, transform=transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )

    # Add more augmentations for better generalization
    train_dataset = Cub2011(args.data, train=True, transform=transforms.Compose([
            transforms.RandomResizedCrop( 224, scale=(0.20, 1.0), ratio=(0.80, 1.25) ),
            transforms.RandomHorizontalFlip(),

            transforms.RandomRotation(45, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.RandomGrayscale(0.2),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0, hue=0),

            transforms.ToTensor(),
            normalize,
        ])
    )

    val_dataset = Cub2011(args.data, train=False, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # Create wandb object only for Rank0 GPU
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
        # wandb start run
        wandb.init(project=args.exp_name, entity='keshik',
        tags=[ 'student', str(args.teacher_alpha), str(args.temperature) ], group='cub200-2011')
        wandb.config.update(args)
        wandb.watch(model, log="gradients", log_freq=24) # approximately log every epoch

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        tr_loss, tr_top1, tr_top5, current_lr = train(train_loader, model, teacher, criterion_train, optimizer, scaler, epoch, args)

        # evaluate on validation set
        val_loss, val_top1, val_top5 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = val_top1.item() > best_acc1
        best_acc1 = max(val_top1.item(), best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            
            if is_best:
                best_acc5 = val_top5

            # Log best_acc metrics
            wandb.log( {"val/best_acc1": best_acc1, "val/best_acc5": best_acc5} )

            # Log lr
            wandb.log( {'train/lr': current_lr} )
            
            # Log wandb metrics
            wandb.log( {"train/epoch_loss": tr_loss, "train/top1_acc": tr_top1, "train/top5_acc": tr_top5} )
            wandb.log( {"val/epoch_loss": val_loss, "val/top1_acc": val_top1, "val/top5_acc": val_top5} )

            # Log best acc1, acc5 metrics
            wandb.log( {"val/best_acc1": best_acc1, "val/best_acc5": best_acc5} )

            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'best_acc1': best_acc1,
                'arch': args.arch,
                'teacher-arch': teacher_model_name,
                'temperature': args.temperature, 
                'pretrained': args.pretrained,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler' : scaler.state_dict(),
            }, is_best, filename=os.path.join(args.output_dir, 'cub-student={}-teacher={}({})-T={}-checkpoint-seed={}.pth.tar'.format(args.arch, \
                teacher_model_name, args.teacher_alpha, args.temperature, args.seed)) )

    
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
        wandb.finish()


def train(train_loader, model, teacher, criterion, optimizer, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        current_lr = adjust_learning_rate(optimizer, epoch, i, len(train_loader), args )

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)


        with torch.cuda.amp.autocast(enabled=args.use_amp):
            output = model(images)

            # Get teacher scores
            with torch.no_grad():
                teacher.eval()
                teacher_scores = teacher(images)
            
            loss = criterion(output, teacher_scores)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        
        # loss.backward()
        # optimizer.step()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg, top5.avg, current_lr


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                output = model(images)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        #shutil.copyfile(filename, 'model_best.pth.tar')
        shutil.copyfile(filename, filename.replace('checkpoint', 'best'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, step, len_epoch, args):
    """Use Shen et.al scheme. Only warmup a bit for better convergence"""
    lr = args.lr * (0.1 ** (epoch // 80))
    #lr = args.lr * (0.5 ** (epoch // 50))

    """Warmup"""
    # if epoch < 5:
    #     lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
