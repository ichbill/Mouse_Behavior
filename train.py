# credits:
# how to use DDP module with DDP sampler: https://gist.github.com/sgraaf/5b0caa3a320f28c27c12b5efeb35aa4c
# how to setup a basic DDP example from scratch: https://pytorch.org/tutorials/intermediate/dist_tuto.html
import os
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import random_split
from dataset import MouseDataset
from model import MouseModel
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import math

def get_dataset():
    world_size = dist.get_world_size()
    
    video_path ="/home/nikk/dataset/frame_folder"
    pred_path ='/home/nikk/dataset/alphapose-results.json'
    label_path = '/home/nikk/dataset/Formalin_acute_pain_1.csv'
    audio_path='/home/nikk/dataset/Formalin_Ultrasound_recording.wav'

   
    dataset = MouseDataset(video_path, pred_path, label_path, audio_path, 1500)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    
    train_sampler = DistributedSampler(train_set,num_replicas=world_size)
    val_sampler = DistributedSampler(test_set,num_replicas=world_size)
    batch_size = 16
    print(batch_size)
    #nt(128 / float(world_size))
    print(world_size, batch_size)
    train_loader = DataLoader(
        dataset=train_set,
        sampler=train_sampler,
        batch_size=batch_size
    )
    val_loader = DataLoader(
        dataset = test_set,
        sampler=val_sampler,
        batch_size=batch_size
    )

    print("Returning your tran loader val loader and batch size")

    return train_loader, val_loader, batch_size

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
def reduce_dict(input_dict, average=True):
    world_size = float(dist.get_world_size())
    names, values = [], []
    for k in sorted(input_dict.keys()):
        names.append(k)
        values.append(input_dict[k])
    values = torch.stack(values, dim=0)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values /= world_size
    reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
def train(model,train_loader,optimizer,batch_size):
    device = torch.device(f"cuda:{dist.get_rank()}")
    train_num_batches = int(math.ceil(len(train_loader.dataset) / float(batch_size)))
    model.train()
    # let all processes sync up before starting with a new epoch of training
    print(train_num_batches)
    # dist.barrier()
    criterion = nn.CrossEntropyLoss().to(device)
    for steps, (images, behavior_feat, audio, labels) in enumerate(tqdm(train_loader)):
        images, behavior_feat, audio, labels = images.to(device), behavior_feat.to(device), audio.to(device), labels.to(device)
        
        output = model(audio)
        loss = criterion(output, labels.float())

        optimizer.zero_grad()
        loss.backward()
        # average gradient as DDP doesn't do it correctly
        average_gradients(model)
        optimizer.step()
        
        loss_ = {'loss': torch.tensor(loss.item()).to(device)}
        print(loss)
        #train_loss += reduce_dict(loss_)['loss'].item()
        # cleanup
        # dist.barrier()
        # data, target, output = data.cpu(), target.cpu(), output.cpu()
    #train_loss_val = train_loss / train_num_batches
    #print(train_loss_val)
    return None

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
            res.append(correct_k.div_(batch_size))
        return res

def val(model, val_loader,batch_size):
    device = torch.device(f"cuda:{dist.get_rank()}")
    #val_num_bacthes = 32
    #print(val_loader.dataset)
    val_num_batches = int(math.ceil(len(val_loader.dataset) / float(batch_size)))

    #val_num_batches = int((math.ceil(len(val_loader.dataset)) / float(batch_size)))
    print(val_num_batches)
    model.eval()
    # let all processes sync up before starting with a new epoch of training
    # dist.barrier()
    criterion = nn.CrossEntropyLoss().to(device)
    val_loss = 0.0
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for images, behavior_feat, audio, labels in val_loader:
            images, behavior_feat, audio, labels = images.to(device), behavior_feat.to(device), audio.to(device), labels.to(device)
            outputs = model(audio)

            # outputs = model()

            probs = torch.sigmoid(outputs)
            print(probs)
            preds = (probs > 0.5).float()
            print(preds)
            total += labels.size(0)
            correct += (preds == labels).all(dim=1).sum().item()
            print(correct)
            loss = criterion(outputs, labels.float())
            
            loss_ = {'loss': torch.tensor(loss.item()).to(device)}
            val_loss += reduce_dict(loss_)['loss'].item()
        print(correct)
    accuracy = 100 * (correct/total)
    val_loss_val = val_loss / val_num_batches
    return val_loss_val, accuracy

def run(rank, world_size):
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(1234)
    train_loader, val_loader, batch_size = get_dataset()
    model = MouseModel(50, 3).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # use if model contains batchnorm.
    model = DDP(model,device_ids=[rank],output_device=rank)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    #optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.5)
    history =  {
            "rank": rank,
            "train_loss_val": [],
            "train_acc_val": [],
            "val_loss_val": [],
            "val_acc_val": []
        }
    if rank == 0:
        history = {
            "rank": rank,
            "train_loss_val": [],
            "train_acc_val": [],
            "val_loss_val": [],
            "val_acc_val": []
        }
    for epoch in range(50):
        train_loss_val = train(model,train_loader,optimizer,batch_size)
        val_loss_val, val_accuraccy = val(model,val_loader,batch_size)
        print(val_loss_val)
        print(val_accuraccy)
        print("Done")
        print()
        print(f'Rank {rank} epoch {epoch}: {val_accuraccy:.2f}')
        #rint(f'Rank {rank} epoch {epoch}: {train_loss_val:.2f}')
        print(f'Rank {rank} epoch {epoch}: {val_loss_val:2f}')
        if rank == 0:
            history['train_loss_val'].append(train_loss_val)
            history['val_loss_val'].append(val_loss_val)
            history['val_acc_val'].append(val_accuraccy)
    print(f'Rank {rank} finished training')
    print(history)
    cleanup(rank)  

def cleanup(rank):
    # dist.cleanup()  
    dist.destroy_process_group()
    print(f"Rank {rank} is done.")
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
def init_process(
        rank, # rank of the process
        world_size, # number of workers
        fn, # function to be run
        # backend='gloo',# good for single node
        # backend='nccl' # the best for CUDA
        backend='nccl'
    ):
    # information used for rank 0
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    #print(torch.cuda.memory_allocated())
    dist.barrier()
    setup_for_distributed(rank == 0)
    fn(rank, world_size)


if __name__ == "__main__":
    world_size = 8
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
