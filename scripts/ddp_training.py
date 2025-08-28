from random import Random
import torch
from torchvision import datasets, transforms
import torch.distributed as dist
from torch import nn 
import torch.nn.functional as F
import os
import torch.multiprocessing as mp
import math

class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(784, 28**2*4)
        self.fc2 = nn.Linear(28**2*4, 28**2*4)
        self.fc3 = nn.Linear(28**2*4, 10)

    def forward(self, x):
        # x: [N, 1, 28, 28] -> [N, 784]
        # N elements in each batch, 1 is the number of channels per pixel, in our case the image is grey scaled, so it is 1
        # 28 and 28 represent the grid --> 28*29 = 784
        # x = x.view(x.size(0), -1)
        # it is better to directly import the dataset with the right format, infact is what i did

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()  # from random import Random
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)
        # take all the indexes and shuffle them, in the for loop below it creates n lists of indexes, each list is a random partition of the dataset
        for frac in sizes:
            part_len = int(frac * data_len) # 30_000
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]


    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


""" Partitioning MNIST """
def partition_dataset():
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.view(-1))  
                                #  transforms.Normalize((0.1307,), (0.3081,))
                                # https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
                             ]))
    size = dist.get_world_size()
    bsz = 128 // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, # type: ignore
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


""" Distributed Synchronous SGD Example """
def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    model.to(f"cuda:{rank}")
    optimizer = torch.optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    num_batches = math.ceil(len(train_set.dataset) / float(bsz)) # type: ignore

    # instead of putting all the dataset in the GPUs all at ones, we put it piece by piece
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            target = F.one_hot(target, 10).to(torch.float16)
            optimizer.zero_grad()
            data = data.to(rank)
            target = target.to(rank)
            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.item() # gradients are not computed yet here
            # [param.grad for param in model.parameters()]
            # [None, None, None.....]
            loss.backward() # computes the gradients, now we find the gradients inside the model.parameters(), each set has its own gradients
            # we can average all the gradients inside each GPU with all_reduce as it is done in the function average_gradients()
            average_gradients(model) # we pass the model object that contains the gradients
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), "models/model.pth")



###########################################################

def init_process(rank, size, fn, backend='nccl'):
    try:
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        dist.init_process_group(backend, rank=rank, world_size=size) # this is the default process group
        
        fn(rank, size)
        # if using nccl we have to destroy process group to avoid resource memory leaking
    except Exception as e:
        print(e)
    finally:
        # in this way we destroy the processes even if there is an error
        dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    processes = []
    mp.set_start_method("spawn") # https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn
                                 # https://docs.pytorch.org/docs/stable/notes/multiprocessing.html
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

