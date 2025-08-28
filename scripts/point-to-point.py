"""run.py:"""
#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# NB gloo supports only collectives in CUDA, send and receive operations in cuda MUST be made using nccl backend
# https://docs.pytorch.org/docs/stable/distributed.html

def run_gloo(rank, size): # use backend gloo
    
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0], tensor.device)

def run_nccl_cuda(rank, size): # use backend nccl
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tensor = torch.zeros(1, device = device)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0], tensor.device)





def run_non_blocking(rank, size): # use backend gloo
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait() # type: ignore
    print('Rank ', rank, ' has data ', tensor[0])



def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size) # this is the default process group
    # it would be better to set the device here when we initialize the process, with torch.cuda.set_device(rank)
    fn(rank, size)
    # if using nccl we have to destroy process group to avoid resource memory leaking
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    processes = []
    mp.set_start_method("spawn") # https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn
                                 # https://docs.pytorch.org/docs/stable/notes/multiprocessing.html
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run_nccl_cuda))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()




