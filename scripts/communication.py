"""run.py:"""
#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

""" All-Reduce example."""
def run(rank, size):
    """ Simple collective communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1,device=f'cuda:{rank}') # collective operations are fine on CUDA with backend gloo
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


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




# >>> # xdoctest: +SKIP("no rank")
# >>> # All tensors below are of torch.int64 type.
# >>> # We have 2 process groups, 2 ranks.
# >>> device = torch.device(f"cuda:{rank}")
# >>> tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
# >>> tensor
# tensor([1, 2], device='cuda:0') # Rank 0
# tensor([3, 4], device='cuda:1') # Rank 1
# >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
# >>> tensor
# tensor([4, 6], device='cuda:0') # Rank 0
# tensor([4, 6], device='cuda:1') # Rank 1
# >>> # All tensors below are of torch.cfloat type.
# >>> # We have 2 process groups, 2 ranks.
# >>> tensor = torch.tensor(
# ...     [1 + 1j, 2 + 2j], dtype=torch.cfloat, device=device
# ... ) + 2 * rank * (1 + 1j)
# >>> tensor
# tensor([1.+1.j, 2.+2.j], device='cuda:0') # Rank 0
# tensor([3.+3.j, 4.+4.j], device='cuda:1') # Rank 1
# >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
# >>> tensor
# tensor([4.+4.j, 6.+6.j], device='cuda:0') # Rank 0
# tensor([4.+4.j, 6.+6.j], device='cuda:1') # Rank 1