import torch
import numpy as np


@torch.no_grad() # with torch.no_grad():
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, dataloader, device, i = -1,
             task_id = -1, class_mask = None, target_task_map = None, acc_matrix = None, args = None):
    
    criterion = torch.nn.CrossEntropyLoss()
    


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, dataloader, device, task_id = -1,
                      class_mask = None, target_task_map = None, acc_matrix = None, args = None):
    
    stat_matrix = np.zeros((4, args.num_tasks)) # 4 for Acc@1, Acc@5, Loss, Acc@task

    

