import torch
import numpy as np

from vits import utils
from vits.utils import MetricLogger

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import os
'''
@torch.no_grad() # with torch.no_grad():
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, device, i = -1,
             task_id = -1, class_mask = None, target_task_map = None, acc_matrix = None, args = None):
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter = " ")
    header = 'Test: [Task {}]'.format(i + 1)

    # Switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking = True)
            target = target.to(device, non_blocking = True)

            # Compute output
            with torch.no_grad():
                if original_model is not None:
                    pretrain_output = original_model(input)
                    pretrain_logits = pretrain_output['logits']

                    if args.train_mask and class_mask is not None:
                        mask = []
                        for id in range(task_id + 1):
                            mask.extend(class_mask[id])
                        not_mask = np.setdiff1d(np.arange(args.nb_classed), mask)
                        not_mask = torch.tensor(not_mask, dtype = torch.int64).to(device)
                        pretrain_logits = pretrain_logits.index_fill(dim = 1, index = not_mask, value = float('-inf'))
                    
                    prompt_id = torch.max(pretrain_logits, dim = 1)[1]

                    # Translate cls to task_id
                    prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device = device).unsqueeze(-1)
                else:
                    raise NotImplementedError("Original model is None")
            
            output = model(input, task_id = task_id, prompt_id = prompt_id)
            logits = output['logits']
            prompt_idx = output['prompt_idx'] # tensor Batch * topK

            if args.task_inc and class_mask is not None:
                # Adding mask to output logits
                mask = class_mask[i]
                mask = torch.tensor(mask, dtype = torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device = device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask
            
            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target)
            task_inference_acc = utils.task_inference_accuracy(prompt_idx, target, target_task_map)

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n = input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n = input.shape[0])
            metric_logger.meters['Acc@task'].update(task_inference_acc.item(), n = input.shape[0])
    

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@task {task_acc.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} '
        .format(task_acc = metric_logger.meters['Acc@task'],
                top1 = metric_logger.meters['Acc@1'],
                top5 = metric_logger.meters['Acc@5'],
                losses = metric_logger.meters['loss']                
                )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, device, task_id = -1,
                      class_mask = None, target_task_map = None, acc_matrix = None, args = None):
    
    stat_matrix = np.zeros((4, args.num_tasks)) # 4 for Acc@1, Acc@5, Loss, Acc@task

    for i in range(task_id + 1):
        test_stats = evaluate(model = model, original_model = original_model, data_loader = data_loader[i]['val'], device = device,
                              i = i, task_id = task_id, class_mask = class_mask, target_task_map = target_task_map, args = args)
        
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        stat_matrix[3, i] = test_stats['Acc@task']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis = 1), task_id + 1)

    diagonal = np.diag(acc_matrix)


    result_str = "[Average acuuracy till task{}]\tAcc@task: {:.4f}\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1,
        avg_stat[3],
        avg_stat[0],
        avg_stat[1],
        avg_stat[2]
    )

    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis = 1) - acc_matrix[:, task_id])[:task_id])

        backward = np.mead((acc_matrix[:, task_id] - diagonal)[:task_id])

        # Compute CAA (Continual Average Accuracy)
        mean_acc = [np.sum(acc_matrix[:, i]) / (i + 1) for i in range(task_id + 1)]
        caa = np.mean(mean_acc)
        
        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}\tCAA: {:.4f}".format(forgetting, backward, caa)
    
    print(result_str)

    return test_stats

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module,
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable, optimizer: torch.optim.Optimizer, 
                       lr_scheduler, device: torch.device, class_mask = None, target_mask_map = None, args = None):
    
    global cls_mean
    global cls_cov
    global old_head

    cls_mean = dict()
    cls_cov = dict()

    # Create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    print('-' * 20)
    print('Learnable parameters:')

    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
    
    print('-' * 20)
    print(f"Prompt shape: {model.e_prompt.prompt.shape}")

    if args.prompt_key:
        print(f"Prompt key shape: {model.e_prompt_prompt_key.shape}")

    for task_id in range(args.num_tasks):
        if task_id > 0:
            model.e_prompt.act_scale.requires_grad(False)
            old_head = model.get_head()
        
        print('-' * 20)
        print('Learnable parameters')

        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)
        
        print('-' * 20)

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            if args.larger_prompt_lr:
                # This is a simple yet effective trick that helps to learn task_specific prompt better
                base_params = []
                base_fc_params = []
                for name, p in model_without_ddp.named_parameters():
                    if 'prompt' in name and p.requires_grad == True:
                        base_params.append(p)
                    if 'prompt' not in name and p.requires_grad == True:
                        base_fc_params.append(p)
                
                base_params = {
                    'params': base_params,
                    'lr': args.lr,
                    'weight_decay': args.weight_decay
                }

                base_fc_params = {
                    'params': base_fc_params,
                    'lr': args.lr * 0.1,
                    'weight_decay': args.weight_decay
                }

                network_params = [base_params, base_fc_params]
                optimizer = create_optimizer(args, network_params)
            else:
                optimizer = create_optimizer(args, model)
            
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            else:
                lr_scheduler = None
            
            # Load original model checkpoint
            if args.trained_original_model:
                original_checkpoint_path = os.path.join(args.trained_original_model, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
                if os.path.exists(original_checkpoint_path):
                    print('Loading checkpoint from:', original_checkpoint_path)
                    original_checkpoint = torch.load(original_checkpoint_path, map_location = device)
                    original_model.load_state_dict(original_checkpoint['model'])
                else:
                    print('No checkpoint found at', original_checkpoint_path)
                    return
            
            # if model already trained
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))

            if os.path.exists(checkpoint_path) and (not args.reset):
                print("Model already trained for task {}".format(task_id + 1))
                print('Loading checkpoint from:', checkpoint_path)

                # Load model checkpoint
                checkpoint = torch.load(checkpoint_path, map_location = device)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if args.sched is not None and args.sched != 'constant':
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
                print('-' * 20)
                print(f'Evaluate task {} after CA'.format(task_id + 1))
'''
                
print("Evaluate task {} after CA".format(4 + 1))