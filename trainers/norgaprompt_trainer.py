import torch
from vits.datasets import build_continual_dataloader
from timm.models import create_model

def train(args):
    device = torch.device(args.device)
    data_loader, data_loader_per_cls, class_mask, target_task_map = build_continual_dataloader(args)

    print(f"Creating original model: {args.original_model}")

    original_model = create_model(
        args.original_model,
        pretrained = args.pretrained,
        num_classes = args.nb_classes,
        drop_rate = args.drop,
        drop_path_rate = args.drop_path,
        drop_block_rate = None,
        mlp_structure = args.original_model_mlp_structure
    )

    print(f"Creating model : {args.model}")