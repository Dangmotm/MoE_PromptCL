import argparse

def get_args_parser(subparsers):
    subparsers.add_argument('--batch-size', default = 16, type = int, help = 'Batch size per device')
    subparsers.add_argument('--epochs', default = 5, type = int, help = 'Number of epoch')

    # Model parameters
    subparsers.add_argument('--original_model', default = 'vit_base_patch16_224', type = str, metavar = 'OMODEL', help = 'Name of original model to train')
    subparsers.add_argument('--model', default = 'vit_base_patch16_224', type = str, metavar = 'MODEL', help = 'Name of model to train')
    subparsers.add_argument('--input-size', default = 224, type = int, help = 'images input size')
    subparsers.add_argument('--pretrained', default = True, help = 'Load pretrained model or not')
    subparsers.add_argument('--drop', type = float, default = 0.0, metavar = 'Percentage', help = 'Dropout rate')
    subparsers.add_argument('--drop-path', type = float, default = 0.0, metavar = 'Percentage', help = 'Drop path rate')

    # Optimizer parameters
    subparsers.add_argument('--opt', default = 'adam', type = str, metavar='OPTIMIZER', help = 'Optimizer')
    subparsers.add_argument('--opt-eps', default = 1e-8, type = float, metavar = 'EPSILON', help = 'Optimizer Epsilon')
    subparsers.add_argument('--opt-betas', default = (0.9, 0.999), type = float, nargs = '+', metavar = 'BETA', help = 'Optimizer Betas, use opt default)')
    subparsers.add_argument('--clip-grad', type = float, default = 1.0, metavar = 'NORM', help = 'Clip gradient norm')
    subparsers.add_argument('--momentum', type = float, default = 0.9, metavar = 'M', help = 'SGD momentum')
    subparsers.add_argument('--weight-decay', type = float, default = 0.0, help = 'Ưeight decay')
    subparsers.add_argument('--reinit_optimizer', type = bool, default = True, help = 'Reinit optimizer')

    # Learning rate schedule parameters
    subparsers.add_argument('--sched', default = 'step', type = str, metavar = 'SCHEDULER', help = 'LR scheduler')
    subparsers.add_argument('--lr', type = float, default = 0.03, metavar = 'LR', help = 'learning rate')
    subparsers.add_argument('--lr-noise', type = float, nargs = '+', default = None, metavar = 'Percentage, Percentage', help = 'learning rate noise on/off epoch percentages')
    subparsers.add_argument('--lr-noise-pct', type = float, default = 0.67, metavar = 'PERCENT', help = 'learning rate noise limit percent')
    subparsers.add_argument('--lr-noise-std', type=float, default = 1.0, metavar = 'STDDEV', help = 'learning rate noise std-dev')
    subparsers.add_argument('--warmup-lr', type = float, default = 1e-6, metavar = 'LR', help = 'warmup learning rate')
    subparsers.add_argument('--min-lr', type = float, default = 1e-5, metavar = 'LR', help = 'lower lr bound for cyclic schedulers that hit 0')
    subparsers.add_argument('--decay-epochs', type = float, default = 30, metavar = 'N', help = 'epoch interval to decay LR')
    subparsers.add_argument('--warmup-epochs', type = int, default = 0, metavar = 'N', help = 'epochs to warmup LR, if scheduler supports')
    subparsers.add_argument('--cooldown-epochs', type = int, default = 10, metavar = 'N', help = 'epochs to cooldown LR at min_lr, after cyclic schedule ends')
    subparsers.add_argument('--patience-epochs', type = int, default = 10, metavar = 'N', help = 'patience epochs for Plateau LR scheduler')
    subparsers.add_argument('--decay-rate', '--dr', type = float, default = 0.1, metavar = 'RATE', help = 'LR decay rate')
    subparsers.add_argument('--unscale_lr', type = bool, default = True, help = 'scaling lr by batch size')
