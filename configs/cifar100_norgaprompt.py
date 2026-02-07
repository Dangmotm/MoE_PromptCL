import argparse

def get_args_parser(subparsers):
    subparsers.add_argument('--batch-size', default = 16, type = int, help = 'Batch size per device')
