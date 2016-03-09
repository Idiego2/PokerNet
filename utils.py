#!/usr/bin/env python
"""PokerNet utility functions"""

from __future__ import print_function
from argparse import ArgumentParser as Parser
import os
import shutil


def get_parser():
    """Parse command-line arguments"""
    parser = Parser(description='PokerNet utility functions')
    parser.add_argument('-c', '--clean', help='remove simulation files',
                        action='store_true')
    return parser


def cleanup():
    """Remove results"""
    if os.path.exists('results'):
        shutil.rmtree('results')
        print('Removed results/ directory.')
    else:
        print('No results directory found.')


def command_line_runner():
    """Handle command-line interaction"""
    parser = get_parser()
    args = vars(parser.parse_args())
    if args['clean']:
        cleanup()


if __name__ == '__main__':
    command_line_runner()
