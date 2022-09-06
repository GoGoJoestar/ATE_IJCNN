"""Arguments for configuration."""

from __future__ import absolute_import
import six
import argparse



def str2bool(v):
    """str to bool"""
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")

class ArgumentGroup(object):
    def __init__(self, parser, title, desc):
        self._group = parser.add_argument_group(title=title, description=desc)

    def add_arg(self, name, type, default, help, positional_arg=False, **kargs):
        prefix = "" if positional_arg else "--"
        type = str2bool if type == bool else type
        self._group.add_argument(
            prefix + name,
            default=default,
            type=type,
            help=help + ' Default: %d(default)s.',
            **kargs
        )