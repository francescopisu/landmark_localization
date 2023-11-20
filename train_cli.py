import os, sys
sys.path.append(os.getcwd())

import click
from pathlib import Path
from typing import List, Optional, Dict
from config.defaults import get_defaults

BASE = Path.cwd()

def bootstrap(kwargs_list: Optional[List] = None,
              conf_file_path: str = None,
              ) -> Dict:
    defaults = get_defaults()

    if conf_file_path:
        if not isinstance(conf_file_path, Path):
            conf_file_path = Path(conf_file_path)

        # first, merge from file
        if conf_file_path.is_file():
            defaults.merge_from_file(conf_file_path)

    if kwargs_list:
        defaults.merge_from_list(kwargs_list)

    defaults.freeze()

    set_all_seeds(defaults.MISC.SEED)

    return defaults


@click.group()
def cli():
    pass


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.pass_context
@click.option("--placeholder", default=True)
def placeholder(ctx, placeholder, *args, **kwargs):
    pass

cli.add_command(placeholder)


if __name__ == '__main__':
    cli()