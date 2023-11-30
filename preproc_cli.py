import os, sys
sys.path.append(os.getcwd())
from tqdm import tqdm
import click
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from typing import List, Optional, Dict

from config.defaults import get_defaults
from landmark_localization.utils.misc import get_all_hsr_file_paths, set_all_seeds
from landmark_localization.utils.image import extract_patch, Geometry

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
def extract_ostia_patches(ctx, *args, **kwargs):
    cfg = bootstrap(kwargs_list=ctx.args,
                    conf_file_path=None)
    
    # file_paths = get_all_hsr_file_paths(cfg.DATA.DATA_PATH)
    # file_paths = list(filter(lambda p: "1000000" not in p.as_posix(), file_paths))
    
    ref = pd.read_csv(cfg.DATA.REFERENCES_PATH)

    for i, data in tqdm(enumerate(ref.itertuples()),
                             total=ref.shape[0],
                             desc="Patch extraction"):
        pid, file_path = data.pid, data.path
        tqdm.write(f"Current patient: {pid}")
        
        rco = [data.rco_x, data.rco_y, data.rco_z] 
        lco = [data.lco_x, data.lco_y, data.lco_z]
        
        image_sitk = sitk.ReadImage(file_path)
        image_arr = sitk.GetArrayFromImage(image=image_sitk)
        origin = image_sitk.GetOrigin()
        spacing = image_sitk.GetSpacing()
        direction = image_sitk.GetDirection()
        geom = Geometry(origin=origin,
                        spacing=spacing,
                        direction=direction,
                        desc="orig")
        
        print(f"RCO in LPS: {rco}")
        print(f"LCO in LPS: {lco}")
        rco_ijk, lco_ijk = geom.convert([rco, lco], mode="LPS -> ijk")
        print(f"RCO in IJK: {rco_ijk}")
        print(f"LCO in IJK: {lco_ijk}")
        
        rco_lps, lco_lps = geom.convert([rco_ijk, lco_ijk], mode="ijk -> LPS")
        print(f"RCO in LPS: {rco_lps}")
        print(f"LCO in LPS: {lco_lps}")

        

cli.add_command(extract_ostia_patches)

if __name__ == '__main__':
    cli()