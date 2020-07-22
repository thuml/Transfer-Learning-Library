import os
import shutil


def create_exp_dir(path, scripts_to_save=None):
    os.makedirs(path, exist_ok=True)

    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        script_path = os.path.join(path, 'scripts')
        if os.path.exists(script_path):
            shutil.rmtree(script_path)
        os.mkdir(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            print(dst_file)
            shutil.copytree(script, dst_file)