import os

from configs.Config_unet import get_config
from datasets.example_dataset.create_splits import create_splits
from datasets.example_dataset.download_dataset import download_dataset
from datasets.example_dataset.preprocessing import preprocess_data
from experiments.UNetExperiment import UNetExperiment

c = get_config()

dataset_name = 'Task04_Hippocampus'
download_dataset(dest_path=c.data_root_dir, dataset=dataset_name)

preprocess_data(root_dir=os.path.join(c.data_root_dir, dataset_name))
create_splits(output_dir=c.split_dir, image_dir=c.data_dir)

exp = UNetExperiment(config=c, name='unet_experiment', n_epochs=c.n_epochs,
                     seed=42, append_rnd_to_name=c.append_rnd_string, visdomlogger_kwargs={"auto_start": c.start_visdom})

exp.run()
# exp.run_test(setup=False)