from configs.Config_unet import get_config
from experiments.UNetExperiment import UNetExperiment

c = get_config()

exp = UNetExperiment(config=c, name='unet_experiment', n_epochs=c.n_epochs,
                     seed=42, append_rnd_to_name=c.append_rnd_string, visdomlogger_kwargs={"auto_start": c.start_visdom}, globs=globals())

exp.run()
exp.run_test(setup=False)
