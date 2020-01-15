
from configs.Config_unet_spleen import get_config
import subprocess

if __name__ == "__main__":
    c = get_config()
    n_epochs = c.n_epochs
    learning_rate = c.learning_rate
    step = 0

    while True:
        result = subprocess.run(['python', 'run_train_pipeline.py', '{}'.format(n_epochs), '{}'.format(learning_rate)])
        # var = result.stdout

        if divmod(step, 2)[1] == 0:
            n_epochs = n_epochs + 20
        else:
            learning_rate = learning_rate / 2
        step += 1
