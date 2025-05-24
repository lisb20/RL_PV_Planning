## Codes of the paper City-level Grid-aware Rooftop Photovoltaic Planning via Multi-agent Reinforcement Learning


### Dataset
- **./data/case300.mat**: Casefile from Matpower (https://github.com/MATPOWER/matpower/blob/master/data/case300.m).

- **./config.yaml**: Power flow configs.
- **./data/power_2class.mat**: Power demand templates.
- **./data/solarbej.npy**: PV generation template.

### Train the model
Run train.py, and checkpoints will be recorded in the **./ckpt** folder. Read the buffer file will get the final results.