# Configuration
The scripts make heavy usage of configuration files, through the `hydra`
library.
You can install the library using `pip`, as in:
```pip install hydra_core```

The configuration files are stored in the `config` folder and can be divided
into configurations for the dataset (`dataset`, `channels` and `normalisation`)
and for the training procedure (`module` and `training`).

In particular, `channels` and `normalisation` are referred within the `dataset`
config file and this latter is passed to the `ParsedDataModule` class to
instantiate the datamodule.
Internal references are made using the `field1: ${field2}` syntax, which leads
to `field1` to refer to `field2`.
