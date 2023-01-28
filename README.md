# Implementation of KGDC

It is the implementation we used for our experiments.


It is not the final version. We will clean up our code and make it more easier to use.


### Usage

1. Download **Wizard of Wikipedia Dataset** and **ConceptNet 5.5**
2. Run **Process.py** to process the raw data.
3. Run **Datasets.py**, that will create a cache file (Remember the directory!) 
4. If you want to change settings, check `settings/train_config.json`、`settings/eval_config.json` and `settings/model_config.json`。
5. Run **Model.py**.
6. More details can be found in our source code. (But it may be hard to read. Sorry for the bad coding style :( )
