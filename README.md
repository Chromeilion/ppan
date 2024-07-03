# PPAN
PPAN is a model for automatic onset detection from video.

## Installing
Make sure Cython is installed and that you're using python <= 3.11. 
Then, from the repo root, simply run: 
```shell
pip install .
```
This will set up ppan in your current Python environment.

## Running
PPAN offers training and validation scripts. These can be run through
the CLI. For more information run PPAN with -h like so:

```shell
python -m ppan -h
```

Because PPAN is built using the Huggingface library, one can easily use 
accelerate to do multi-gpu training. PPAN additionally logs training 
statistics to Weights & Biases, so if you'd like to run training you 
will need an account set up.
To see how the model can be trained using accelerate, have a look at 
```train_script.sh```.

### Configuring
PPAN can be configured by using environment variables or a .env file. To 
see all available environment variables have a look at ```__main__.py```.
All variables can also be passed as command line arguments.

## Important Notes
To speed up data loading PPAN utilizes a cache file. If you change the 
seed/dataset and are running into index errors, deleting the cache files
(one per dataset, e.g train/test) should fix it.
On first run, PPAN may take some time to create the cache files, however 
this is only done once.
