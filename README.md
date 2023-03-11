# K-SVD

All relevant figures are present in the 'sample_runs' directory. 

## Usage

The script can be run directly or using pip.

### Run Script Directly
Install requirements
```bash
pip install -r requirements.txt
```

Run script
```bash
python ksvd/main.py
```

### Using pip

#### Linux
Install package with requirements
```bash
pip3 -e .
```

Run script
```bash
python3 -m ksvd.main
```

By default, the pretrained model is used. In order to train a new model, add 'use_pretrained=null' to the command line arguments.
After running, the output will be saved under the 'output' directory, in a folder similar to 'sample_run'

