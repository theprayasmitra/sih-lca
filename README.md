1. Clone the repository

```
git clone https://github.com/your-username/sih-lca.git
cd sih-lca
```

2. Using Git Bash create a Python virtual environment
```
python3 -m venv venv
```
This creates the virtual environment required for the model.

Then to enter the venv:

```
source venv/Scripts/activate
```
For linux:
```
source venv/bin/activate
```

3. Install pip dependencies
```
pip install -r requirements.txt
```

Then run the training script for the model:
```
cd src
python train.py
```
To use the interactive predictor:
```
python predict.py
```

