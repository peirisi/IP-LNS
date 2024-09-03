## The files are not fully organized yet; the uploaded files are key components of the original files.
## Once the original files are organized, they will be uploaded.
### Installation and dependencies
```
pip install -r requirements.txt
```
We use Gurobi 11.0 as MIP solver.After activating the Gurobi license, the environment setup is complete.
### Quick start guide
Step1: generate initial state for dataset generation
```
python gen_instance_dataset/gen_instance.py --instance 80_8_std
```
setp2: generate high-quality,middle and low-quality commitments
```
python gen_instance_dataset/gen_contrastive_json.py
```
step3: generate initial prediction and neighborhood prediction datasets
```

```
