## Reproduce
```bash
python3 -m venv mujoco-env
python3 -m pip install -r requirements.txt
python3 train_dt.py && eval_dt.py && plot_dt.py
```