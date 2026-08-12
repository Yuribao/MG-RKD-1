conda create -y -n rkd python=3.8.20
eval "$(conda shell.bash hook)"
conda activate rkd

pip install --no-cache-dir -r requirements.txt
