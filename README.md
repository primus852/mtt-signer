# Installation (Ubunut 18.04)
## Clone & Install OpenCv
- `git clone https://github.com/primus852/mtt-signer`
- `cd mtt-signer`
- `sudo apt install python3-opencv`

## (optional) Install Virtual Env
- `sudo apt-get install python3-venv`
- `python3 -m venv mtt-signer-env`
- `source mtt-signer-env/bin/activate`

## (optional) Upgrade pip
- `pip3 install --upgrade pip`
- `sudo python3 -m pip install -U setuptools`

## Install dependencies
- `pip install opencv-python`
- `pip install -qr ./yolov5/requirements.txt`

## Train the Model
- `cd yolov5`
- `python3 ./train.py --img 416 --batch 16 --epochs 1000 --data ../config.yaml --cfg ../model.yaml --weights '' --name results  --cache`

## Evaluate the Training
- `python3 ./yolov5/detect.py --weights runs/train/results/weights/best.pt --img 416 --conf 0.4 --source ../data/test/images`

## Results
Results can be found in `/yolov5/runs/train`


