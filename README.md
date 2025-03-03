Pro Tip before you start: 
https://youtu.be/dvUvajxnnq4

Consider using virtual environment to avoid conflicts (but you would have to reinstall even the packages that you already have, again, don't do this in <u>hostel wifi</u>)
```
python3 -m venv amazon
source amazon/bin/activate
```

To exit the virtual environment,
```
deactivate
```

Requirements files are in the CRAFT and deep-text folders
```
pip install -r requirements.txt
```

`trial.ipynb` to download small_sample

For Link Refiner cmd prompt (Windows):
```
python CRAFT-pytorch\pipeline.py --trained_model=craft_mlt_25k.pth --refiner_model=craft_refiner_CTW1500.pth --test_folder=small_sample --refine
```

For Link Refiner cmd prompt (Linux):
```
python3 CRAFT-pytorch/pipeline.py --trained_model=craft_mlt_25k.pth --refiner_model=craft_refiner_CTW1500.pth --test_folder=small_sample --refine
```

Cropping:
```
python3 CRAFT-pytorch/crop_images.py
```

For Deep Text REcognition prompt Command :(Windows)
```
python deep-text-recognition-benchmark/demo.py --Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC --image_folder CropWords --saved_model None-VGG-BiLSTM-CTC.pth