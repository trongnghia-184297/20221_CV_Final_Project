# 20221_CV_Final_Project
*Project1*

**Install requirements**
```
cd project1
pip install -r requirements.txt
```

**Usage**
```
python main.py -i path/to/image -o path/to/output -s num_step
```

**Explanation**
- -i = path to input image
- -o = path to output folder, folder will be created if not exist.
- -s = number of images per each visualization in process, s must be 1,2,3 or 4

**Example**
ython main.py -i imgs/noise_pepper.png -o output -s 3