# 20221_CV_Final_Project
## Project1

**Install requirements**
We assume that you have *python* package in your system:
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
- -o = name of output folder, output folder will be created if not exist.
- -s = number of images per each visualization in process, s must be 1,2,3 or 4

**Example**
```
python main.py -i imgs/noise_pepper.png -o output -s 3
```