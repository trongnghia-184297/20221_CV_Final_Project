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


## Project2

**Install requirements**
We assume that you have *python* package in your system:
```
cd project2
pip install -r requirements.txt
```

**Usage**
```
python main.py --template_img_path=path_to_template, --detect_img_path=path_to_detect --algorithm=algorithm_name --matcher=matcher_name --MIN_MATCH_COUNT1=threshold1 --MIN_MATCH_COUNT2=threshold2 --save_path=output_folder_path
```

**Explanation**
- template_img_path = Path to the template image
- detect_img_path = Path to the detect image
- algorithm = Algorithm name, current implementation has sift|asift algorithm
- matcher = Matcher name, current implementation has bruteforce|kdtree matcher
- MIN_MATCH_COUNT1 = Threshold for good matches
- MIN_MATCH_COUNT2 = Threshold for unique matched points in the detect image
- save_path = Path to the output folder

**Example**
```
python main.py --template_img_path=data/templates/IMG_0613.jpeg --detect_img_path=data/images/1674896240445.jpg --algorithm=sift --matcher=bruteforce --MIN_MATCH_COUNT1=60 --MIN_MATCH_COUNT2=30 --save_path=./
```