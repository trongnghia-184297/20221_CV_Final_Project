import os


save_path = 'temp/'
template_images = ['data/templates/'+i for i in os.listdir('data/templates')]
detect_images = ['data/images/' + i for i in os.listdir('data/images')]
algos = ['sift', 'asift']
matchers = ['kdtree', 'bruteforce']

count = 1
for template_image in template_images:
    for detect_image in detect_images:
        for algo in algos:
            for matcher in matchers:
                os.system("python main.py --template_img_path={} --detect_img_path={} --algorithm={} --matcher={} --save_path={} --logs=1".format(template_image, detect_image, algo, matcher, save_path))
                print('DONE', count, template_image, detect_image, algo, matcher)
                count += 1