import os
import argparse
from module import counting
from module import denoise, count_connected_comp

# Argument
parser = argparse.ArgumentParser()

parser.add_argument('-i',
                    '--images',
                    help='Path to images folder',
                    required=True)

parser.add_argument('-o',
                    '--output',
                    help='Path to output folder',
                    required=True)

parser.add_argument('-s',
                    '--step',
                    help='Step for visualization',
                    default= 3
                    )

parser.add_argument('-c',
                    '--count',
                    help='Counting opting',
                    required=True
                    )


args = parser.parse_args()

# Extract command line arguments
path_in = args.images
path_out = args.output
step = args.step
count_option = args.count

# Extract image name
name_list = path_in.split("/")
name_png = name_list[-1]

# Denoise
denoise_operation = denoise.Denoise(path_in)
denoise_operation.denoise()

# Counting operation
path_in_counting = 'denoise' + "/" + name_png
if count_option=="contour":
    counting_operation = counting.Counting(path_in, path_in_counting, path_out, step)
    counting_operation.counting_object()
elif count_option == "connected":
    count_method = count_connected_comp.count_connected_comp()
    total_count = count_method.Counting(path_in_counting)
    print(f'Object counting using connected componets = {total_count}')




