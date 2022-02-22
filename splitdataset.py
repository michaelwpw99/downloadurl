import splitfolders

input_folder = "IMAGES/32x32"
output = "IMAGES/32x32_SPLIT"

splitfolders.ratio(input_folder, output, seed=42, ratio=(.6,.2,.2))
