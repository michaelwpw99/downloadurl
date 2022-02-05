import splitfolders

input_folder = "IMAGES"
output = "NEW_IMAGES"

splitfolders.ratio(input_folder, output, seed=42, ratio=(.6,.2,.2))
