from PIL import Image
import os
directory = 'benign500'




def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


if __name__ == '__main__':
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
          im = Image.open(directory + "/" + filename)
          new_image = add_margin(im, 107, 111, 107, 111, (255, 255, 255))
          newfilepath = 'resizedbenign/' + filename
          new_image.save(newfilepath, quality=100)
