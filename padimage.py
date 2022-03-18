from PIL import Image
import os
directory = 'IMAGES/malware'




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
          new_image = add_margin(im, 13, 11, 13, 12, (0,255,0))
          newfilepath = 'IMAGES/malware/train' + filename
          new_image.save(newfilepath, quality=100, subsampling=0)
