from PIL import Image


def resize(image, output_size=(416, 416)):
    '''
        Resize a given image to expected output_size keeping its aspect
    ratio and adding zeros to fill to the desired format.
    '''
    raw_image = Image.fromarray(image)
    raw_image.thumbnail(output_size, Image.ANTIALIAS)
    resized_image = Image.new('RGB', output_size, (0, 0, 0))
    resized_image.paste(raw_image,
                        (int((output_size[0] - raw_image.size[0]) / 2),
                         int((output_size[1] - raw_image.size[1]) / 2)))

    return resized_image
