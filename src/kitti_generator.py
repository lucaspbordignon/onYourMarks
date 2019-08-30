import cv2 as cv


def kitti_generator(path='../data/kitti/image_2/'):
    index = 0
    has_image = True
    image = cv.imread(path + image_filename(index))

    while(has_image):
         yield (has_image, image)

         index += 1
         image = cv.imread(path + image_filename(index))
         has_image = image is not None


def image_filename(index):
    return '00000{}_10.png'.format(index)
