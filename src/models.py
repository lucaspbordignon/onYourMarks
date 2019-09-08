paths = {
    'ssd_mobilenet_v1_coco': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz',
    'ssd_mobilenet_v2_coco': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
    'ssd_inception_v2_coco': 'http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz',
    'faster_rcnn_inception_v2_coco': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz',
    'mask_rcnn_inception_v2_coco': 'http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz'
}


def graph_path(path):
    return '../model/{}/frozen_inference_graph.pb'.format(path)
