paths = {
    'mask_rcnn_inception_v2_coco': 'http://download.tensorflow.org/models' +
                                   '/object_detection/mask_rcnn_inception' +
                                   '_v2_coco_2018_01_28.tar.gz'
}

tensors = {
    'mask_rcnn_inception_v2_coco': {
        'input': 'image_tensor:0',
        'output': [
            'detection_boxes:0',
            'num_detections:0',
            'detection_scores:0',
            'detection_classes:0',
            'detection_masks:0',
        ],
    },
}


def graph_path(path):
    return '../model/{}/frozen_inference_graph.pb'.format(path)
