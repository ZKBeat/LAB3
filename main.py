import os
from dataclasses import dataclass
from typing import Tuple, Iterable, Any, Dict, List

import cv2  # type: ignore
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

input_hw_size = (608, 608)

tf.compat.v1.enable_eager_execution()

COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush']

INPUT_IMAGE = './index.jpg'
STORAGE_FROZEN_GRAPHS_DIR = './'
TensorName = str


@dataclass
class RunnerDescriptor:
    graph_path: str
    input_tensor_names: Tuple[TensorName, ...]
    output_tensor_names: Tuple[TensorName, ...]


@dataclass
class MetricDescriptor:
    name: str
    accuracy: float
    runner_descriptor: RunnerDescriptor


def wrap_frozen_graph(graph_def: tf_v1.GraphDef, inputs: Iterable[str], outputs: Iterable[str]) -> Any:
    def imports_graph_def() -> Any:
        tf_v1.import_graph_def(graph_def, name="")

    wrapped_import = tf_v1.wrap_function(imports_graph_def, [])
    import_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

class Yolo2BasePostProc:
    def __init__(self, anchors: np.ndarray,
                 num_classes: int,
                 image_shape: Tuple[int, int],
                 score_threshold: float,
                 iou_threshold: float) -> None:
        self.anchors = anchors
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    @staticmethod
    def _yolo_boxes_to_corners(box_xy: tf.Tensor, box_wh: tf.Tensor) -> tf.Tensor:
        """Convert boxes to corners."""
        half_one = tf.convert_to_tensor(0.5, dtype=tf.float32)
        box_xy1 = box_xy - half_one * box_wh
        box_xy2 = box_xy + half_one * box_wh
        boxes = tf.concat((box_xy1, box_xy2), axis=-1)
        return boxes

    @staticmethod
    def _yolo_filter_boxes(
            box_confidence: tf.Tensor,
            boxes: tf.Tensor,
            box_class_probs: tf.Tensor,
            confidence_threshold: float = .5
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Filter boxes by confidence threshold."""
        box_scores = box_confidence * box_class_probs
        box_classes = tf.argmax(box_scores, axis=-1)  # best score index
        box_class_scores = tf.math.reduce_max(box_scores, axis=-1)  # best score
        prediction_mask = box_class_scores >= confidence_threshold
        boxes = tf.boolean_mask(boxes, prediction_mask)
        scores = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)
        return scores, boxes, classes

    @staticmethod
    def _scale_boxes(boxes: tf.Tensor, image_shape: Tuple[int, int]) -> tf.Tensor:

        height = image_shape[0]
        width = image_shape[1]
        image_dims = tf.stack([width, height, width, height])
        image_dims = tf.reshape(image_dims, [1, 4])
        boxes = boxes * tf.cast(image_dims, dtype='float32')
        return boxes

    @staticmethod
    def _non_max_suppression(
            scores: tf.Tensor,
            boxes: tf.Tensor,
            classes: tf.Tensor,
            max_boxes: int = 20,
            iou_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Non max suppression for boxes and scores."""
        nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
        boxes = tf.gather(boxes, nms_indices)
        scores = tf.gather(scores, nms_indices)
        classes = tf.gather(classes, nms_indices)

        scores = tf.reshape(scores, (-1, 1)).numpy()
        classes = tf.reshape(classes, (-1, 1)).numpy()
        boxes = boxes.numpy()

        return scores, boxes, classes

    @staticmethod
    def __yolo2_head(
            feats: np.ndarray,
            anchors: Any,
            fmap_size: int,
            num_classes: int
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        feats = feats.astype(np.float32)
        num_anchors = anchors.shape[0]
        result = tf.reshape(feats, (1, fmap_size, fmap_size, num_anchors, num_classes + 5))

        coord_x = tf.cast(tf.reshape(tf.tile(tf.range(fmap_size), [fmap_size]), (1, fmap_size, fmap_size, 1, 1)),
                          tf.float32)
        coord_y = tf.transpose(coord_x, (0, 2, 1, 3, 4))
        coords = tf.tile(tf.concat([coord_x, coord_y], -1), [1, 1, 1, 5, 1])
        dims = tf.cast(tf.shape(result)[1:3], dtype=tf.float32)
        dims = tf.reshape(dims, (1, 1, 1, 1, 2))

        # pred_xy and pred_wh shape (m, fmap_size, fmap_size, Anchors, 2)
        pred_xy = tf.sigmoid(result[:, :, :, :, 0:2])
        pred_xy = (pred_xy + coords)
        pred_xy = pred_xy / dims
        pred_wh = tf.exp(result[:, :, :, :, 2:4])
        pred_wh = (pred_wh * anchors)
        pred_wh = pred_wh / dims
        # pred_confidence
        box_conf = tf.sigmoid(result[:, :, :, :, 4:5])
        # pred_class
        box_class_prob = tf.math.softmax(result[:, :, :, :, 5:])
        # Reshape
        box_xy = pred_xy[0, ...]
        box_wh = pred_wh[0, ...]
        box_confidence = box_conf[0, ...]
        box_class_probs = box_class_prob[0, ...]

        return box_confidence, box_xy, box_wh, box_class_probs

    def yolo_eval(
            self,
            feats: np.ndarray,
            fmap_size: int,
            max_boxes: int = 10,
            confidence_threshold: float = .5,
            iou_threshold: float = .5
    ) -> np.ndarray:
        yolo_outputs = self.__yolo2_head(feats=feats, anchors=self.anchors, fmap_size=fmap_size,
                                         num_classes=self.num_classes)

        box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
        boxes = self._yolo_boxes_to_corners(box_xy, box_wh)
        scores, boxes, classes = self._yolo_filter_boxes(
            box_confidence,
            boxes,
            box_class_probs,
            confidence_threshold=confidence_threshold)
        boxes = self._scale_boxes(boxes, self.image_shape)

        scores, boxes, classes = self._non_max_suppression(
            scores=scores,
            boxes=boxes,
            classes=classes,
            max_boxes=max_boxes,
            iou_threshold=iou_threshold)

        return np.hstack((classes, boxes, scores)).astype('float32')


class TFRunner:
    def __init__(self, descriptor: RunnerDescriptor):
        self.descriptor = descriptor
        self.runner_descriptor = descriptor
        with tf.io.gfile.GFile(descriptor.graph_path, "rb") as file:
            graph_def = tf.compat.v1.GraphDef()
            _ = graph_def.ParseFromString(file.read())

        self.frozen_graph_wrapper = wrap_frozen_graph(graph_def=graph_def,
                                                      inputs=descriptor.input_tensor_names,
                                                      outputs=descriptor.output_tensor_names)

    def __call__(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self._run_inference(input_data)

    def _run_inference(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        inputs_dict = {}
        outputs_dict = {}
        for key, value in input_data.items():
            node_dict = {
                key: tf.constant(value, dtype=value.dtype)
            }
            inputs_dict.update(node_dict)
        keys = list(input_data.keys())
        tensor_list = [inputs_dict[key] for key in keys]
        outputs = self.frozen_graph_wrapper(*tensor_list)
        for index, _ in enumerate(outputs):
            node_dict = {self.runner_descriptor.output_tensor_names[index]: outputs[index].numpy()}
            outputs_dict.update(node_dict)
        return outputs_dict


class Yolo2:
    def __init__(self, descriptor: MetricDescriptor):
        self._descriptor = descriptor
        # super().__init__(descriptor, dataset_path, label_file_path)

    @property
    def runner_descriptor(self) -> RunnerDescriptor:
        return self._descriptor.runner_descriptor

    @staticmethod
    def preprocess_yolo_common(data: Iterable[Image.Image], output_size: Tuple[int, int]) -> np.ndarray:
        # channels = len(data[0].mode)
        channels = 3  # All Coco networks requires 3 channels
        result = np.ndarray((0, *output_size, channels), dtype=np.float32)
        for img in data:
            img = img.convert('RGB')
            tensor = np.asarray(img).astype(np.float32)
            tensor = cv2.resize(tensor, output_size)
            tensor = np.expand_dims(tensor, axis=0)
            result = np.concatenate((result, tensor), 0)
        return result

    @staticmethod
    def _div255(tensor: np.ndarray) -> np.ndarray:
        return tensor / 255.0

    @classmethod
    def pre(cls, images: Iterable[Image.Image]) -> np.ndarray:
        output_data = Yolo2.preprocess_yolo_common(images, input_hw_size)
        output_data = Yolo2._div255(output_data)

        return output_data

    @staticmethod
    def post(images_sizes: List[Tuple[int, int]], inference_output: Dict[str, np.ndarray]) -> List[np.ndarray]:

        idx = 0
        predictions = []
        for images_size in images_sizes:

            original_image_size = images_size[::-1]

            out_name = list(inference_output.keys())[0]
            prediction = inference_output[out_name]
            prediction = prediction[idx:idx + 1]

            yolo = Yolo2BasePostProc(
                anchors=np.array([
                    (0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
                    (7.88282, 3.52778), (9.77052, 9.16828),
                ], dtype=np.float32),
                num_classes=len(COCO_CLASS_NAMES),
                image_shape=original_image_size,
                score_threshold=0.5,
                iou_threshold=0.5,
            )

            prediction = yolo.yolo_eval(
                feats=prediction,
                fmap_size=(input_hw_size[0] // 32),
                confidence_threshold=0.5,
                iou_threshold=0.5,
            )
            predictions.append(prediction)
            idx += 1

        return predictions

def get_img_sizes(images: Iterable[Image.Image]) -> List[Tuple[int, int]]:
    sizes = []
    for image in images:
        sizes.append(image.size)
    return sizes

def get_spaced_colors(number: int) -> List[Tuple[int, int, int]]:
    """Get spaced colors for drawing."""
    max_value = 255 ** 3
    interval = int(max_value / number)
    colors = [hex(ind)[2:].zfill(6) for ind in range(0, max_value, interval)]
    return [(int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)) for color in colors]

def draw_boxes(
    img: Image,
    boxes: Any,
    class_names: Any,
    font: str = r'C:\Users\1.000\AppData\Local\Microsoft\Windows\Fonts\ofont.ru_Bebas Neue.ttf',
        ) -> Image:
    if class_names is None:
        class_names = COCO_CLASS_NAMES
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font=font,
                              size=(img.size[0] + img.size[1]) // 100)
    colors = get_spaced_colors(len(class_names))
    if isinstance(boxes, dict):
        for cls in list(boxes.keys()):
            box_ = boxes[cls]
            if np.shape(box_)[0] != 0:
                box = box_[0]
                color = colors[cls]
                xy_coords, confidence = box[:4], box[4]
                xy_coords = np.asarray([xy_coords[0], xy_coords[1], xy_coords[2], xy_coords[3]])
                x0_coord, y0_coord = xy_coords[0], xy_coords[1]
                thickness = (img.size[0] + img.size[1]) // 200
                for tick in np.linspace(0, 1, thickness):
                    xy_coords[0], xy_coords[1] = xy_coords[0] + tick, xy_coords[1] + tick
                    xy_coords[2], xy_coords[3] = xy_coords[2] - tick, xy_coords[3] - tick
                    draw.rectangle(xy_coords, outline=tuple(color))
                text = '{} {:.1f}%'.format(class_names[cls],
                                           confidence * 100)
                text_size = draw.textsize(text, font=font)
                draw.rectangle(
                    [x0_coord, y0_coord - text_size[1], x0_coord + text_size[0], y0_coord],
                    fill=tuple(color))
                draw.text((x0_coord, y0_coord - text_size[1]), text, fill='black',
                          font=font)
    elif isinstance(boxes, np.ndarray):
        confidence = 0
        for cls in range(boxes.shape[0]):
            box = boxes[cls]
            color = colors[int(box[0])]
            class_ = int(box[0])
            if box.shape[0] == 6:
                xy_coords, confidence = box[1:5], box[5]
            else:
                xy_coords = box[1:5]
            xy_coords = np.asarray([xy_coords[0], xy_coords[1], xy_coords[2], xy_coords[3]])
            x0_coord, y0_coord = xy_coords[0], xy_coords[1]
            thickness = (img.size[0] + img.size[1]) // 200
            for tick in np.linspace(0, 1, thickness):
                xy_coords[0], xy_coords[1] = xy_coords[0] + tick, xy_coords[1] + tick
                xy_coords[2], xy_coords[3] = xy_coords[2] - tick, xy_coords[3] - tick
                draw.rectangle([xy_coords[0], xy_coords[1], xy_coords[2], xy_coords[3]], outline=tuple(color))
            if box.shape[0] == 6:
                text = '{} {:.1f}%'.format(class_names[class_], confidence * 100)
            else:
                text = '{}'.format(class_names[class_])
            text_size = draw.textsize(text, font=font)
            draw.rectangle(
                [x0_coord, y0_coord - text_size[1], x0_coord + text_size[0], y0_coord],
                fill=tuple(color))
            draw.text((x0_coord, y0_coord - text_size[1]), text, fill='white',
                      font=font)
    else:
        raise TypeError('unsupported type of boxes %s' % type(boxes))
    img = img.convert('RGB')
    return img

def generate_input_node_dict(node_name: str, data: np.ndarray) -> Dict[str, np.ndarray]:
    return {node_name: data}


def print_hi(name):
    runner_descriptor = RunnerDescriptor(
        graph_path=os.path.join(STORAGE_FROZEN_GRAPHS_DIR, 'yolo2_original_frozen.pb'),
        input_tensor_names=('input_1:0',),
        output_tensor_names=('conv_23/BiasAdd:0',),
    )

    metric_descriptor = MetricDescriptor(
        accuracy=0.36,
        name='yolo2',
        runner_descriptor=runner_descriptor,
    )
    network = Yolo2(metric_descriptor)

    runner = TFRunner(runner_descriptor)
    input_img = (Image.open(INPUT_IMAGE),)

    sizes = get_img_sizes(input_img)

    input_data = network.pre(input_img)

    input_dict = generate_input_node_dict(network.runner_descriptor.input_tensor_names[0], input_data)

    output = runner(input_dict)
    predictions = network.post(sizes, output)

    result = draw_boxes(input_img[0], predictions[0], None)
    result.save(f"res_{1}.jpg")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
