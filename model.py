from skimage import io
import cv2
import numpy as np
import math
from results import Results
import json
from torch import Tensor  # torch only used for nms
from torchvision.ops import nms
from typing import Optional, Union
import os
from numpy.typing import ArrayLike

class Model:
    def __init__(self, model_path:str, do_postprocessing:bool=True, runtime:Optional[str]=None, labels:Optional[Union[str, dict]]=None, conf_threshold:float=0.25, iou_threshold:float=0.75):
        self.model_path = model_path
        self.runtime = model_path.split(".")[-1] if runtime is None else runtime
        self.do_postprocessing = do_postprocessing
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        if isinstance(labels, str):
            with open(labels) as f:
                self.labels = json.load(f)
        else:
            self.labels = labels

        if self.runtime == "onnx":
            import onnxruntime as ort
            self.interpreter = ort.InferenceSession(self.model_path)
        elif self.runtime == "tflite":
            import tensorflow.lite as tflite
            self.interpreter = tflite.Interpreter(self.model_path)
        else:
            raise NotImplementedError(self.runtime)

    def __call__(self, img_path:str) -> list[ArrayLike]:
        self.img_path = img_path
        model_input = self.preprocess(img_path)
        model_outputs = self.run_inference(model_input)
        if self.do_postprocessing:
            return self.postprocess(model_outputs)
        return model_outputs

    def get_input_img_shape(self) -> tuple[int]:
        if self.runtime == "onnx":
            n, c, h, w = self.interpreter.get_inputs()[0].shape
        elif self.runtime == "tflite":
            n, h, w, c = self.interpreter.get_input_details()[0]["shape"]
        return h, w

    def preprocess(self, img_path:str, pad_color:Optional[tuple[int]]=(0, 0, 0)) -> ArrayLike:
        img_arr = io.imread(img_path)
        orig_h, orig_w = img_arr.shape[:-1]
        input_h, input_w = self.get_input_img_shape()
        ratio = min(input_h / orig_h, input_w / orig_w)
        new_h, new_w = int(orig_h * ratio), int(orig_w * ratio)
        img_arr = cv2.resize(img_arr, (new_w, new_h))  # resize
        dw, dh = input_w - new_w, input_h - new_h
        img_arr = cv2.copyMakeBorder(img_arr, dh // 2, -(dh // -2), dw // 2, -(dw // -2), cv2.BORDER_CONSTANT, value=pad_color)  # letterboxing
        img_arr = img_arr[np.newaxis, ...]  # add batch dim
        img_arr = img_arr.astype(np.float32) / 255.0  # normalize if float model
        if self.runtime == "onnx":
            return np.transpose(img_arr, (0, 3, 1, 2))
        return img_arr

    def run_inference(self, model_input:ArrayLike) -> list[ArrayLike]:
        if self.runtime == "onnx":
            assert len(self.interpreter.get_inputs()) == 1, "only 1 input tensor supported"
            return self.interpreter.run(None, {self.interpreter.get_inputs()[0].name: model_input})
        elif self.runtime == "tflite":
            assert len(self.interpreter.get_input_details()) == 1, "only 1 input tensor supported"
            self.interpreter.allocate_tensors()
            self.interpreter.set_tensor(self.interpreter.get_input_details()[0]["index"], model_input)
            self.interpreter.invoke()
            return [self.interpreter.get_tensor(details["index"]) for details in self.interpreter.get_output_details()]
        else:
            raise NotImplementedError(self.runtime)

    def generate_xy_anchor_grid(self, num_preds:int) -> tuple[int, ArrayLike]:
        a = np.zeros((2, num_preds))
        h, w = self.get_input_img_shape()
        stride = int(math.sqrt((h * w) / num_preds))
        maxh, maxw = h // stride, w // stride
        a = np.stack((np.tile(np.arange(maxw), maxh), np.repeat(np.arange(maxh), maxw)), axis=-1) + 0.5
        return stride, a

    def postprocess(self, model_outputs:list[ArrayLike]) -> Results:
        assert len(model_outputs) == 6
        num_classes = model_outputs[3].shape[-1]
        anchors_per_stride = model_outputs[0].shape[-1] // 4
        out = []
        for bbox, cls in zip(model_outputs[:3], model_outputs[3:]):
            assert bbox.shape[0] == 1
            assert bbox.shape[1] == cls.shape[1]
            assert bbox.shape[2] == 4 * anchors_per_stride
            assert cls.shape[2] == num_classes
            bbox = np.exp(bbox.reshape((1, -1, 4, anchors_per_stride)))
            bbox /= np.sum(bbox, axis=3, keepdims=True)  # softmax
            bbox = np.sum(bbox * np.arange(anchors_per_stride), axis=3)  # conv
            slice_a, slice_b = bbox[:, :, 0:2], bbox[:, :, 2:4]
            stride, anchor_grid = self.generate_xy_anchor_grid(bbox.shape[1])
            xy = (anchor_grid + (slice_b - slice_a) / 2) * stride
            wh = (slice_b + slice_a) * stride
            cls = 1.0 / (1.0 + np.exp(-cls))  # sigmoid on class probs
            out.append(np.concatenate((xy, wh, cls), axis=2))
        output = np.concatenate(out, axis=1)
        output = output[:, np.where(np.max(output[0, :, 4:], axis=1) > self.conf_threshold)[0], :]
        output = self.non_max_suppression(output)
        return Results(output, img_path=self.img_path, model_input_shape=self.get_input_img_shape(), labels_dict=self.labels)

    def non_max_suppression(self, data:ArrayLike) -> ArrayLike:
        scores = np.max(data[0, :, 4:], axis=1).reshape(-1)
        boxes = np.hstack((data[0,:,:2] - data[0,:,2:4] / 2, data[0,:,:2] + data[0,:,2:4] / 2))  # xywh to xyxy
        idxs = nms(Tensor(boxes), Tensor(scores), iou_threshold=self.iou_threshold).numpy()
        return data[:, idxs, :]
