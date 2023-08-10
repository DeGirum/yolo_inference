from skimage.io import imread
import cv2
import numpy as np
import math
from results import Results
import json
from torch import Tensor
from torchvision.ops import nms
from typing import Optional, Union
from numpy.typing import ArrayLike
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

class Model:
    def __init__(self, 
                 model_path:str, 
                 do_postprocessing:bool=True, 
                 do_bbox_decoding:bool=True, 
                 runtime:Optional[str]=None, 
                 labels:Optional[Union[str, dict]]=None, 
                 conf_threshold:float=0.25, 
                 iou_threshold:float=0.75, 
                 postprocessor_inputs:Optional[list[int]]=None
                 ):
        
        self.model_path = model_path
        self.runtime = model_path.split('.')[-1] if runtime is None else runtime
        self.do_postprocessing = do_postprocessing
        self.do_bbox_decoding = do_bbox_decoding
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.infer_postprocessor_inputs = postprocessor_inputs is None
        self.postprocessor_inputs = postprocessor_inputs
        if isinstance(labels, str):
            with open(labels) as f:
                self.labels = json.load(f)
        else:
            self.labels = labels
        if self.runtime == 'onnx':
            import onnxruntime as ort
            self.interpreter = ort.InferenceSession(self.model_path)
        elif self.runtime == 'tflite':
            import tensorflow.lite as tflite
            delegate = None
            try:
                delegate = tflite.experimental.load_delegate(EDGETPU_SHARED_LIB)
                self.interpreter = tflite.Interpreter(self.model_path, experimental_delegates=[delegate])
            except:
                print('EdgeTPU delegate not found')
                self.interpreter = tflite.Interpreter(self.model_path)                
        else:
            raise NotImplementedError(self.runtime)
        
    def __call__(self, img_path:str) -> list[ArrayLike]:
        self.img_path = img_path
        model_input = self.preprocess(img_path)
        model_outputs = self.run_inference(model_input)
        if self.do_postprocessing:
            self.postprocessor_inputs = self.infer_postprocessor_input_order(model_outputs) if self.infer_postprocessor_inputs else self.postprocessor_inputs
            quant_params = None if self.get_output_dtype() == np.float32 else [self.get_output_quant_params()[i] for i in self.postprocessor_inputs]
            return self.postprocess([model_outputs[i] for i in self.postprocessor_inputs], quant_params=quant_params)
        return model_outputs
    
    def infer_postprocessor_input_order(self, model_outputs: list[ArrayLike]) -> list[int]:
        if not self.do_bbox_decoding:
            return [0]
        num_classes = -1
        for o in model_outputs:
            if o.shape[2] != 64:
                num_classes = o.shape[2]
                break
        assert num_classes != -1, 'cannot infer postprocessor inputs via output shape if there are 64 classes'
        return [i for i,_ in sorted(enumerate(model_outputs), key = lambda x: (x[1].shape[2] if num_classes > 64 else -x[1].shape[2], -x[1].shape[1]))]

    def get_input_img_shape(self) -> tuple[int]:
        if self.runtime == 'onnx':
            n, c, h, w = self.interpreter.get_inputs()[0].shape
        elif self.runtime == 'tflite':
            n, h, w, c = self.interpreter.get_input_details()[0]['shape']
        return h, w

    def get_input_dtype(self):
        if self.runtime == 'onnx':
            return np.float32  # only support float inputs/outputs for onnx
        elif self.runtime == 'tflite':
            return self.interpreter.get_input_details()[0]['dtype']
        
    def get_output_dtype(self):
        if self.runtime == 'onnx':
            return np.float32  # only support float inputs/outputs for onnx
        elif self.runtime == 'tflite':
            return self.interpreter.get_output_details()[0]['dtype']
        
    def get_input_quant_params(self) -> tuple[float]:
        assert self.runtime == 'tflite'
        assert self.get_input_dtype() == np.uint8 or self.get_input_dtype() == np.int8
        return self.interpreter.get_input_details()[0]['quantization']
    
    def get_output_quant_params(self):
        assert self.runtime == 'tflite'
        assert self.get_input_dtype() == np.uint8 or self.get_input_dtype() == np.int8
        return [details['quantization'] for details in self.interpreter.get_output_details()]

    def preprocess(self, img_path:str, pad_color:Optional[tuple[int]]=(0, 0, 0)) -> ArrayLike:
        img_arr = imread(img_path)
        if len(img_arr.shape) == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        orig_h, orig_w = img_arr.shape[:-1]
        input_h, input_w = self.get_input_img_shape()
        ratio = min(input_h / orig_h, input_w / orig_w)
        new_h, new_w = int(orig_h * ratio), int(orig_w * ratio)
        img_arr = cv2.resize(img_arr, (new_w, new_h))  # resize
        dw, dh = input_w - new_w, input_h - new_h
        img_arr = cv2.copyMakeBorder(img_arr, dh // 2, -(dh // -2), dw // 2, -(dw // -2), cv2.BORDER_CONSTANT, value=pad_color)  # letterboxing
        img_arr = img_arr[np.newaxis, ...]  # add batch dim
        img_arr = img_arr.astype(np.float32) / 255.0  # normalize
        if self.get_input_dtype() == np.uint8 or self.get_input_dtype() == np.int8:
            scale, zero = self.get_input_quant_params()
            img_arr = img_arr / scale + zero
            img_arr = img_arr.astype(self.get_input_dtype())
        if self.runtime == 'onnx':
            return np.transpose(img_arr, (0, 3, 1, 2))
        return img_arr

    def run_inference(self, model_input:ArrayLike) -> list[ArrayLike]:
        if self.runtime == 'onnx':
            assert len(self.interpreter.get_inputs()) == 1, 'only 1 input tensor supported'
            return self.interpreter.run(None, {self.interpreter.get_inputs()[0].name: model_input})
        elif self.runtime == 'tflite':
            assert len(self.interpreter.get_input_details()) == 1, 'only 1 input tensor supported'
            self.interpreter.allocate_tensors()
            self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], model_input)
            self.interpreter.invoke()
            return [self.interpreter.get_tensor(details['index']) for details in self.interpreter.get_output_details()]
        else:
            raise NotImplementedError(self.runtime)

    def generate_xy_anchor_grid(self, num_preds:int) -> tuple[int, ArrayLike]:
        a = np.zeros((2, num_preds))
        h, w = self.get_input_img_shape()
        stride = int(math.sqrt((h * w) / num_preds))
        maxh, maxw = h // stride, w // stride
        a = np.stack((np.tile(np.arange(maxw), maxh), np.repeat(np.arange(maxh), maxw)), axis=-1) + 0.5
        return stride, a

    def postprocess(self, model_outputs:list[ArrayLike], quant_params:Optional[list[tuple[float]]]=None) -> Results:
        if quant_params is not None:  # dequantize
            for i in range(len(model_outputs)):
                scale, zero = quant_params[i]
                model_outputs[i] = (model_outputs[i].astype(np.float32) - zero) * scale
        if self.do_bbox_decoding:
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
        else:
            assert len(model_outputs) == 1
            input_h, input_w = self.get_input_img_shape()
            output = np.transpose(model_outputs[0], (0, 2, 1))
            output[:,:,:4] *= (input_w, input_h, input_w, input_h)  # ultralytics export has box coords normalized
        output = output[:, np.where(np.max(output[0, :, 4:], axis=1) > self.conf_threshold)[0], :]
        output = self.non_max_suppression(output)
        return Results(output, img_path=self.img_path, model_input_shape=self.get_input_img_shape(), labels_dict=self.labels)

    def non_max_suppression(self, data:ArrayLike) -> ArrayLike:
        scores = np.max(data[0, :, 4:], axis=1).reshape(-1)
        boxes = np.hstack((data[0,:,:2] - data[0,:,2:4] / 2, data[0,:,:2] + data[0,:,2:4] / 2))  # xywh to xyxy
        idxs = nms(Tensor(boxes), Tensor(scores), iou_threshold=self.iou_threshold).numpy()
        return data[:, idxs, :]
