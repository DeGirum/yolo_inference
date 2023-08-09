# yolo_inference

A wrapper for convenient inference with difference export formats for YoloV8.

## Setup
Set up your environment with all the necessary dependencies by running:
```bash
$ pip install -r requirements.txt
```

## Usage
```python
from model import Model
from matplotlib.pyplot import imshow

model = Model("< path to onnx/tflite model >")
res = model("< image url/path >")
imshow(res.image_overlay)
```

Single image detection example at: [example.ipynb](example.ipynb) \
Full dataset mAP evaluation example at: [map_eval.ipynb](map_eval.ipynb)


## Models

Some of our exported models are available for use [here](https://github.com/DeGirum/yolo_inference/releases/tag/v1.0). For more variations, sign up for the DeGirum cloud platform to get access to our full ultralytics model zoo at [cs.degirum.com](https://cs.degirum.com)!


## Arguments
```python
Model( 
    model_path:str, 
        # path to model, must be .onnx or .tflite file

    do_postprocessing:bool=True, 
        # do non-max-suppression and export results as Results objects
        # if False, export model output tensors directly

    do_bbox_decoding:bool=True, 
        # decode bounding box coordinates from model exported in DeGirum format (6 outputs)
        # MUST BE FALSE TO USE WITH MODELS EXPORTED VIA ULTRALYTICS REPO (1 output)

    labels:Optional[Union[str, dict[str,str]]]=None, 
        # either a path to a json file with a labels dictionary, or the dictionary object itself

    conf_threshold:float=0.25, 
        # confidence threshold for filtering results

    iou_threshold:float=0.75, 
        # IOU threshold for non-max suppression

    device:str='CPU',
        # device to perform inference on. currently only support 'CPU' or 'EDGETPU'

    postprocessor_inputs:Optional[list[int]]=None
        # list of indices to rearrange model outputs before passing to postprocessor
        # if None, infer order of postprocessor inputs via model output shape
        # otherwise, manually override the order, in the following format:
        '''
          0. bounding box data [1, a, 64]
          1. bounding box data [1, b, 64]
          2. bounding box data [1, c, 64]
          3. probability data [1, a, num_classes]
          4. probability data [1, b, num_classes]
          5. probability data [1, c, num_classes]
        '''
        # NOTE: postprocessor input inference does not work with models that have 64 classes, 
        # as it relies on the differences in output shapes to check the order. for these models, 
        # you must input the indices manually

    )
```
