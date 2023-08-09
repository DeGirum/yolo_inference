import numpy as np
from skimage import io
import cv2
from typing import Optional, Any
from numpy.typing import ArrayLike

class Results:
    def __init__(self, data:ArrayLike, model_input_shape:tuple[int], labels_dict:Optional[dict[str,str]]=None, img_path:Optional[str]=None):
        self.data = data
        self.labels_dict = labels_dict
        self.img_path = img_path
        self.orig_img = io.imread(img_path)
        self.img_h = self.orig_img.shape[0]
        self.img_w = self.orig_img.shape[1]
        self.model_input_shape = model_input_shape
        h, w = self.model_input_shape
        self.ratio = min(h / self.img_h, w / self.img_w)
        self.pad_top = int((h - self.img_h * self.ratio) // 2)
        self.pad_left = int((w - self.img_w * self.ratio) // 2)        
        self.results = self.get_results_dict()
        self.image_overlay = self.get_image_overlay()

    def __repr__(self):
        out = ''
        for r in self.results:
            out += f'- bbox: {r["bbox"].tolist()}\n  category_id: {r["category_id"]}\n  label: {r["label"]}\n  score: {r["score"]}\n'
        return out        
    
    def get_resized_box_coords(self, boxes:ArrayLike) -> ArrayLike:
        new_boxes = np.zeros(boxes.shape)
        new_boxes[:,0] = np.clip((boxes[:,0] - self.pad_left) / self.ratio, a_min=0, a_max=self.img_w)
        new_boxes[:,1] = np.clip((boxes[:,1] - self.pad_top) / self.ratio, a_min=0, a_max=self.img_h)
        new_boxes[:,2] = np.clip(boxes[:,2] / self.ratio, a_min=0, a_max=self.img_w)
        new_boxes[:,3] = np.clip(boxes[:,3] / self.ratio, a_min=0, a_max=self.img_h)
        return np.hstack((new_boxes[:, :2] - new_boxes[:,2:4] / 2, new_boxes[:, :2] + new_boxes[:,2:4] / 2))  # xywh to xyxy

    def get_results_dict(self) -> list[dict[str, Any]]:
        data = self.data[0, :, :]
        data[:,:4] = self.get_resized_box_coords(data[:,:4])
        out = []
        for row in data:
            category_id = np.argmax(row[4:])
            out.append({
                'bbox': row[:4],
                'category_id': category_id,
                'label': '' if self.labels_dict is None else self.labels_dict[str(category_id)],
                'score': row[4 + category_id]
            })
        return out
    
    def get_image_overlay(self) -> ArrayLike:
        assert self.img_path is not None
        img = io.imread(self.img_path)
        for res in self.results:
            l, t, r, b = res['bbox'].astype(np.int32)
            label = f'{res["label"]} {res["score"] : 0.3f}'
            img = cv2.rectangle(img, (l, b), (r, t), color=(255,255,0), thickness=3)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            img = cv2.rectangle(img, (l, t - 20), (l + w, t), (255,255,0), -1)
            img = cv2.putText(img, label, (l, t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        return img