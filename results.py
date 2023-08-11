import numpy as np
from skimage.io import imread
import cv2
from typing import Optional, Any
from numpy.typing import ArrayLike

class Results:
    def __init__(self, data:ArrayLike, model_input_shape:tuple[int], labels_dict:Optional[dict[str,str]]=None, img_path:Optional[str]=None):
        self.data = data
        self.labels_dict = labels_dict
        self.img_path = img_path
        self.orig_img = imread(img_path)
        self.img_h, self.img_w = self.orig_img.shape[:2]
        h, w = model_input_shape
        self.ratio = min(h / self.img_h, w / self.img_w)
        self.pad = [(w - self.img_w * self.ratio) // 2, (h - self.img_h * self.ratio) // 2] * 2

    def __repr__(self):
        return ''.join(f'- bbox: {r["bbox"].tolist()}\n  category_id: {r["category_id"]}\n  label: {r["label"]}\n  score: {r["score"]}\n' for r in self.results)

    @property
    def results(self) -> list[dict[str, Any]]:
        boxes = np.clip((self.data[0,:,:4] - self.pad) / self.ratio, a_min=0, a_max=(self.img_w, self.img_h, self.img_w, self.img_h))
        return [{
                'bbox': box,
                'category_id': category_id,
                'label': '' if self.labels_dict is None else self.labels_dict[str(category_id)],
                'score': self.data[0,i, 4 + category_id]
                } for i, (box, category_id) in enumerate(zip(boxes, np.argmax(self.data[0,:,4:], axis=-1)))]

    @property
    def image_overlay(self) -> ArrayLike:
        assert self.orig_img is not None
        img = self.orig_img.copy()
        for res in self.results:
            l, t, r, b = res['bbox'].astype(np.int32)
            label = f'{res["label"]} {res["score"] : 0.3f}'
            img = cv2.rectangle(img, (l, b), (r, t), color=(255,255,0), thickness=3)
            fsize = max(self.orig_img.shape) / 750
            weight = max(1, int(fsize))
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fsize, weight)
            img = cv2.rectangle(img, (l, t - h), (l + w, t), (255,255,0), -1)
            img = cv2.putText(img, label, (l, t - 5), cv2.FONT_HERSHEY_SIMPLEX, fsize, (0,0,0), weight)
        return img