import numpy as np

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def save_results_coco_json(results, jdict, image_id, class_map):
    for result in results:
        box = xyxy2xywh(result['bbox'].reshape(1,4))  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy ce
        box=box.reshape(-1).tolist()
        jdict.append({'image_id': image_id,
                      'category_id': class_map[result['category_id']],
                      'bbox': [float(x) for x in box],
                      'score': float(result['score'])})