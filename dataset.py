import os
import torch

from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoDataset(Dataset):

    def __init__(self, root, annotation, transforms):

        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.img_ids = list(sorted(self.coco.imgs.keys()))[:50]
    
    def __getitem__(self, idx):
        
        img_id = self.img_ids[idx]
        anns_ids = self.coco.getAnnIds(imgIds=img_id)
        img_name = self.coco.loadImgs(ids=img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, img_name))
        anns = self.coco.loadAnns(ids=anns_ids)
        num_objs = len(anns)
        boxes = []
        labels = []
        areas = []
        masks = []

        for i in range(num_objs):
            xmin = anns[i]["bbox"][0]
            ymin = anns[i]["bbox"][1]
            xmax = xmin + anns[i]["bbox"][2]
            ymax = ymin + anns[i]["bbox"][3]

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(anns[i]["category_id"])
            areas.append(anns[i]["area"])
            masks.append(self.coco.annToMask(anns[i]))
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        is_crowd = torch.as_tensor([anns[0]["iscrowd"]], dtype=torch.int64)
        image_id = torch.tensor([img_id])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["masks"] = masks
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = is_crowd
        
        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.img_ids)