from pycocotools.coco import COCO
import numpy as np
import cv2
import csv

annotation_file = 'labels.json'

coco = COCO(annotation_file)

img_ids = coco.getImgIds()

with open('coco_features.csv', mode='w', newline='') as file:

    writer = csv.writer(file)
    writer.writerow(['x', 'y', 'width', 'height', 'keypoint1_x', 'keypoint1_y', ..., 'label'])
     
    for ii in range(640):

        img_id = img_ids[ii]
        img_info = coco.loadImgs(img_id)[0]

        image = cv2.imread(f'images/{img_info["file_name"]}')

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:

            bbox = ann['bbox']

            category_id = ann['category_id']

            keypoints = ann.get('keypoints', None)

            features = bbox
            
            if keypoints:
                features.extend(keypoints)

             # Feature columns

            bbox = ann['bbox']
            keypoints = ann.get('keypoints', [])
            label = ann['category_id']

            features = bbox + keypoints
            writer.writerow(features + [label])

            print(f'Extracted features: {features}, Label: {category_id}')

            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255,0,0),2)


cv2.imshow('COCO Dataset Final Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

