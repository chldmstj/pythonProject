from Mask_RCNN.mrcnn import utils
import os
import json
import numpy as np
from PIL import Image
import cv2


class DefectDataset(utils.Dataset):
    def load_dataset(self, dataset_dir):
        self.add_class('dataset', 1, 'defect')
        # self.add_class('dataset', 2, 'Arduino_Nano')
        # self.add_class('dataset', 3, 'ESP8266')
        # self.add_class('dataset', 4, 'Heltec_ESP32_Lora')

        # find all images
        for i, filename in enumerate(os.listdir(dataset_dir+'/origin')):
            if '.bmp' in filename:
                self.add_image('dataset',
                               image_id=i,
                               path=os.path.join(dataset_dir,'origin', filename),
                               annotation=os.path.join(dataset_dir,'label_Data', filename.replace('.bmp', '.json')))

        #print(self.image_info)

    def extract_masks(self, filename):
        json_file = os.path.join(filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        #masks = np.zeros([600, 800, len(img_anns['shape'])], dtype='uint8')np.reshape
        #masks = np.zeros([400,600,2],dtype='uint8')


        classes = []
        #mask = np.zeros([400, 600], dtype=np.uint8)
        #cv2.fillPoly(mask, np.array([img_anns['shape']['mask']], dtype=np.int32), 1)


        mask = np.array(img_anns['shape']['mask'],dtype=np.int32)
        masks = mask.reshape(400,600,1)

        #masks[:,:,1] = mask
        #classes.append(img_anns['shape']['label'])
        classes.append(self.class_names.index(img_anns['shape']['label']))

        # for i, anno in enumerate(img_anns['shapes']):
        #     mask = np.zeros([600, 800], dtype=np.uint8)
        #     cv2.fillPoly(mask, np.array([anno['points']], dtype=np.int32), 1)
        #     masks[:, :, i] = mask
        #     classes.append(self.class_names.index(anno['label']))
        #     # print(masks.shape)
        #print(masks.shape)
        #print(classes)
        return masks, classes

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        masks, classes = self.extract_masks(path)
        return masks, np.asarray(classes, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']