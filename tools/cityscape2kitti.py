
import numpy as np
import cv2
import os
from glob import glob
from collections import namedtuple

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'kittiId'     , # An integer ID that is associated with this label for KITTI-360
                    # NOT FOR RELEASING

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'ignoreInInst', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations of instance segmentation or not

    'color'       , # The color of this label
    ] )

labels = [
    #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   ignoreInInst   color
    Label(  'unlabeled'            ,  0 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        1 ,         0 , 'flat'            , 1       , False        , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        3 ,         1 , 'flat'            , 1       , False        , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,        2 ,       255 , 'flat'            , 1       , False        , True         , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,        10,       255 , 'flat'            , 1       , False        , True         , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        11,         2 , 'construction'    , 2       , True         , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        7 ,         3 , 'construction'    , 2       , False        , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        8 ,         4 , 'construction'    , 2       , False        , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,        30,       255 , 'construction'    , 2       , False        , True         , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,        31,       255 , 'construction'    , 2       , False        , True         , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,        32,       255 , 'construction'    , 2       , False        , True         , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        21,         5 , 'object'          , 3       , True         , False        , True         , (153,153,153) ),
    Label(  'polegroup'            , 18 ,       -1 ,       255 , 'object'          , 3       , False        , True         , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        23,         6 , 'object'          , 3       , True         , False        , True         , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        24,         7 , 'object'          , 3       , True         , False        , True         , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        5 ,         8 , 'nature'          , 4       , False        , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        4 ,         9 , 'nature'          , 4       , False        , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        9 ,        10 , 'sky'             , 5       , False        , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,        19,        11 , 'human'           , 6       , True         , False        , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,        20,        12 , 'human'           , 6       , True         , False        , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,        13,        13 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,        14,        14 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,        34,        15 , 'vehicle'         , 7       , True         , False        , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,        16,       255 , 'vehicle'         , 7       , True         , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,        15,       255 , 'vehicle'         , 7       , True         , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,        33,        16 , 'vehicle'         , 7       , True         , False        , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,        17,        17 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,        18,        18 , 'vehicle'         , 7       , True         , False        , False        , (119, 11, 32) ),
    Label(  'gate'                 , 35 ,        6 ,         4 , 'construction'    , 2       , False        , True         , True         , (190,153,153) ),
    Label(  'stop'                 , 36 ,        29,       255 , 'construction'    , 2       , True         , True         , True         , (150,120, 90) ),
    Label(  'lamp'                 , 38 ,        25,       255 , 'object'          , 3       , True         , True         , True         , (0,   64, 64) ),
    Label(  'trash bin'            , 39 ,        26,       255 , 'object'          , 3       , True         , True         , True         , (0,  128,192) ),
    Label(  'vending machine'      , 40 ,        27,       255 , 'object'          , 3       , True         , True         , True         , (128, 64,  0) ),
    Label(  'box'                  , 41 ,        28,       255 , 'object'          , 3       , True         , True         , True         , (64,  64,128) ),
    Label(  'unknown construction' , 42 ,        35,       255 , 'void'            , 0       , False        , True         , True         , (102,  0,  0) ),
    Label(  'unknown vehicle'      , 43 ,        36,       255 , 'void'            , 0       , False        , True         , True         , ( 51,  0, 51) ),
    Label(  'unknown object'       , 44 ,        37,       255 , 'void'            , 0       , False        , True         , True         , ( 32, 32, 32) ),
    Label(  'license plate'        , -1 ,        -1,        -1 , 'vehicle'         , 7       , False        , True         , True         , (  0,  0,142) ),
]

cityscapeId2kittiId   = { label.trainId: label.id for label in labels  }
kittiId2color   = { label.id: label.color for label in labels  }

def convert(images_list, result_dir):
    for path in images_list:
        pred = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        H, W = pred.shape
        pred_color = np.zeros((H,W,3))
        for i in range(H):
            for j in range(W):
                pred[i,j] = np.array(cityscapeId2kittiId[pred[i,j]])
                pred_color[i,j] = np.array(kittiId2color[pred[i,j]])


        pred_color = pred_color[...,::-1]
        np.save(os.path.join(result_dir, path[-14:-4]+'.npy'), pred)
        cv2.imwrite(os.path.join(result_dir,path[-14:-4]+'_color.png'), pred_color)

for root, dirs, files in os.walk('data/datasets/KITTI-360/deeplab/'):
    for temp_dir in dirs:
        dir1 = os.path.join(root, temp_dir)
        for root1, dirs1, files1 in os.walk(dir1):
            for temp_dir1 in dirs1:
                if temp_dir1 == 'image_01':
                    continue
                dir2 = os.path.join(root1, temp_dir1)
                result_dir =  dir2
                images_list = sorted(glob(os.path.join(dir2, "*.png")))
                convert(images_list, result_dir)
