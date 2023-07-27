#!/usr/bin/python

from __future__ import print_function, absolute_import, division
import os
import json
from skimage import io, filters
import numpy as np
from collections import namedtuple
from collections import defaultdict
from matplotlib import cm
import xml.etree.ElementTree as ET
import glob
import struct

# get current date and time
import datetime
import locale

# A point in a polygon
Point = namedtuple('Point', ['x', 'y'])

from abc import ABCMeta, abstractmethod
from tools.kitti360scripts.helpers.labels     import labels, id2label, kittiId2label, name2label

MAX_N = 10000
def local2global(semanticId, instanceId):
    globalId = semanticId * MAX_N + instanceId
    if isinstance(globalId, np.ndarray):
        return globalId.astype(np.int)
    else:
        return int(globalId)

def global2local(globalId):
    semanticId = globalId // MAX_N
    instanceId = globalId % MAX_N
    if isinstance(globalId, np.ndarray):
        return semanticId.astype(np.int), instanceId.astype(np.int)
    else:
        return int(semanticId), int(instanceId)

annotation2global = defaultdict()
semantic_instance = defaultdict()

# Abstract base class for annotation objects
class KITTI360Object:
    __metaclass__ = ABCMeta

    def __init__(self):
        # the label
        self.label    =  ""
        # colormap
        self.cmap = cm.get_cmap('Set1')
        self.cmap_length = 119

    def getColor(self, idx):
        if idx==0:
            return np.array([0,0,0])
        return np.asarray(self.cmap(idx % self.cmap_length)[:3])*255.

    def assignColor(self):
        if self.semanticId>=0:
            self.semanticColor = id2label[self.semanticId].color
            if self.instanceId>0:
                self.instanceColor = self.getColor(self.instanceId)
            else:
                self.instanceColor = self.semanticColor

# Class that contains the information of a single annotated object as 3D bounding box
class KITTI360Bbox3D(KITTI360Object):
    # Constructor
    def __init__(self):
        KITTI360Object.__init__(self)
        # the polygon as list of points
        self.vertices  = []
        self.faces  = []
        self.lines = [[0,5],[1,4],[2,7],[3,6],
                      [0,1],[1,3],[3,2],[2,0],
                      [4,5],[5,7],[7,6],[6,4]]

        # the ID of the corresponding object
        self.semanticId = -1
        self.instanceId = -1
        self.annotationId = -1

        # the window that contains the bbox
        self.start_frame = -1
        self.end_frame = -1

        # timestamp of the bbox (-1 if statis)
        self.timestamp = -1

        # projected vertices
        self.vertices_proj = None
        self.meshes = []

        # name
        self.name = '' 

    def __str__(self): 
        return self.name

    def generateMeshes(self):
        self.meshes = []
        if self.vertices_proj:
            for fidx in range(self.faces.shape[0]):
                self.meshes.append( [ Point(self.vertices_proj[0][int(x)], self.vertices_proj[1][int(x)]) for x in self.faces[fidx]] )

    def parseOpencvMatrix(self, node):
        rows = int(node.find('rows').text)
        cols = int(node.find('cols').text)
        data = node.find('data').text.split(' ')
        mat = []
        for d in data:
            d = d.replace('\n', '')
            if len(d) < 1:
                continue
            mat.append(float(d))
        mat = np.reshape(mat, [rows, cols])
        return mat

    def parseVertices(self, child):
        transform = self.parseOpencvMatrix(child.find('transform'))
        R = transform[:3,:3]
        T = transform[:3,3]
        vertices = self.parseOpencvMatrix(child.find('vertices'))
        faces = self.parseOpencvMatrix(child.find('faces'))
        vertices = np.matmul(R, vertices.transpose()).transpose() + T  
        self.vertices = vertices
        self.faces = faces

    def label2name(self, name):
        classmap = {
            'ground' : 'terrain' ,
            'unknownGround': 'ground',
            'pedestrian' : 'person',
            'smallPole' : 'smallpole',
            'bigPole' : 'pole',
            'driveway' : 'parking',
            'egovehicle' :'ego vehicle' ,
            'rectificationborder' :'rectification border' ,
            'outofroi'  :'out of roi',
            'railtrack' :'rail track',
            'guardrail' :'guard rail',
            'trafficLight' :'traffic light',
            'trafficSign' : 'traffic sign',
            'trashbin' : 'trash bin',
            'vendingmachine' : 'vending machine',
            'unknownConstruction' :'unknown construction',
            'unknownvehicle' :'unknown vehicle',
            'unknownVehicle' : 'unknown vehicle',
            'unknownObject':'unknown object',
            'licenseplate' :'license plate',
        }
        if name in classmap.keys():
            name = classmap[name]
        return name

    def parseBbox(self, child):
        self.annotationId = int(child.find('index').text)
        self.name = self.label2name(child.find('label').text)
        self.semanticId = name2label[self.name].id

        global semantic_instance
        if not self.semanticId in semantic_instance:
            semantic_instance[self.semanticId] = 1
        else:
            semantic_instance[self.semanticId] += 1
        self.instanceId = semantic_instance[self.semanticId]
        self.timestamp = int(child.find('timestamp').text)

        global annotation2global
        annotation2global[self.annotationId] = local2global(self.semanticId, self.instanceId)
        self.parseVertices(child)

# Class that contains the information of the point cloud a single frame
class KITTI360Point3D(KITTI360Object):
    # Constructor
    def __init__(self):
        KITTI360Object.__init__(self)

        self.vertices = []
        self.vertices_proj = None

        # the ID of the corresponding object
        self.semanticId = -1
        self.instanceId = -1
        self.annotationId = -1

        # name
        self.name = '' 

        # color
        self.semanticColor = None
        self.instanceColor = None

    def __str__(self): 
        return self.name

    def generateMeshes(self):
        pass

# The annotation of a whole image, including semantic and instance
class Annotation2D:
    # Constructor
    def __init__(self, colormap='Set1'):
        # the width of that image and thus of the label image
        self.imgWidth  = 0
        # the height of that image and thus of the label image
        self.imgHeight = 0
        self.instanceId = None
        self.semanticId = None
        self.instanceImg = None
        self.semanticImg = None

        # savedId = semanticId * N + instanceId
        self.N = 1000

        # colormap
        self.cmap = cm.get_cmap(colormap)

        if colormap == 'Set1':
            self.cmap_length = 9 
        else:
            raise "Colormap length need to be specified!"

    def getColor(self, idx):
        if idx==0:
            return np.array([0,0,0])
        return np.asarray(self.cmap(idx % self.cmap_length)[:3])*255.

    # Load confidence map
    def loadConfidence(self, imgPath):
        self.confidenceMap = io.imread(imgPath)
        self.confidenceMap = np.asarray(self.confidenceMap).astype(np.float) / 255.

    # Load instance id 
    def loadInstance(self, imgPath, gtType='instance', toImg=True, contourType='instance', semanticCt=True, instanceCt=True):
        instanceId = io.imread(imgPath)
        self.semanticId = np.asarray( instanceId // self.N )
        self.instanceId = np.asarray( instanceId % self.N )

        if not toImg:
            return

        if gtType=='semantic':
            self.toSemanticImage()

        elif gtType=='instance':
            self.toInstanceImage()

        if semanticCt or instanceCt:
            self.getBoundary()

        if gtType=='semantic' and semanticCt:
            boundaryImg = self.toBoundaryImage(contourType=contourType, instanceOnly=False)
            self.semanticImg = self.semanticImg * (1-boundaryImg) + \
                    np.ones_like(self.semanticImg) * boundaryImg * 255

        if gtType=='instance' and instanceCt:
            boundaryImg = self.toBoundaryImage(contourType=contourType, instanceOnly=True)
            self.instanceImg = self.instanceImg * (1-boundaryImg) + \
                    np.ones_like(self.instanceImg) * boundaryImg * 255

    def toSemanticImage(self):
        self.semanticImg = np.zeros((self.semanticId.size, 3))
        for label in labels:
            mask = self.semanticId==label.id
            mask = mask.flatten()
            self.semanticImg[mask] = np.asarray(label.color)
        self.semanticImg = self.semanticImg.reshape(*self.semanticId.shape, 3)

    def toInstanceImage(self):
        self.instanceImg = np.zeros((self.instanceId.size, 3))
        print(self.instanceImg)
        uniqueId = np.unique(self.instanceId)
        for uid in uniqueId:
            mask = self.instanceId==uid
            mask = mask.flatten()
            self.instanceImg[mask] = np.asarray(self.getColor(uid))
        self.instanceImg = self.instanceImg.reshape(*self.instanceId.shape, 3)

    def getBoundary(self):
        # semantic contours
        uniqueId = np.unique(self.semanticId)
        self.semanticContours = {}
        for uid in uniqueId: 
            mask = (self.semanticId==uid).astype(np.uint8) * 255
            mask_filter = filters.laplace(mask)
            self.semanticContours[uid] = np.expand_dims(np.abs(mask_filter)>0, 2)

        # instance contours
        globalId = local2global(self.semanticId, self.instanceId)
        uniqueId = np.unique(globalId)
        self.instanceContours = {}
        for uid in uniqueId:
            mask = (globalId==uid).astype(np.uint8) * 255
            mask_filter = filters.laplace(mask)
            self.instanceContours[uid] = np.expand_dims(np.abs(mask_filter)>0, 2)

    def toBoundaryImage(self, contourType='instance', instanceOnly=True):
        if contourType=='semantic':
            contours = self.semanticContours
            assert(instanceOnly==False)
        elif contourType=='instance':
            contours = self.instanceContours
        else:
            raise ("Contour type can only be 'semantic' or 'instance'!")

        if not instanceOnly: 
            boundaryImg = [contours[k] for k in contours.keys()]
        else:
            boundaryImg = [contours[k] for k in contours.keys() if global2local(k)[1]!=0]
        boundaryImg = np.sum(np.asarray(boundaryImg), axis=0)
        boundaryImg = boundaryImg>0
        return boundaryImg


class Annotation2DInstance:
    def __init__(self, gtPath, cam=0):
        # trace the instances in all images
        self.instanceDict = defaultdict(list)
        instanceDictCached = os.path.join(gtPath, 'instanceDict.json')
        print(instanceDictCached)
        if os.path.isfile(instanceDictCached) and os.path.getsize(instanceDictCached)>0:
            cachedDict = json.load( open(instanceDictCached) )
            for k,v in cachedDict.items():
                self.instanceDict[int(k)] = v
            return

        obj = Annotation2D()
        gtPaths = glob.glob( os.path.join(gtPath, 'instance', '*.png') )
        print (f'Found {len(gtPaths)} label images...')

        for i,imgPath in enumerate(gtPaths):
            if i%1000==0:
                print(f'Processed {i}/{len(gtPaths)} label images...')
            obj.loadInstance(imgPath, toImg=False)
            globalId = local2global(obj.semanticId, obj.instanceId)
            globalIdUnique = np.unique(globalId)
            for idx in globalIdUnique:
                self.instanceDict[int(idx)].append(os.path.basename(imgPath))
        json.dump(self.instanceDict, open(instanceDictCached, 'w'))

    # returns the paths that contains the specific instance
    def __call__(self, semanticId, instanceId):
        globalId = local2global(semanticId, instanceId)
        return self.instanceDict[globalId]

# Meta class for KITTI360Bbox3D
class Annotation3D:
    # Constructor
    def __init__(self, labelDir='', sequence=''):
        labelPath = os.path.join(labelDir, 'train', '%s.xml' % sequence)
        if not os.path.isfile(labelPath):
            raise RuntimeError('%s does not exist! Please specify KITTI360_DATASET in your environment path.' % labelPath)
        else:
            print('Loading %s...' % labelPath)
        self.init_instance(labelPath)

    def init_instance(self, labelPath):
        # load annotation
        tree = ET.parse(labelPath)
        root = tree.getroot()
        self.objects = defaultdict(dict)
        for child in root:
            if child.find('transform') is None:
                continue
            obj = KITTI360Bbox3D()
            obj.parseBbox(child)
            # globalId = local2global(obj.semanticId, obj.instanceId)
            self.objects[obj.annotationId][obj.timestamp] = obj
        annotationIds = np.asarray(list(self.objects.keys()))
        print(f'Loaded {len(annotationIds)} instances')

    def __call__(self, semanticId, instanceId, timestamp=None):
        globalId = local2global(semanticId, instanceId)
        if globalId in self.objects.keys():
            # static object
            if len(self.objects[globalId].keys())==1:
                if -1 in self.objects[globalId].keys():
                    return self.objects[globalId][-1]
                else:
                    return None
            # dynamic object
            else:
                return self.objects[globalId][timestamp]
        else:
            return None

class Annotation3DPly:
    # parse fused 3D point cloud
    def __init__(self, labelDir='', sequence='', isLabeled=True, isDynamic=False, showStatic=True):

        if isLabeled and not isDynamic:
            # x y z r g b semanticId instanceId isVisible
            self.fmt = '=fffBBBiiB'
            self.fmt_len = 24
        elif isLabeled and isDynamic:
            # x y z r g b semanticId instanceId isVisible timestamp
            self.fmt = '=fffBBBiiBi'
            self.fmt_len = 28
        elif not isLabeled and not isDynamic:
            # x y z r g b
            self.fmt = '=fffBBBB'
            self.fmt_len = 16
        else:
            raise RuntimeError('Invalid binary format!')

        # True for training data, False for testing data
        self.isLabeled = isLabeled
        # True for dynamic data, False for static data
        self.isDynamic = isDynamic
        # True for inspecting static data, False for inspecting dynamic data
        self.showStatic = showStatic

        pcdFolder = 'static' if self.showStatic else 'dynamic'
        trainTestDir = 'train' if self.isLabeled else 'test'
        self.pcdFileList = sorted(glob.glob(os.path.join(labelDir, trainTestDir, sequence, pcdFolder, '*.ply')))
        
        print('Found %d ply files in %s' % (len(self.pcdFileList), sequence))

    def readBinaryPly(self, pcdFile, n_pts):

        with open(pcdFile, 'rb') as f:
            plyData = f.readlines()

        headLine = plyData.index(b'end_header\n')+1
        plyData = plyData[headLine:]
        plyData = b"".join(plyData)

        n_pts_loaded = len(plyData)/self.fmt_len
        assert(n_pts_loaded==n_pts)
        n_pts_loaded = int(n_pts_loaded)

        data = []
        for i in range(n_pts_loaded):
            pts=struct.unpack(self.fmt, plyData[i*self.fmt_len:(i+1)*self.fmt_len])
            data.append(pts)
        data=np.asarray(data)

        return data

    def writeBinaryPly(self, pcdFile, data):
        fmt = '=fffBBBiiB'
        fmt_len = 24
        n_pts = data.shape[0]

        with open(pcdFile, 'wb') as f:
            f.write(b'ply\n')
            f.write(b'format binary_little_endian 1.0\n')
            f.write(b'comment author Yiyi Liao\n')
            f.write(b'element vertex %d\n' % n_pts)
            f.write(b'property float x\n')
            f.write(b'property float y\n')
            f.write(b'property float z\n')
            f.write(b'property uchar red\n')
            f.write(b'property uchar green\n')
            f.write(b'property uchar blue\n')
            f.write(b'property int semantic\n')         

# a dummy example
if __name__ == "__main__":
    ann = Annotation3D()