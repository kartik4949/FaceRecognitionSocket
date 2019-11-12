import cv2
import numpy as np

class DetectFaceBoundBoxes():
    def __init__(self,model):
        self.modelCaffe = model
        print('caffe model initialized....')

    def detectFace(self,image,h,w):
        a = None
        b = None
        confidence = None
        
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.modelCaffe.setInput(blob)
        detections = self.modelCaffe.forward()
        max_confidence = 0
        for i in range(0, detections.shape[2]):

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #print(startX, startY, endX, endY)

            confidence = detections[0, 0, i, 2]
            #print('confidence',confidence)
            
            if(confidence > 0.4 ):
                if(i==0):
                    max_confidence = confidence
                if(confidence>=max_confidence):
                    max_confidence = confidence
                    a =  (startX, startY) 
                    b = (endX, endY)  
                    #print(max_confiden
        #print(a,b,max_confidence)        
        return a, b , max_confidence
    def detect_multiFace(self,image,h,w):
        a = None
        b = None
        confidence = None
        face_list  = []
        confidence_list = []
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.modelCaffe.setInput(blob)
        detections = self.modelCaffe.forward()
        max_confidence = 0
        for i in range(0, detections.shape[2]):

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #print(startX, startY, endX, endY)

            confidence = detections[0, 0, i, 2]
            #print('confidence',confidence)
            
            if(confidence > 0.4 ):
                a =  (startX, startY) 
                b = (endX, endY)
                confidence_list.append(confidence)
                face_list.append([a,b])
                #image = cv2.rectangle(image,a,b,(0,255,0),1)

        return face_list
        


def Manage_attribute(attr = None,typespec = None):
    def checkattribute(func):
        @wraps(func)    
        def wrapper(*args,**kwargs):
            for key,value in kwargs.items():
                print(key)
                if(attr in key):
                    if not isinstance(value,typespec):
                        raise TypeError("%s Attribute is not instance of %s",(attr,typespec))
            return func(*args,**kwargs)
        return wrapper
    return checkattribute



class Face_recognition:
    def __init__(self,FILE_PATH):
        self.FILE_PATH = FILE_PATH
        '''model = cv2.dnn.readNetFromCaffe('OpenCV_repo_data/deploy.prototxt', 'OpenCV_repo_data/weights.caffemodel')
        self.df = DetectFaceBoundBoxes(model)
        self.Neighbors_KNN = Neighbors_KNN'''
        self.model = cv2.dnn.readNetFromCaffe('OpenCV_repo_data/deploy.prototxt', 'OpenCV_repo_data/weights.caffemodel')
        self.df = DetectFaceBoundBoxes(self.model)
        
    @property
    def get_path(self):
        return self.path
    
    @get_path.setter
    def get_gath(self,path):
        if not isinstance(path,str):
            raise TypeError('String Expected')
    
    @get_path.deleter
    def get_gath(self):
        raise TypeError('String Expected')
        
    def load_model(self,name):
        with open(name, "rb") as input_file:
            return pickle.load(input_file)
        

    def detectFace(self, img:list ,modelsvc) -> list:
        h , w = img.shape[:2]
        a,b,confidence = self.df.detectFace(img , h , w)
        if(a is not None ):
            x1,y1 = a
            x2,y2 = b
            if(x1 > 0 and  x2 > 0  and y1 > 0 and y2 > 0):
                

                tempimg = img
                croped_image = tempimg[y1:y2,x1:x2]
                if(len(croped_image) != 0 ):

                    tempimg =  cv2.resize(croped_image, (160,160), interpolation = cv2.INTER_AREA)
                    #tempimg =  cv2.resize(tempimg, (160,160), interpolation = cv2.INTER_AREA)
                else:
                    tempimg =  cv2.resize(tempimg, (160,160), interpolation = cv2.INTER_AREA)

                tempimg = ( tempimg - self._mean ) / self._std

                _embeddings = self._model.predict(np.reshape(tempimg,(1,160,160,3)))
                _embeddings = self._normalizer.transform(_embeddings)
                result = modelsvc.predict(_embeddings)
                resultlabel = self._le.inverse_transform(result)
                img = cv2.putText(img,resultlabel[0],(x1,y1),self._font, 1,(0,255,0),1,cv2.LINE_AA)
                img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
        return img
    
    def multi_detectFace(self,img):
        #print(img_path)
        #img = cv2.imread(img_path)
        #print(img.shape)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h , w = img.shape[:2]
        face_list = self.df.detect_multiFace(img , h , w)
        final_tempimg = []
        for faces in face_list:
            if(len(faces) != 0 ):
                x1,y1 = faces[0]
                x2,y2 = faces[1]

                if(x1 > 0 and  x2 > 0  and y1 > 0 and y2 > 0):
                    img = cv2.rectangle(img,(x1,y1),( x2,y2),(0,255,0),1,cv2.LINE_8)
                    
        return img

'''import sys
if __name__ == '__main__':
    args = sys.argv
    imagepath = args[1]
    FILE_PATH = args[-1]
    with Face_recognition(FILE_PATH) as fr:
        fr.multi_detectFace(imagepath)'''


    