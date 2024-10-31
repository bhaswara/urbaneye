# The class containing the model
from numpy.lib.type_check import imag
import torch
from PIL import Image
import torchvision
from torchvision import transforms
import numpy as np
import io
import easyfsl
import base64
import time

#import matplotlib.pyplot as plt

class ProtoNet:
    def __init__(self):
        trained_weight_path = 'weights/protonet_byol_13class_5shot.pth'
        support_image_path = 'weights/support_image.pt'
        support_label_path = 'weights/support_label.pt'

        self.classes = ['Arus Lalu Lintas', 'Banjir', 'Coretan', 'Iklan Rokok', 'Jalan', 'Jaringan Air Bersih', 
                        'Jaringan Komunikasi', 'Parkir Liar', 'Pohon', 'Saluran Air Kali Sungai', 'Sampah', 'Taman', 
                        'Tata Ruang dan Bangunan', 'Trotoar', 'Tutup Saluran', 'UMKM']

        self.device = "cpu"

        self.support_image = torch.load(support_image_path, map_location=self.device)
        self.support_label = torch.load(support_label_path, map_location=self.device)

        self.model = self.load_model(trained_weight_path)

        self.model.eval()
    
    def infer(self, image):
        #input_image_ori = cv2.imread(image)
        #input_image = cv2.cvtColor(input_image_ori, cv2.COLOR_BGR2RGB)

        #image_array = np.asarray(bytearray(image), dtype=np.uint8)
        #img_opencv = cv2.imdecode(image_array, -1)
        #input_image = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        b64 = base64.b64encode(open(image,'rb').read())
        input_image = Image.open(io.BytesIO(base64.b64decode(b64)))
        #input_image = Image.open(image)

        temp_label_srtd = []
        temp_conf_srtd = []
        with torch.no_grad():
            preprocess = transforms.Compose([transforms.Resize((84,84)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                        std = [0.229, 0.224, 0.225])])

            image = preprocess(input_image)
            image = image.unsqueeze(0)
            self.model.process_support_set(self.support_image.to(self.device), self.support_label.to(self.device))
            pred = self.model(image.to(self.device))
            pred = np.array(pred.squeeze(0))
            similarity = []
            for i in range(len(pred)):
                similarity.append(1 - np.abs(pred[i]) / np.max(np.abs(pred)).item())
            similarity_perc = [round(x*100,2) for x in similarity]
            zipped = list(zip(similarity_perc, self.classes))
            srtd = sorted(zipped, key=lambda t: t[0], reverse=True)

            for t in srtd[:5]:
                prob, label = t
                temp_label_srtd.append(label)
                temp_conf_srtd.append(prob)
        end_time = time.time()
        exec_time = end_time-start_time
            
        return (input_image, temp_label_srtd, temp_conf_srtd, exec_time)
    
    def load_model(self, fpath):
        check = torch.load(fpath, map_location=self.device)
        model = check['model']
        model.load_state_dict(check['state_dict'])
        return model

'''
if __name__ == '__main__':
    start = time.time()
    model = ProtoNet()
    a,b,c = model.infer('test_images/2969_Iklan Rokok.jpg')
    end = time.time()
    elapsed = end-start
    print(b)
    print(c)
    print(elapsed)
    #print(confidence)
    #imS = cv2.resize(image, (640, 480)) 

    #cv2.imshow('test', imS)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
'''