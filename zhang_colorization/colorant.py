import cv2
import numpy as np

class Colorant:

    def __init__(self, proto, model, hull):
        self.net = cv2.dnn.readNetFromCaffe(proto, model)
        pts_in_hull = np.load(hull)
         
        pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)

        self.net.getLayer(self.net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
        self.net.getLayer(self.net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

        self.INPUT_SIZE = 224

    # Receives a RGB with [0-255] values numpy array representing the image
    def run(self, img):
        img = (img * 1.0 / 255).astype(np.float32)
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        img_l = img_lab[:, :, 0]

        img_l_rs = cv2.resize(img_l, (self.INPUT_SIZE, self.INPUT_SIZE))  # resize image to network input size
        img_l_rs -= 50  # subtract 50 for mean-centering

        self.net.setInput(cv2.dnn.blobFromImage(img_l_rs))
        out = self.net.forward()
        out = out[0,:,:,:].transpose((1, 2, 0))

        h, w, _ = img.shape
        out = cv2.resize(out, (w, h))
        img_lab_out = np.concatenate((img_l[:, :, np.newaxis], out), axis=2) # concatenate with original image L
        img_rgb_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2RGB), 0, 1)

        img_rgb_out = (img_rgb_out * 255).astype(np.uint8)

        return img_rgb_out
