import argparse
import cv2
import numpy as np
import onnxruntime


class LDC():
    def __init__(self, modelpath):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.net = onnxruntime.InferenceSession(modelpath, so)
        self.input_height = self.net.get_inputs()[0].shape[2]
        self.input_width = self.net.get_inputs()[0].shape[3]
        self.input_name = self.net.get_inputs()[0].name
        output_names = [x.name for x in self.net.get_outputs()]

    def detect(self, srcimg):
        img = cv2.resize(srcimg, dsize=(self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        outs = self.net.run(None, {self.input_name: blob})

        image_width, image_height = srcimg.shape[1], srcimg.shape[0]
        for index, result in enumerate(outs):
            mask = np.squeeze(result)
            mask = 1 / (1 + np.exp(-mask))  ### sigmoid
            min_value = np.min(mask)
            max_value = np.max(mask)
            mask = (mask - min_value) * 255 / (max_value - min_value + 1e-12)
            mask = mask.astype('uint8')
            mask = cv2.bitwise_not(src=mask)
            mask = cv2.resize(mask, (image_width, image_height))

            outs[index] = mask

        average_image = np.uint8(np.mean(outs, axis=0))
        fuse_image = outs[index]
        return average_image, fuse_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='images/IMG_2567.jpg')
    parser.add_argument("--modelpath", type=str, default='weights/LDC_640x360.onnx')
    args = parser.parse_args()

    mynet = LDC(args.modelpath)
    srcimg = cv2.imread(args.imgpath)
    average_image, fuse_image = mynet.detect(srcimg)

    cv2.namedWindow('srcimg', cv2.WINDOW_NORMAL)
    cv2.imshow('srcimg', srcimg)
    cv2.namedWindow('LDC Output(Average)', cv2.WINDOW_NORMAL)
    cv2.imshow('LDC Output(Average)', average_image)
    cv2.namedWindow('LDC Output(Fuse)', cv2.WINDOW_NORMAL)
    cv2.imshow('LDC Output(Fuse)', fuse_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()