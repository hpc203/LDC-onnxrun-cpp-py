# LDC-onnxrun-cpp-py
使用ONNXRuntime部署一种用于边缘检测的轻量级密集卷积神经网络，包含C++和Python两个版本的程序。
在本套程序里提供了3个onnx文件，每个onnx文件只有2.92M，从文件大小就可以看出
这个神经网络模型是一个轻量级的。

起初我想使用opencv做部署的，但是opencv-dnn的推理输出的特征图，跟onnxruntime的
输出特征图有差异，导致最后生成的可视化结果图片差异很大，从视觉效果看，
onnxruntime的的推理结果是正确合理的。
