# AdaIN

机器学习课程大作业，使用PyTorch实现AdaIN进行风格迁移。

## 使用说明

main.py定义了网络结构和训练过程。

content文件夹存放内容图片，style文件夹存放风格图片。

每训练一个epoch会在models文件夹做一个存档。

transfer.py加载训练好的模型进行风格迁移。输出图片大小为256\*256。
