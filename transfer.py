import torch
import torchvision.transforms as transforms
import cv2
from main import StyleTransferNetwork

# 输入的内容图片、风格图片
input_content_image_path = ".\\content\\126.jpg"
input_style_image_path = ".\\style\\2323713.jpg"
# 输出结果图片
output_file_path = "gg.jpg"

model = StyleTransferNetwork()
cuda = torch.cuda.is_available()
if cuda:
    model = model.cuda()
checkpoint = torch.load(".\\models\\AdaIN_epoch_5")
model.load_state_dict(checkpoint)
model.eval()

content_img = cv2.imread(input_content_image_path)
content_img = cv2.resize(content_img, (256, 256))
content_img = transforms.ToTensor()(content_img).unsqueeze(0)
if cuda:
    content_img = content_img.cuda()

style_img = cv2.imread(input_style_image_path)
style_img = cv2.resize(style_img, (256, 256))
style_img = transforms.ToTensor()(style_img).unsqueeze(0)
if cuda:
    content_img = content_img.cuda()

output_tensor = model.output_pic(content_img, style_img)
output_tensor = output_tensor.squeeze(0)
output_img = transforms.ToPILImage()(output_tensor)
output_img.show()
output_img.save(output_file_path)
