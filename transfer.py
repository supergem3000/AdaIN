import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
from main import StyleTransferNetwork

# 输入的内容图片、风格图片
input_content_image_path = ""
input_style_image_path = ""
# 输出结果图片
output_file_path = ""

model = StyleTransferNetwork()
cuda = torch.cuda.is_available()
if cuda:
    model = model.cuda()
# 加载模型。CPU或GPU对不上的话，map_location指定一下即可
checkpoint = torch.load("")
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
    style_img = style_img.cuda()

output_tensor = model.output_pic(content_img, style_img)
output_tensor = output_tensor.squeeze(0)
output_img = transforms.ToPILImage()(output_tensor)
b, g, r = output_img.split()
output_img = Image.merge("RGB", (r, g, b))
output_img.show()
output_img.save(output_file_path)
