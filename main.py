import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import cv2
import os
from tqdm import tqdm
import random


class StyleTransferNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 只要VGG-19提取特征的部分，不要classifier
        # 论文6.1提到，只使用到预训练的VGG-19前面的一些层（到relu4_1）
        # 论文6.2 计算style loss时分别使用relu1_1, relu2_1, relu3_1, relu4_1这些层的输出
        encoder = torchvision.models.vgg19(pretrained=True).features[:21]
        # print(encoder)
        self.encoder_layer1 = encoder[:2]
        self.encoder_layer2 = encoder[2:7]
        self.encoder_layer3 = encoder[7:12]
        self.encoder_layer4 = encoder[12:21]
        # AdaIN 训练的是decoder
        self.decoder = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode="nearest"),
            # nn.ReflectionPad2d((1, 1, 1, 1)),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)),
            # nn.ReLU(),
            # nn.ReflectionPad2d((1, 1, 1, 1)),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)),
            # nn.ReLU(),
            # nn.ReflectionPad2d((1, 1, 1, 1)),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)),
            # nn.ReLU(),
            # nn.ReflectionPad2d((1, 1, 1, 1)),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)),
            # nn.ReLU(),  # 36~28

            # nn.Upsample(scale_factor=2, mode="nearest"),
            # nn.ReflectionPad2d((1, 1, 1, 1)),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)),
            # nn.ReLU(),
            # nn.ReflectionPad2d((1, 1, 1, 1)),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)),
            # nn.ReLU(),
            # nn.ReflectionPad2d((1, 1, 1, 1)),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)),
            # nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),  # 27~19

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),  # 18~10

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),  # 10~5

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1)),
            # nn.ReLU(),  # 4~0
        )
        self.loss_fn = nn.MSELoss()
        return

    def mean(self, content):
        # 每个tensor四维，batch_size, 通道数，width， height
        batch_size, channel_num = content.size()[:2]
        return content.reshape(batch_size, channel_num, -1).mean(dim=2).reshape(batch_size, channel_num, 1, 1)

    def std(self, content):
        batch_size, channel_num = content.size()[:2]
        return content.reshape(batch_size, channel_num, -1).std(dim=2).reshape(batch_size, channel_num, 1, 1)

    def adain(self, content, style):
        content_mean = self.mean(content)
        content_std = self.std(content)
        style_mean = self.mean(style)
        style_std = self.std(style)
        # AdaIN公式
        # AdaIN = σ(y) * ((x-μ(x)) / σ(x)) + μ(y)
        # 参考GitHub，σ(x)加1e-5是为了防止分母为0的情况，所有加了一个很小的值
        val = style_std * ((content - content_mean) /
                           (content_std + 1e-5)) + style_mean
        return val

    def encode(self, content):
        return self.encode_layers(content)[-1]

    def encode_layers(self, content):
        feats = []
        feats.append(self.encoder_layer1(content))
        feats.append(self.encoder_layer2(feats[0]))
        feats.append(self.encoder_layer3(feats[1]))
        feats.append(self.encoder_layer4(feats[2]))
        return feats

    def decode(self, content):
        return self.decoder(content)

    def forward(self, content, style, _lambda=10):
        content_feat = self.encode(content)
        style_feats = self.encode_layers(style)
        # 论文6.2，t表示AdaIN output，函数f是经过encoder，g是经过decoder
        t = self.adain(content_feat, style_feats[-1])
        g_t = self.decode(t)
        g_t_feats = self.encode_layers(g_t)
        f_g_t = g_t_feats[-1]

        # loss_c = ||f(g(t)) - t||_2
        loss_c = self.loss_fn(f_g_t, t)
        # loss_s = Σ||μ(φ_i(g(t))) - μ(φ_i(s))||_2 + Σ||σ(φ_i(g(t))) - σ(φ_i(s))||_2
        loss_s = self.loss_fn(self.mean(g_t_feats[0]), self.mean(
            style_feats[0])) + self.loss_fn(self.std(g_t_feats[0]), self.std(style_feats[0]))
        for i in range(1, 4):
            loss_s += self.loss_fn(self.mean(g_t_feats[i]), self.mean(
                style_feats[i])) + self.loss_fn(self.std(g_t_feats[i]), self.std(style_feats[i]))
        # loss = loss_c + λ * loss_s
        return loss_c + _lambda * loss_s

    def output_pic(self, content, style):
        content_feat = self.encode(content)
        style_feat = self.encode(style)
        t = self.adain(content_feat, style_feat)
        return self.decode(t)


class TrainDataset(Dataset):
    def __init__(self, content_dir, style_dir):
        content_img_paths = []
        style_img_paths = []
        for img_name in os.listdir(content_dir):
            content_img_paths.append(os.path.join(content_dir, img_name))
        for img_name in os.listdir(style_dir):
            style_img_paths.append(os.path.join(style_dir, img_name))
        self.content_img_paths = content_img_paths
        self.style_img_paths = style_img_paths
        return

    def __getitem__(self, index):
        if index == 0:
            self.shuffle()
        content_img_path = self.content_img_paths[index]
        style_img_path = self.style_img_paths[index]
        # 把图片都缩放到256*256
        content_img = cv2.imread(content_img_path)
        content_img = cv2.resize(content_img, (256, 256))
        content_img = transforms.ToTensor()(content_img)
        style_img = cv2.imread(style_img_path)
        style_img = cv2.resize(style_img, (256, 256))
        style_img = transforms.ToTensor()(style_img)
        return content_img, style_img

    def __len__(self):
        return min(len(self.content_img_paths), len(self.style_img_paths))

    def shuffle(self):
        # 随机打乱顺序
        random.shuffle(self.content_img_paths)
        random.shuffle(self.style_img_paths)
        return


def save_model(epoch):
    if not os.path.exists("models"):
        os.mkdir("models")
    torch.save(model.state_dict(), os.path.join(
        "models", "AdaIN_epoch_{}".format(epoch)))
    return


def train(epochs):
    model.train()
    for epoch in range(0, epochs):
        print("epochs: " + str(pre + 1 + epoch))
        losses = 0.0
        for i, (content_img, style_img) in tqdm(enumerate(data_loader), total=len(data_loader)):
            if cuda:
                content_img = content_img.cuda()
                style_img = style_img.cuda()
            loss = model(content_img, style_img)
            losses += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        save_model(pre + 1 + epoch)
        print("loss: " + str(losses))
    return


if __name__ == "__main__":

    # 内容图片、风格图片训练集对应文件夹
    content_dir = "content"
    style_dir = "style"

    dataset = TrainDataset(content_dir=content_dir, style_dir=style_dir)
    data_loader = DataLoader(dataset, batch_size=16)
    pre = 1
    cuda = torch.cuda.is_available()

    model = StyleTransferNetwork()
    if cuda:
        model = model.cuda()

    # 如果之前训练过一部分，可以加载模型继续训练
    if pre >= 0:
        checkpoint = torch.load(os.path.join(
            "models", "AdaIN_epoch_{}".format(pre)))
        model.load_state_dict(checkpoint)

    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)

    # 训练几次
    train(200)
