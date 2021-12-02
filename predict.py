import json

import torch
import os
from PIL import Image
from model import AlexNet
from torchvision import transforms
import matplotlib.pyplot as plt


data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)
class_indict = {
    "0": "雏菊",
    "1": "蒲公英",
    "2": "玫瑰",
    "3": "向日葵",
    "4": "郁金香"
}
imgs = os.listdir("data/test")
test_images = []
i = 1
for img in imgs:
    image = Image.open("data/test/"+img)
    image = image.resize((512, 384))
    ax = plt.subplot(5, 5, i)
    plt.imshow(image)
    i = i + 1
    # 去除坐标轴
    plt.xticks([])
    plt.yticks([])
    # 去除黑框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    image = data_transform(image)
    image = torch.unsqueeze(image, dim=0)
    test_images.append(image)
plt.show()
model = AlexNet(num_classes=5)
model.load_state_dict(torch.load("AlexNet.pth"))
model.eval()
with torch.no_grad():
    for test_image in test_images:
        outputs = model(test_image)
        print(class_indict[str(outputs.argmax().numpy())])
