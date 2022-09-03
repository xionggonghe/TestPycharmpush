from Unet import UNet
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToPILImage
from PIL import Image


if __name__ == '__main__':
    # 读取模型
    model = UNet()
    # y = torch.load(r"C:\project\pro\Derain\model\model_latest.pth")
    # y.state_dict()
    model.load_state_dict(torch.load(r"C:\project\pro\Derain\model\model_latest.pth")["state_dict"])
    # 加载图像
    transformations = transforms.Compose([
        # transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    filepath = r"C:\project\pro\Derain\dataset\rain_data_test_Heavy\input\rain-1.png"
    image = Image.open(filepath)
    img_PIL = image.convert('RGB')
    img_PIL_Tensor = transformations(img_PIL)
    x = img_PIL_Tensor.unsqueeze(0)
    outputs = model(x)

    # 显示图像
    show = ToPILImage() # 可以把Tensor转成Image，方便可视化
    show(outputs[0, :, :, :]).show()


