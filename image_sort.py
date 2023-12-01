import glob
import cv2
import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torchvision


# def sort(im_paths):
# model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# knn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')

# weights = ResNet50_Weights.DEFAULT
# preprocess = weights.transforms()
# feats = []
# with torch.inference_mode():
#     for im in im_paths:
#         img_transformed = preprocess(Image.open(im))
#         feats.append(model(img_transformed.unsqueeze(0)).squeeze(0))
# feats = np.array(feats)
# print(feats.shape)
# nbrs = knn.fit(feats)
# distances, indices = nbrs.kneighbors(feats)
# print(indices)
# print(distances)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True,use_style= False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.use_style = use_style
        self.style_layers = [2,3] if self.use_style  == True else []
        self.register_buffer("mean", torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3]):
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(
                224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(
                224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in self.style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


def sort(paths):
    lossfn = VGGPerceptualLoss()
    ims = [cv2.imread(path) for path in paths]
    for i in range(len(ims)):
        for j in range(len(ims)):
            loss = lossfn(torch.tensor(ims[i]), torch.tensor(ims[j]))
            print(loss)
        print()


if __name__ == "__main__":
    img_path_list = glob.glob("in2/*")
    sort(img_path_list)
