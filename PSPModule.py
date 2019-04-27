import torch
from torch import nn
from torch.nn import functional as F

import feature_extractor as extractors


class PSPModule(nn.Module):
    def __init__(self, features, sizes = (1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size = (size, size))
        conv = nn.Conv2d(features, features/4, kernel_size = 1, bias = False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input = stage(feats), size = (h, w), mode = 'bilinear') for stage in self.stages] + [feats]
        print(priors)
        concat = torch.cat(priors, dim = 1)
        return self.relu(concat)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        h, w = 8 * x.size(2), 8 * x.size(3)
        p = F.upsample(input = x, size = (h, w), mode = 'bilinear')
        return self.conv(p)

class PSPNet(nn.Module):
    def __init__(self, n_classes = 20, sizes = (1, 2, 3, 6), feat_extr_size = 2048, aux_features_size = 1024, backend = 'resnet101', pretrained = True):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(feat_extr_size, sizes)
        self.drop_1 = nn.Dropout2d(p = 0.3)
        self.up = PSPUpsample(2 * feat_extr_size, n_classes)
        self.drop_2 = nn.Dropout2d(p = 0.15)
        self.classifier = nn.Softmax(dim = 1)
        self.aux_up = PSPUpsample(aux_features_size, n_classes)
        self.aux_classifier = nn.Softmax(dim = 1)

    def forward(self, x):
        f, class_f = self.feats(x) 
        
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up(p)
        p = self.drop_2(p)
                
        auxiliary = self.aux_up(class_f)

        return self.classifier(p), self.aux_classifier(auxiliary)