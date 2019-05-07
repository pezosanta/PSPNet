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
        conv = nn.Conv2d(in_channels = features, out_channels = int(features/4), kernel_size = 1, bias = False)
        torch.nn.init.xavier_uniform_(conv.weight)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)        
        priors = [F.upsample(input = stage(feats), size = (h, w), mode = 'bilinear') for stage in self.stages] + [feats]
                       
        concat = torch.cat(priors, dim = 1)

        print('concat')
        print(concat.shape)

        return self.relu(concat)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()        
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        '''
        nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        '''

    def forward(self, x):        
        p = self.conv(x)
        p = self.batchnorm(p)
        p = self.relu(p)

        print('final_conv')
        print(p.shape)

        h, w = int(8 * x.size(2)), int(8 * x.size(3))
        p = F.upsample(input = p, size = (h, w), mode = 'bilinear')

        print('upasmple')
        print(p.shape)

        return p

class PSPNet(nn.Module):
    def __init__(self, n_classes = 20, sizes = (1, 2, 3, 6), feat_extr_size = 2048, aux_features_size = 1024, backend = 'resnet101', pretrained = True):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(feat_extr_size, sizes)
        self.drop_1 = nn.Dropout2d(p = 0.3)
        self.up = PSPUpsample(in_channels = int(2 * feat_extr_size), out_channels = n_classes)
        self.drop_2 = nn.Dropout2d(p = 0.15)
        self.classifier = nn.Softmax(dim = 1)
        self.aux_up = PSPUpsample(aux_features_size, n_classes)
        self.aux_classifier = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f, class_f = self.feats(x) 
        print('intoPSP')
        print(f.shape)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up(p)
        p = self.drop_2(p)
        p = self.classifier(p)
                
        auxiliary = self.aux_up(class_f)
        auxiliary = self.aux_classifier(auxiliary)

        p = self.sigmoid(p)
        auxiliary = self.sigmoid(auxiliary)
        
        
        return p, auxiliary