# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of breast_cancer_classifier.
#
# breast_cancer_classifier is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# breast_cancer_classifier is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with breast_cancer_classifier.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
"""
Defines the heatmap generation model used in run_producer.py
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad
import torchvision.models.densenet as densenet


class ModifiedDenseNet121(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.densenet = densenet.densenet121(*args, **kwargs)
        self._is_modified = False

    def _modify_densenet(self):
        """
        Replace Conv2d and MaxPool2d to resolve the differences in padding 
        between TensorFlow and PyTorch
        """
        assert not self._is_modified
        for full_name, nn_module in self.densenet.named_modules():
            if isinstance(nn_module, (nn.Conv2d, nn.MaxPool2d)):
                module_name_parts = full_name.split(".")
                parent = self._get_module(self.densenet, module_name_parts[:-1])
                actual_module_name = module_name_parts[-1]
                assert "conv" in module_name_parts[-1] or "pool" in module_name_parts[-1]
                setattr(parent, actual_module_name, TFSamePadWrapper(nn_module))
        self._is_modified = True

    def load_from_path(self, model_path):
        self.densenet.load_state_dict(torch.load(model_path))
        self._modify_densenet()

    def forward(self, x):
        if not self._is_modified:
            self._modify_densenet()
        features = self.densenet.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.densenet.classifier(out)
        return out

    @classmethod
    def _get_module(cls, model, module_name_parts):
        obj = model
        for module_name_part in module_name_parts:
            obj = getattr(obj, module_name_part)
        return obj


class TFSamePadWrapper(nn.Module):
    """
    Outputs a new convolutional or pooling layer which uses TensorFlow-style "SAME" padding
    """
    def __init__(self, sub_module):
        super(TFSamePadWrapper, self).__init__()
        self.sub_module = copy.deepcopy(sub_module)
        self.sub_module.padding = 0
        if isinstance(self.sub_module.kernel_size, int):
            self.kernel_size = (self.sub_module.kernel_size, self.sub_module.kernel_size)
            self.stride = (self.sub_module.stride, self.sub_module.stride)
        else:
            self.kernel_size = self.sub_module.kernel_size
            self.stride = self.sub_module.stride

    def forward(self, x):
        return self.sub_module(self.apply_pad(x))

    def apply_pad(self, x):
        pad_height = self.calculate_padding(x.shape[2], self.kernel_size[0], self.stride[0])
        pad_width = self.calculate_padding(x.shape[3], self.kernel_size[1], self.stride[1])

        pad_top, pad_left = pad_height // 2, pad_width // 2
        pad_bottom, pad_right = pad_height - pad_top, pad_width - pad_left
        return pad(x, [pad_top, pad_bottom, pad_left, pad_right])

    @classmethod
    def calculate_padding(cls, in_dim, kernel_dim, stride_dim):
        if in_dim % stride_dim == 0:
            return max(0, kernel_dim - stride_dim)
        return max(0, kernel_dim - (in_dim % stride_dim))
