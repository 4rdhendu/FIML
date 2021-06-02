import torch.nn as nn
import torch
import torch.nn.functional as F


class SemanticClassifier(nn.Module):
    # TODO: inchannels will depend on the kind of backbone that is being used (this needs to be very generic)
    def __init__(self, num_classes=25, in_channels=2560):
        super(SemanticClassifier, self).__init__()

        # classification head - 3fc layers and
        self.fc1 = nn.Linear(in_channels, 120)
        self.fc2 = nn.Linear(120, num_classes)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    # class_gt_mask = None, one_hot = None, get_only_class_vec = False,
    def forward(self, feat):

        class_scores = self.get_classification_score(feat)

        return class_scores

    def get_classification_score(self, feat):

        fc1_out = F.relu(self.fc1(feat))
        class_out = self.fc2(fc1_out)

        return class_out
