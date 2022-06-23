import timm
import torch
from FeatureCloud.app.engine.app import LogLevel
from torch import nn
import timm


class modified_model(nn.Module):
    def __init__(self, model_name, is_pretrained, in_features):
        super().__init__()
        self.log('beforetimm', LogLevel.DEBUG)

        model = timm.create_model(model_name=model_name, pretrained=is_pretrained, in_chans=1, num_classes=14)

        self.model_wo_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.atelectasis = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.cardiomegaly = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.effusion = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.cardiomegaly = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.infiltration = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.mass = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.nodule = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.pneumonia = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.pneumothorax = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.consolidation = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.edema = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.emphysema = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.fibrosis = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.pleural = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        self.hernia = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        """
        self.disease3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2)
        )
        """

    def forward(self, x):
        x = self.model_wo_fc(x)
        x = torch.flatten(x, 1)

        return {
            'atelectasis': self.atelectasis(x),
            'cardiomegaly': self.cardiomegaly(x),
            'effusion': self.cardiomegaly(x),
            'infiltration': self.cardiomegaly(x),
            'mass': self.cardiomegaly(x),
            'nodule': self.cardiomegaly(x),
            'pneumonia': self.cardiomegaly(x),
            'pneumothorax': self.cardiomegaly(x),
            'consolidation': self.cardiomegaly(x),
            'edema': self.cardiomegaly(x),
            'emphysema': self.cardiomegaly(x),
            'fibrosis': self.cardiomegaly(x),
            'pleural': self.cardiomegaly(x),
            'hernia': self.cardiomegaly(x),
           # 'epoch': self.epoch(x)
        }
