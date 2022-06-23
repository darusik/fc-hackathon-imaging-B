import timm
import torch
from torch import nn
import timm


class MultilabelModel(nn.Module):
    def __init__(self, model_name, is_pretrained, in_chans, num_classes, in_features):
        super().__init__()

        model = timm.create_model(model_name=model_name, pretrained=is_pretrained, in_chans=in_chans, num_classes=num_classes)

        self.model_wo_fc = nn.Sequential(*(list(model.children())[:-1]))

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

    def forward(self, x):
        x = self.model_wo_fc(x)
        x = torch.flatten(x, 1)

        return {
            'atelectasis': self.atelectasis(x),
            'cardiomegaly': self.cardiomegaly(x),
            'effusion': self.effusion(x),
            'infiltration': self.infiltration(x),
            'mass': self.mass(x),
            'nodule': self.nodule(x),
            'pneumonia': self.pneumonia(x),
            'pneumothorax': self.pneumothorax(x),
            'consolidation': self.consolidation(x),
            'edema': self.edema(x),
            'emphysema': self.emphysema(x),
            'fibrosis': self.fibrosis(x),
            'pleural': self.pleural(x),
            'hernia': self.hernia(x),
        }
