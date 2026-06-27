import torch.nn as nn
from network.swin_unet import SwinTransformerSys as Swin_unet


class RCML(nn.Module):
    def __init__(self, args): 
        super(RCML, self).__init__()
        self.num_modality = len(args.modalities)
        self.modalities = args.modalities
        self.net = Swin_unet 
        self.NetworkGroups = nn.ModuleList([self.net(img_size=128,window_size=8,use_checkpoint=args.use_checkpoint) for i in range(self.num_modality)])
        self.soft_plus = nn.Softplus()
        pass

    def forward(self, X):
        # get evidence
        evidences = dict()
        X_predict = dict()

        for modality, net in zip(self.modalities, self.NetworkGroups):
            modality_data = X[modality]
            modality_data = modality_data.view(-1,*modality_data.shape[2:])
            temp = net(modality_data)
            X_predict[modality] = temp.view(X[modality].shape[0], X[modality].shape[1], *temp.shape[1:])
            evidences[modality] = self.soft_plus(X_predict[modality])
            
        # fusion
        evidence_a = evidences[self.modalities[0]]
        for i in range(1, self.num_modality):
            evidence_a = (evidences[self.modalities[i]] + evidence_a) / 2  
        return evidences, evidence_a
