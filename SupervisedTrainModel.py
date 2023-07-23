import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from dgl.nn.pytorch.conv import GINConv, GATConv, GraphConv, ChebConv

from MTL_MGAWS.utils import AutomaticWeightedLoss
from MTL_MGAWS.model.module import BottleneckNet, Feedforward




# used for supervised
class SupervisedTrainModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class, cheb_k, mask_rate, testmode=False):
        super(SupervisedTrainModel, self).__init__()
        self.mask_rate = mask_rate
        self.testmode = testmode
        linear_hidden = 1024
        k = cheb_k

        # share encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.encoder.append(ChebConv(hidden_dim, hidden_dim, k=k))

        # channel mask decoder
        self.CM_decoder = nn.ModuleList()
        self.CM_decoder.append(ChebConv(hidden_dim, hidden_dim, k=k))
        self.CM_decoder.append(nn.Linear(hidden_dim, input_dim, bias=False))

        # frequency mask decoder
        self.FM_decoder = nn.ModuleList()
        self.FM_decoder.append(ChebConv(hidden_dim, hidden_dim, k=k))
        self.FM_decoder.append(nn.Linear(hidden_dim, input_dim, bias=False))

        # build classify
        self.classify = nn.Sequential(
            nn.Linear(hidden_dim * 62, linear_hidden),  # 32
            nn.BatchNorm1d(linear_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(linear_hidden, linear_hidden // 2),
            nn.BatchNorm1d(linear_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(linear_hidden // 2, num_class)
        )

        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.awl_loss = AutomaticWeightedLoss(num=3)


    def forward(self, g, x, label=None):
        if not self.testmode:
            # random.seed(0)
            # mask_rate = random.uniform(0.2, 0.8)
            mask_rate = self.mask_rate

            # channel random mask and encoder decoder
            cm_use_g, cm_use_x, (mask_channels, keep_channels) = self.channel_mask_noise(g, x, mask_rate)
            cm_dec_in = F.relu(self.encoder[0](cm_use_x))
            cm_dec_in = F.relu(self.encoder[1](cm_use_g, cm_dec_in, lambda_max=[2]*g.batch_size))
            cm_out = F.relu(self.CM_decoder[0](cm_use_g, cm_dec_in, lambda_max=[2]*g.batch_size))
            cm_out = self.CM_decoder[1](cm_out)

            # frequency band random mask and encoder decoder
            fm_use_g, fm_use_x, (mask_bands, keep_bands) = self.frequencyband_mask_noise(g, x, mask_rate)
            fm_dec_in = F.relu(self.encoder[0](fm_use_x))
            fm_dec_in = F.relu(self.encoder[1](fm_use_g, fm_dec_in, lambda_max=[2]*g.batch_size))
            fm_out = F.relu(self.CM_decoder[0](fm_use_g, fm_dec_in, lambda_max=[2]*g.batch_size))
            fm_out = self.CM_decoder[1](fm_out)

            # encoder and classify
            out = F.relu(self.encoder[0](x))
            out = F.relu(self.encoder[1](g, out, lambda_max=[2]*g.batch_size))
            out = out.view(g.batch_size, -1)
            out = self.classify(out)

            # loss calculation
            # ----- channel reconstruction loss-----
            cm_init = x[mask_channels]
            cm_rec = cm_out[mask_channels]
            cm_loss = self.mse(cm_init, cm_rec)
            # cm_loss = torch.tensor(0.)

            # ----- frequency band reconstruction loss-----
            fm_init = x[:, mask_bands]
            fm_rec = fm_out[:, mask_bands]
            fm_loss = self.mse(fm_init, fm_rec)
            # fm_loss = torch.tensor(0.)

            # ----- classify loss-----
            class_loss = self.ce(out, label)

            # Automatic Weighted Loss
            loss = self.awl_loss(cm_loss, fm_loss, class_loss)
            # loss = self.awl_loss(cm_loss, class_loss)
            # loss = class_loss #+ cm_loss + fm_loss

            return out, loss, [cm_loss, fm_loss, class_loss]

        else:
            # test
            # encoder and classify
            out = F.relu(self.encoder[0](x))
            out = F.relu(self.encoder[1](g, out, lambda_max=[2] * g.batch_size))
            out = out.view(g.batch_size, -1)
            out = self.classify(out)
            # ----- classify loss-----
            loss = self.ce(out, label)

            return out, loss


    def channel_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        out_x[mask_nodes] = 0.0
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def frequencyband_mask_noise(self, g, x, mask_rate=0.3):
        frequency_bands = x.size(1)
        perm = torch.randperm(frequency_bands, device=x.device)

        # random masking
        num_mask_bands = int(mask_rate * frequency_bands)
        mask_bands = perm[: num_mask_bands]
        keep_bands = perm[num_mask_bands:]

        out_x = x.clone()
        out_x[:, mask_bands] = 0.0
        use_g = g.clone()

        return use_g, out_x, (mask_bands, keep_bands)



















