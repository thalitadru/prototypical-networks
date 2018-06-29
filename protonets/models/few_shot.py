import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np
from sklearn.cluster import k_means

from protonets.models import register_model

from .utils import euclidean_dist

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Multiproto(nn.Module):
    def __init__(self, samples, n_dim, n_proto, n_classes,
                 use_hidden=True,
                 concat=True,
                 temperature=1,
                 init='kmeans++', n_init=10):
        super().__init__()
        self.temperature = temperature
        self.concat = concat

        n_class = samples.size(0)
        n_samples = samples.size(1)
        assert n_dim == samples.size(2)

        n_proto_per_class = n_proto // n_classes
        n_proto = n_proto_per_class*n_class

        self.n_proto = n_proto
        self.n_proto_per_class = n_proto_per_class
        self.n_classes = n_classes

        if init == 'kmeans++':
            #init with kmeans one iter to use kmeans++
            x_kmeans = samples.view(n_class*n_samples, -1).data.cpu().numpy()
            y_data = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_samples, 1).reshape(n_class*n_samples,1).long()
            centroid, label, inertia = k_means(
                x_kmeans, int(n_proto),
                precompute_distances=True if n_samples * n_dim > 1e4 else False,
                max_iter=1, n_init=n_init, random_state=0)
            centroid = centroid.astype(np.float32)
            init_proto = torch.from_numpy(centroid)
            if samples.is_cuda:
                init_proto = init_proto.cuda()

            #init wc
            if not concat:
                y = torch.zeros(n_proto, 1).long()
                for c in range(n_proto):
                    idx = [i for i, l in enumerate(label) if l==c]
                    y_proto = y_data[idx]
                    y[c] = torch.unique(y_proto, sorted=True)[-1].long()
                init_wc = torch.FloatTensor(n_proto, n_class)
                init_wc.zero_()
                nn.init.xavier_normal_(init_wc)

                init_wc.scatter_(1,y,1)
                init_wc = init_wc.transpose(0,1)
        else:
            # sample from input samples to initialize protos and wc
            init_proto = []
            for i in range(n_classes):
                perm = torch.randperm(n_samples)
                idx = perm[:n_proto_per_class]
                init_proto.append(samples.view(n_class,n_samples, -1)[i, idx, :])
            init_proto = torch.stack(init_proto, dim=0).view(n_class*n_proto_per_class, -1)
            init_proto +=  nn.init.xavier_normal_(torch.zeros_like(init_proto))
            # initialize wc
            if not concat:
                init_wc = torch.FloatTensor(n_proto, n_class)
                init_wc.zero_()
                y =  torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_proto_per_class, 1).reshape(n_class*n_proto_per_class,1).long()
                nn.init.xavier_normal_(init_wc)
                init_wc.scatter_(1,y,1)
                init_wc = init_wc.transpose(0,1)

        # add hidden layer
        if use_hidden:
            self.hidden = nn.Linear(n_dim, n_dim)
            init_hidden =  nn.init.xavier_normal_(torch.zeros_like(self.hidden.weight))
            init_hidden += torch.eye(n_dim, n_dim)
            self.hidden.weight = nn.Parameter(init_hidden)
        else:
            self.hidden = None

        # then prototype coordinates
        # shape n_protos x n_dim
        self.prototypes = nn.Parameter(init_proto)

        if concat:
            self.output = nn.Linear(n_dim*2, n_classes, bias=False)
            nn.init.xavier_normal_(self.output.weight)
        else:
            self.output = nn.Linear(n_proto, n_classes, bias=False)
            self.output.weight = nn.Parameter(init_wc)

    def transform(self, samples, flat=False):
        if flat:
            samples = F.relu(self.hidden(samples))
        else:
            n_class = samples.size(0)
            n_samples = samples.size(1)

            samples = F.relu(self.hidden(samples.view(n_class*n_samples, -1)).view(n_class, n_samples, -1))
        return samples

    def forward(self, samples):
        n_class = samples.size(0)
        n_samples = samples.size(1)
        if samples.ndimension() != 3:
            samples = samples.view(n_class, n_samples, -1)

        if self.hidden is not None:
            samples = self.transform(samples)

        dists = euclidean_dist(samples.view(n_class*n_samples, -1), self.prototypes)

        qj = F.softmax(-dists/self.temperature, dim=1)

        if self.concat:
            # x tilde = qj * pj
            x_tilde = torch.mm(qj, self.prototypes)
            x = samples.view(n_class*n_samples, -1)
            cj = self.output(torch.cat((x, x_tilde), dim=-1))
        else:
            cj = self.output(qj)
        return cj

    def proto_proba(self):
        return self.predict_proba(self.prototypes.view(self.n_classes,
                                                       self.n_proto_per_class, -1)).reshape(self.n_proto,-1)

    def proto_class(self):
        return self.predict(self.prototypes.view(self.n_classes,
                                                 self.n_proto_per_class, -1)).reshape(-1)

    def predict_log_proba(self, samples):
        n_class = samples.size(0)
        n_samples = samples.size(1)

        cj = self.forward(samples)

        log_p_y = F.log_softmax(cj, dim=1).view(n_class, n_samples, -1)
        return log_p_y

    def predict(self, samples):
        n_class = samples.size(0)
        n_samples = samples.size(1)

        cj = self.forward(samples)

        log_p_y = F.log_softmax(cj, dim=1).view(n_class, n_samples, -1)
        _, y_hat = log_p_y.max(2)
        return y_hat

    def predict_proba(self, samples):
        n_class = samples.size(0)
        n_samples = samples.size(1)

        cj = self.forward(samples)

        p_y = F.softmax(cj, dim=1).view(n_class, n_samples, -1)
        return p_y

    def score(self, samples):
        n_class = samples.size(0)
        n_samples = samples.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_samples, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        if samples.is_cuda:
            target_inds = target_inds.cuda()

        y_hat = self.predict(samples)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        return acc_val

    def loss(self, samples):
        n_class = samples.size(0)
        n_samples = samples.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_samples, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if samples.is_cuda:
            target_inds = target_inds.cuda()

        log_p_y = self.predict_log_proba(samples)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        return loss_val

    def fit(self, samples, lr=1e-1, max_epochs=1000):
        optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=1e-4)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        for e in range (0, max_epochs):
            optimizer.zero_grad()
            loss = self.loss(samples)
            loss.backward(retain_graph=True)
            optimizer.step()
            #scheduler.step(loss)
        return self


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()

        self.encoder = encoder
        self.multiproto = True


    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        zs = z[:n_class*n_support].view(n_class, n_support, z_dim)
        zq = z[n_class*n_support:]

        if self.multiproto:
            n_proto=2*n_class
            support_model = Multiproto(zs,
                                       z_dim, n_proto, n_class,
                                       init='random')
            if zs.is_cuda:
                support_model = support_model.cuda()
            support_model.fit(zs, lr=1e-1,
                              max_epochs=200)
            zq = zq.view(n_class, n_query, z_dim)
            log_p_y = support_model.predict_log_proba(zq)
        else:
            z_proto = zs.mean(1)

            dists = euclidean_dist(zq, z_proto)

            log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return Protonet(encoder)
