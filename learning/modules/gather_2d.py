import torch
from visualization import Presenter

import torch.nn as nn
"""
Given a 4-D image (batch x channels x X x Y) and a 'list' of 2-D coordinates, extract the channel-vector for each
of the 2D coordinates and return as a 'list' of channel vectors
"""
class Gather2D(nn.Module):

    def __init__(self):
        super(Gather2D, self).__init__()

    def init_weights(self):
        pass

    def dbg_viz(self, image, coords_in_features):
        image = image.data.cpu()
        image[0, :, :] = 0.0
        image -= torch.min(image)
        image /= (torch.max(image) + 1e-9)
        for coord in coords_in_features:
            c = coord.long()
            x = c.data.item()
            y = c.data.item()
            image[0, x, y] = 1.0
        Presenter().show_image(image, "gather_dbg", torch=True, scale=2, waitkey=True)

    def forward(self, image, coords_in_features, axes=(2, 3)):

        # Get rid of the batch dimension
        # TODO Handle batch dimension properly
        if len(coords_in_features.size()) > 2:
            coords_in_features = coords_in_features[0]

        assert coords_in_features.data.type() == 'torch.LongTensor' or\
            coords_in_features.data.type() == 'torch.cuda.LongTensor'

        # TODO: Handle additional batch axis. Currently batch axis must be of dimension 1
        assert len(axes) == 2

        if False:
            self.dbg_viz(image[0], coords_in_features)

        # Gather the full feature maps for each of the 2 batches
        gather_x = coords_in_features[:, 0].contiguous().view([-1, 1, 1, 1])
        gather_y = coords_in_features[:, 1].contiguous().view([-1, 1, 1, 1])

        gather_img_x = gather_x.expand([-1, image.size(1), 1, image.size(3)])
        gather_img_y = gather_y.expand([-1, image.size(1), 1, 1])

        # Make enough
        img_size = list(image.size())
        img_size[0] = coords_in_features.size(0)
        image_in = image.expand(img_size)

        gather_img_x = gather_img_x.clamp(0, image_in.shape[2] - 1)
        gather_img_y = gather_img_y.clamp(0, image_in.shape[3] - 1)

        vec_y = torch.gather(image_in, 2, gather_img_x)
        vec = torch.gather(vec_y, 3, gather_img_y)
        vec = torch.squeeze(vec, 3)
        vec = torch.squeeze(vec, 2)

        return vec
