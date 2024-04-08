import torch
import torch.nn as nn
import numpy as np
###################get dir to load pretrained model
import sys
from os.path import dirname, abspath, join, exists
from dlib.configure import constants
from dlib.utils.shared import count_params
from dlib.div_classifiers.parts.nc_ccam import returnCCAM_

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)
IMG_NET_W_FD = join(root_dir, constants.FOLDER_PRETRAINED_IMAGENET)
###################get dir to load pretrained model



class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // 8, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.mix_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.mix_conv.weight.data.fill_(0.0)
        # self.mix_conv.bias.data.fill_(0.0)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(in_dim)
        nn.init.constant_(self.bn.weight, 0.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.relu = nn.ReLU()

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)
        # out = self.relu(self.bn(self.mix_conv(out)))
        out = (self.bn(self.mix_conv(out)))
        out = out + x
        # print(self.gamma)
        return out

class Self_Attn_low(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn_low, self).__init__()
        self.channel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.mix_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.mix_conv.weight.data.fill_(0.0)
        # self.mix_conv.bias.data.fill_(0.0)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(in_dim)
        nn.init.constant_(self.bn.weight, 0.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.relu = nn.ReLU()

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)
        # out = self.relu(self.bn(self.mix_conv(out)))
        out = (self.bn(self.mix_conv(out)))
        out = out + x
        # print(self.gamma)
        return out

class CNN(nn.Module):
    """
    Simple CNN NETWORK
    """
    def __init__(self, encoder_name=constants.NLCCAM_VGG16, encoder_weights=constants.IMAGENET, pretrain=True, num_classes=None):
        super(CNN, self).__init__()
        assert num_classes is not None, "num_classes is None"
        self.num_classes = num_classes
        
        self.conv = nn.Sequential(
	    # 3 x 128 x 128
	    nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(0.2),
	    # nn.BatchNorm2d(64),
        nn.MaxPool2d(2,2),
        # Self_Attn(64),
	    # 32 x 128 x 128
	    nn.Conv2d(64, 128, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(0.2),
	    # nn.BatchNorm2d(128),
        nn.MaxPool2d(2,2),
        Self_Attn(128),
	    # 64 x 128 x 128
	    nn.Conv2d(128, 256, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(256, 256, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(256, 256, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.MaxPool2d(2,2),
        Self_Attn(256),
	    # 64 x 64 x 64
        nn.Conv2d(256, 512, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.MaxPool2d(2, 2),
		Self_Attn(512),
	    # 128 x 64 x 64
        nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU(0.2),
        # nn.MaxPool2d(2, 2),
        Self_Attn(512)
	    )
	    # 256 x 32 x 32
        self.avg_pool = nn.AvgPool2d(14)
        # 256 x 1 x 1

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
        )

        if pretrain:
            self._init_load_pretrained_w()
            # self.weights_init()
            print('pretrained weight load complete..')
            
        #########################
        self.encoder_name: str = encoder_name
        self.task: str = constants.STD_CL
        self.classification_head = None
        
        self.name = "{}".format(self.encoder_name)
        self.encoder_weights = encoder_weights
        self.cams = None

        self.method = constants.METHOD_NLCCAM
        self.arch = constants.NLCCAMCLASSIFIER
        
        if self.encoder_weights == constants.IMAGENET:
            self._init_load_pretrained_w()


    def _init_load_pretrained_w(self):
        k = 0
        pretrained_imgnet_path = join(IMG_NET_W_FD, 'nl_ccam_vgg16_imgnet.pth')
        if not exists(pretrained_imgnet_path):
            raise ValueError(f"No pretrained model found Download the model from the orginal repo https://github.com/Yangseung/NL-CCAM and place it in the folder{IMG_NET_W_FD} with name nl_ccam_vgg16_imgnet.pth")
        pretrained_weights = torch.load(pretrained_imgnet_path)
        pretrained_list = pretrained_weights.keys()
        for i, layer_name in enumerate(pretrained_list):
            layer_num = int(layer_name.split('.')[1])
            layer_group = layer_name.split('.')[0]
            layer_type = layer_name.split('.')[-1]
            if layer_num >= 10:
                layer_num = layer_num + 1
            if layer_num >= 17:
                layer_num = layer_num + 1
            if layer_num >= 24:
                layer_num = layer_num + 1

            if layer_group != "features":
                break

            if layer_type == 'weight':
                assert self.conv[layer_num].weight.data.size() == pretrained_weights[
                    layer_name].size(), "size error!"
                self.conv[layer_num].weight.data = pretrained_weights[layer_name]
            else:  # layer type == 'bias'
                assert self.conv[layer_num].bias.size() == pretrained_weights[layer_name].size(), "size error!"
                self.conv[layer_num].bias.data = pretrained_weights[layer_name]

    def get_info_nbr_params(self) -> str:
            totaln = count_params(self)
            cl_head_n = 0

            info = self.__str__() + ' \n NBR-PARAMS: \n'

            info += '\tEncoder [{}]: {}. \n'.format(self.name, totaln)
            info += '\tTotal: {}. \n'.format(totaln)

            return info
        
    def forward(self, x, labels=None):
        features = self.conv(x)
        feature_map = features
        flatten = self.avg_pool(features).view(features.size(0), -1)
        output = self.classifier(flatten)
        # self.pred = output
        # return output, features
        if labels is not None:
            assert len(x) == 1, "batch size > 1 not supported for computing CCAM"
            logit1, idx = output.data.sort(1, True)
            h_x = logit1.data
            # self.cams = returnCCAM_(feature_conv=feature_map.detach().cpu().numpy(), 
            #                         weight_softmax=np.squeeze(list(self.parameters())[-2].clone().detach().cpu().numpy()),
            #                         class_idx=labels,
            #                         reverse_idx=idx.cpu().numpy(),
            #                         h_x=h_x[0],
            #                         j=0,
            #                         threshold=0.12,
            #                         thr=100,
            #                         function='quadratic',
            #                         num_classes=self.num_classes)
            # self.cams = torch.tensor(self.cams).unsqueeze(0)
            self.cams = returnCCAM_(feature_conv=feature_map.detach(), 
                                    weight_softmax=list(self.parameters())[-2].clone().detach(),
                                    class_idx=labels,
                                    reverse_idx=idx.cpu().numpy(),
                                    h_x=h_x,
                                    j=0,
                                    threshold=0.12,
                                    thr=100,
                                    function='quadratic',
                                    num_classes=self.num_classes).unsqueeze(0)
        
        return output

    # def get_cam(self):
    #     return self.feature_map, self.pred

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# def get_net(encoder_weights=constants.IMAGENET,
#             num_classes=None):
#     assert num_classes is not None, "num_classes is None"
#     net = CNN(encoder_weights=encoder_weights,
#         encoder_name=constants.METHOD_SCM,
#         num_classes=num_classes)
#     # net.apply(weight_init)
#     print("INIT NETWORK")
#     return net

# def load_net(encoder_weights=constants.IMAGENET,
#              num_classes=None, model_name=None):
#     assert model_name is not None, "model_name is None"
#     net = CNN(encoder_weights=encoder_weights,
#         encoder_name=constants.METHOD_SCM,
#         num_classes=num_classes)
#     net.load_state_dict(torch.load(model_name))
#     return net

def check(encoder_weights=constants.IMAGENET):
    input = torch.rand(1,3,224,224)
    net = CNN(encoder_weights=encoder_weights,
        encoder_name=constants.METHOD_NLCCAM, num_classes=200)
    output = net(input)
    print(output)

def test_NC_CCAM():
    import datetime as dt

    import torch.nn.functional as F

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    txt = {0: 'torch', 1: 'wsol'}

    pretrained = False
    num_classes = 200

    model = CNN(encoder_name=constants.METHOD_NLCCAM, encoder_weights=constants.IMAGENET, 
                num_classes=num_classes)

    model.eval()
    model.to(device)
    print(model.get_info_nbr_params())
    bsize = 1
    h, w = 224, 224
    x = torch.rand(bsize, 3, 224, 224).to(device)
    labels = torch.zeros((bsize,), dtype=torch.long)
    model(x)
    # print(f'logits shape : {logits.shape} x : {x.shape} '
    #       f'classes : {num_classes}')

    t0 = dt.datetime.now()
    model(x, labels=labels)
    cams = model.cams
    print(cams.shape, x.shape)
    # if cams.shape != (1, h, w):
    #     tx = dt.datetime.now()
    #     full_cam = F.interpolate(
    #         input=cams.unsqueeze(0),
    #         size=[h, w],
    #         mode='bilinear',
    #         align_corners=True)
    # print(x.shape, cams.shape)
    print('time: {}'.format(dt.datetime.now() - t0))


if __name__ == "__main__":
    test_NC_CCAM()
    check()
    