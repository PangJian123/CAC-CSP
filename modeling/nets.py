import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class Discriminator(nn.Module):
    # input 2048,dim=1024
    def __init__(self, input_dim, dim, out_dim):
        super(Discriminator, self).__init__()
        self.output_dim = out_dim
        self.model = []
        # Here I change the stride to 2.
        self.model += [nn.Linear(input_dim, dim)]
        self.model += [nn.Linear(dim, dim//2)]
        dim = dim//2
        for i in range(6):
            self.model += [nn.Linear(dim, dim//2)]
            dim //= 2 # dim=32
        # self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [nn.Linear(dim, 8)]
        self.model += [nn.Linear(8, 1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Discriminator_conv(nn.Module):
    # input 2048
    def __init__(self, embed_dim=2048):
        super(Discriminator_conv, self).__init__()
        self.layers = nn.ModuleList()
        hidden_dim = embed_dim//2 # 1024
        first_block = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)

        )
        self.layers.append(first_block)
        # self.gap = nn.AdaptiveAvgPool1d(512)
        # self.layers.append(self.gap)
        for layer_index in range(6):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim//2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim//2),
                nn.ReLU(inplace=True)

            )
            hidden_dim = hidden_dim // 2 # 512,256,128,64,32,16
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_dim, 1)
        self.layers.append(self.Liner)

    def forward(self, x):
        x = x.unsqueeze(2)
        # for y in (self.layers):
        #     x = y(x)
        x = self.layers[0](x) # (input+2*padding-kernel_size)/stride +1
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = self.layers[4](x)
        x = self.layers[5](x)
        x = self.layers[6](x)
        x = x.squeeze(2)
        x = self.layers[7](x)
        return x    #[batch,1]



class Cam_Encoder(nn.Module):
    # input 2048,dim=1024
    def __init__(self, input_dim, dim, out_dim):
        super(Cam_Encoder, self).__init__()
        self.output_dim = out_dim
        self.model = []
        # Here I change the stride to 2.
        self.model += [nn.Linear(input_dim, dim )]
        self.model += [nn.Linear(dim, dim )]
        for i in range(5):
            self.model += [nn.Linear(dim, dim//2)]
            dim //= 2 # dim=32
        # self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [nn.Linear(dim, self.output_dim)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
class Fusion_net(nn.Module):
    def __init__(self, embed_dim):
        super(Fusion_net, self).__init__()
        self.layers = nn.ModuleList()
        for layer_index in range(3):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim // 2, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(embed_dim // 2)
            )
            embed_dim = embed_dim // 2  # 4096,2048,1024
            self.layers.append(conv_block)
        for layer_index in range(1):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim * 2, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(embed_dim * 2)
            )
            embed_dim = embed_dim * 2  # 2048
            self.layers.append(conv_block)
        self.Liner = nn.Linear(embed_dim, 2048)
        self.layers.append(self.Liner)  # 6层

    def forward(self, x):
        x = x.unsqueeze(2)
        for i in range(4):
            x = self.layers[i](x)
        x = x.squeeze(2)
        x = self.layers[4](x)
        return x

class Pre_fusion(nn.Module):
    def __init__(self, embed_dim):  # input2048
        super(Pre_fusion, self).__init__()
        self.layers = nn.ModuleList()
        for layer_index in range(2):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim // 2, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(embed_dim // 2)
            )
            embed_dim = embed_dim // 2  # (2048,1024), (1024,512)
            self.layers.append(conv_block)
        for layer_index in range(2):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim * 2, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(embed_dim * 2)
            )
            embed_dim = embed_dim * 2
            self.layers.append(conv_block)
        self.Liner = nn.Linear(embed_dim, 2048)
        self.layers.append(self.Liner)  # 6层

    def forward(self, x):
        x = x.unsqueeze(2)
        for i in range(4):
            x = self.layers[i](x)
        x = x.squeeze(2)
        x = self.layers[4](x)
        return x


class Cam_Encoder_conv(nn.Module): #
    # input 2048,dim=1024
    def __init__(self, num_class, embed_dim=2048):
        super(Cam_Encoder_conv, self).__init__()
        self.layers = nn.ModuleList()
        num_class = num_class
        hidden_dim = embed_dim//2 # 1024
        first_block = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim)
        )
        self.layers.append(first_block)
        # self.gap = nn.AdaptiveAvgPool1d(1)
        # self.layers.append(self.gap)
        for layer_index in range(3):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim//2, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim//2)
            )
            hidden_dim = hidden_dim // 2  # 512,256,128,64,32
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_dim, 64)
        self.layers.append(self.Liner)
        self.batchnorm1d = nn.BatchNorm1d(64)
        self.batchnorm1d.bias.requires_grad_(False)
        self.batchnorm1d.apply(weights_init_kaiming)
        self.layers.append(self.batchnorm1d)
        self.Liner = nn.Linear(64, num_class, bias=False)
        self.layers.append(self.Liner)


    def forward(self, x):
        x = x.unsqueeze(2)
        # for y in (self.layers):
        #     x = y(x)
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = x.squeeze(2)
        x = self.layers[4](x)
        x = self.layers[5](x)
        x = self.layers[6](x)
        return x    # [batch, num_classes]


# class Cam_Encoder_conv(nn.Module):
#     # input 2048,dim=1024
#     def __init__(self, num_class, embed_dim=2048):
#         super(Cam_Encoder_conv, self).__init__()
#         self.layers = nn.ModuleList()
#         num_class = num_class
#         hidden_dim = embed_dim//2 # 1024
#         first_block = nn.Sequential(
#             nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(hidden_dim)
#         )
#         self.layers.append(first_block)
#         # self.gap = nn.AdaptiveAvgPool1d(1)
#         # self.layers.append(self.gap)
#         for layer_index in range(5):
#             conv_block = nn.Sequential(
#                 nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim//2, kernel_size=3, stride=2, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm1d(hidden_dim//2)
#             )
#             hidden_dim = hidden_dim // 2  # 512,256,128,64,32
#             self.layers.append(conv_block)
#         self.Liner = nn.Linear(hidden_dim, 64)
#         self.layers.append(self.Liner)
#         self.Liner = nn.Linear(64, num_class, bias=False)
#         self.layers.append(self.Liner)
#
#
#     def forward(self, x):
#         x = x.unsqueeze(2)
#         # for y in (self.layers):
#         #     x = y(x)
#         x = self.layers[0](x)
#         x = self.layers[1](x)
#         x = self.layers[2](x)
#         x = self.layers[3](x)
#         x = self.layers[4](x)
#         x = self.layers[5](x)
#         x = x.squeeze(2)
#         x = self.layers[6](x)
#         x = self.layers[7](x)
#         return x    # [batch, num_classes]




class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1, fp16 = False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim, fp16 = fp16)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x




# ----------------------------------------init_weights----------------------------------- #
def init_weights(net):
    net.apply(weights_init_normal)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname== 'Conv2dBlock':
        a = True
    else:
        if classname.find('Conv') != -1:
            init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# ----------------------------------------optimizer------------------------------------ #

def make_optimizer(cfg, model, fix=False):
    params = []
    for key, value in model.named_parameters():
        if ('classifier_1' in key) and (fix == True):   # 'module.classifier_1.weight'
            continue
        if 'classifier_2' in key and (fix == True):
            continue
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    # if cfg.OPTIMIZER_NAME == 'SGD':
    #     optimizer = getattr(torch.optim, cfg.OPTIMIZER_NAME)(params, momentum=cfg.MOMENTUM)
    # else:
        optimizer = getattr(torch.optim, "Adam")(params)
    return optimizer


def get_scheduler(optimizer, args, iterations=-1): # args.step_size, args.gamma,  args.lr_policy
    if args.MODEL.LR_POLICY=='step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.MODEL.LR_STEP_SIZE,
                                        gamma=0.1, last_epoch=iterations)
    elif args.MODEL.LR_POLICY=='multistep':
        step = args.MODEL.LR_STEP_SIZE
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             # milestones=[30, 50, 70, 100, 130],
                                             milestones=[
                                                         # step,  # 40,80
                                                         # step + step // 2,
                                                         step + step // 2 + step // 2,
                                                         ],
                                             gamma=0.1, last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.MODEL.LR_POLICY)
    return scheduler



# ------------------------------------------tools------------------------------#
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
