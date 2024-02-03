import copy
import torch
from torch import nn
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50,resnet101
from convs.ucir_cifar_resnet import resnet32 as cosine_resnet32
from convs.ucir_resnet import resnet18 as cosine_resnet18
from convs.ucir_resnet import resnet34 as cosine_resnet34
from convs.ucir_resnet import resnet50 as cosine_resnet50
from convs.linears import SimpleLinear
from convs.modified_represnet import resnet18_rep
from convs.resnet_cbam import resnet18_cbam,resnet34_cbam,resnet50_cbam


def get_convnet(args, pretrained=True):
    name = args["convnet_type"].lower()
    if name == "resnet32":
        return resnet32()
    elif name == "resnet18":
        return resnet18(pretrained=pretrained,args=args)
    elif name == "resnet34":
        return resnet34(pretrained=pretrained,args=args)
    elif name == "resnet50":
        return resnet50(pretrained=pretrained,args=args)
    elif name == "resnet101":
        return resnet101(pretrained=pretrained,args=args)
    elif name == "cosine_resnet18":
        return cosine_resnet18(pretrained=pretrained,args=args)
    elif name == "cosine_resnet32":
        return cosine_resnet32()
    elif name == "cosine_resnet34":
        return cosine_resnet34(pretrained=pretrained,args=args)
    elif name == "cosine_resnet50":
        return cosine_resnet50(pretrained=pretrained,args=args)
    elif name == "resnet18_rep":
        return resnet18_rep(pretrained=pretrained,args=args)
    elif name == "resnet18_cbam":
        return resnet18_cbam(pretrained=pretrained,args=args)
    elif name == "resnet34_cbam":
        return resnet34_cbam(pretrained=pretrained,args=args)
    elif name == "resnet50_cbam":
        return resnet50_cbam(pretrained=pretrained,args=args)
    elif name == "resnent_hi":
        return resnet50_cbam(pretrained=pretrained,args=args)
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained=True)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class DERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc1 = None # L1
        self.fc2 = None # L2
        self.fc3 = None # L3
        self.softmax_reg1 =None # L1
        self.softmax_reg2 =None # L2
        self.softmax_reg3 =None # L3
        
        self.aux_fc = None
        self.task_sizes = []
        self.args = args
        
    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out = self.fc3(features)

        out1 = self.fc1(features)["logits"] # L1
        out2 = self.fc2(features)["logits"] # L2
        out3 = self.fc3(features)["logits"] # L3
        
        level_1 = self.softmax_reg1(out1)
        level_2 = self.softmax_reg2(torch.cat((level_1, out2), dim=1))
        level_3 = self.softmax_reg3(torch.cat((level_2, out3), dim=1))
       
        aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]
        out.update({"aux_logits": aux_logits, "features": features})

        return out, level_1, level_2, level_3

    def update_fc(self, nb_classes, L2_total_classes, L1_total_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args))
        else:
            self.convnets.append(get_convnet(self.args))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim

        fc1 = self.generate_fc(self.feature_dim, L1_total_classes)
        fc2 = self.generate_fc(self.feature_dim, L2_total_classes)
        fc3 = self.generate_fc(self.feature_dim, nb_classes)

        reg1 = nn.Linear(L1_total_classes, L1_total_classes) # 8, 8
        reg2 = nn.Linear(L1_total_classes + L2_total_classes, L2_total_classes) # 20, 12
        reg3 = nn.Linear(L2_total_classes + nb_classes, nb_classes) # 28, 16

        if self.fc3 is not None:
            nb_output = self.fc3.out_features
            weight = copy.deepcopy(self.fc3.weight.data)
            bias = copy.deepcopy(self.fc3.bias.data)
            fc3.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc3.bias.data[:nb_output] = bias

        del self.fc3
        self.fc3 = fc3
        del self.fc2
        self.fc2 = fc2
        del self.fc1
        self.fc1 = fc1
        
        del self.softmax_reg1
        self.softmax_reg1 = reg1
        del self.softmax_reg2
        self.softmax_reg2 = reg2
        del self.softmax_reg3
        self.softmax_reg3 = reg3
        
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)
 
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc3.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc3.weight.data[-increment:, :] *= gamma


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )

