import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from collections import OrderedDict

"""
We support the following models:
    - AlexNet model customized for CIFAR-10 (AlexCifarNet) with 1756426 parameters
    - LeNet model customized for MNIST with 61706 parameters
    - Further ResNet models
    - Further Vgg models
"""


# AlexNet model customized for CIFAR-10 with 1756426 parameters
class AlexCifarNet(nn.Module):
    supported_dims = {32}

    def __init__(self):
        super(AlexCifarNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), 4096)
        out = self.classifier(out)
        return out


# LeNet model customized for MNIST with 61706 parameters
class LeNet(nn.Module):
    supported_dims = {28}

    def __init__(self, num_classes=10, in_channels=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)  # 6 x 28 x 28
        out = F.max_pool2d(out, 2)  # 6 x 14 x 14
        out = F.relu(self.conv2(out), inplace=True)  # 16 x 7 x 7
        out = F.max_pool2d(out, 2)   # 16 x 5 x 5
        out = out.view(out.size(0), -1)  # 16 x 5 x 5
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)

        return out


# Further ResNet models
def generate_resnet(num_classes=10, in_channels=1, model_name="ResNet18"):
    if model_name == "ResNet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "ResNet34":
        model = models.resnet34(pretrained=True)
    elif model_name == "ResNet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "ResNet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "ResNet152":
        model = models.resnet152(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)

    return model


# Further Vgg models
def generate_vgg(num_classes=10, in_channels=1, model_name="vgg11"):
    if model_name == "VGG11":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG11_bn":
        model = models.vgg11_bn(pretrained=True)
    elif model_name == "VGG13":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG13_bn":
        model = models.vgg11_bn(pretrained=True)
    elif model_name == "VGG16":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG16_bn":
        model = models.vgg11_bn(pretrained=True)
    elif model_name == "VGG19":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG19_bn":
        model = models.vgg11_bn(pretrained=True)

    # first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    # first_conv_layer.extend(list(model.features))
    # model.features = nn.Sequential(*first_conv_layer)
    # model.conv1 = nn.Conv2d(num_classes, 64, 7, stride=2, padding=3, bias=False)

    fc_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(fc_features, num_classes)

    return model


class CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(CNN, self).__init__()

        self.fp_con1 = nn.Sequential(OrderedDict([
            ('con0', nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)),
            ('relu0', nn.ReLU(inplace=True)),
            ]))

        self.ternary_con2 = nn.Sequential(OrderedDict([
            # Conv Layer block 1
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),

            # Conv Layer block 2
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            # nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            ('conv3', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False)),
            ('norm4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        self.fp_fc = nn.Linear(4096, num_classes, bias = False)

    def forward(self, x):
        x = self.fp_con1(x)
        x = self.ternary_con2(x)
        x = x.view(x.size(0), -1)
        x = self.fp_fc(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    
    
if __name__ == "__main__":
    model_name_list = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    for model_name in model_name_list:
        model = generate_resnet(num_classes=10, in_channels=1, model_name=model_name)
        # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        # param_len = sum([np.prod(p.size()) for p in model_parameters])
        param_len = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of model parameters of %s :' % model_name, ' %d ' % param_len)
        
        # another way to calculate the model size
        size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_024**2
        print(f"Model size: {size:.3f} MB")

        
"""
we try to support these models related to quantum computing:
- QCNN
- UNet-Q
- (QSVM)
"""

# from braket.circuits import Circuit

# class CirModel(nn.Module):

#     def __init__(self, n_qubits):
#         super().__init__()
#         self.weight = nn.Parameter(nn.init.uniform_(torch.empty(6*n_qubits), a=0.0, b=2 * np.pi))
#         self.n_qubits = n_qubits

#     def forward(self, x):
#         M_list=[]
#         for i in range(self.n_qubits):
#             cirx = Circuit()
#             cirx.z(i)
#             for j in range(self.n_qubits):
#                 if j == i:
#                     pass
#                 else:
#                     cirx.i(j)
#             mm = torch.tensor(cirx.to_unitary()).type(dtype=torch.complex64)
#             M_list.append(mm)
        
#         """     
#         cir1 = Circuit()
#         cir1.z(0)
#         for i in range(1, self.n_qubits):
#             cir1.i(i)
#         M = torch.tensor(cir1.to_unitary()).type(dtype=torch.complex64)
#         """

#         w = self.weight
#         cir2 = Circuit()
#         for which_q in range(0, self.n_qubits):
#             cir2.ry(which_q, w[0+6*which_q])
#             cir2.rz(which_q, w[1+6*which_q])
#             cir2.ry(which_q, w[2+6*which_q])
#             if which_q < (self.n_qubits-1):
#                 cir2.cnot(which_q, which_q + 1)
#             else:
#                 cir2.cnot(which_q, 0)
#             cir2.ry(which_q, w[3+6*which_q])
#             cir2.rz(which_q, w[4+6*which_q])
#             cir2.ry(which_q, w[5+6*which_q])
#         unitary = torch.tensor(cir2.to_unitary(), requires_grad = True).type(dtype=torch.complex64)
#         #print(unitary.shape)
        
#         out = unitary @ x.T
#         res_list=[]
#         for k in range(self.n_qubits):
#             M = M_list[k]
#             if x.shape[0] == 1:
#                 res = (out.conj().T @ M @ out).real
#             else:
#                 res = (out.conj().T @ M @ out).diag().real
#                 res = res.reshape(-1,1)
#             res_list.append(res)
#         res_list = torch.tensor(torch.cat(res_list,dim=1), requires_grad = True).type(dtype=torch.float64)
#         return res_list
    
# """
# CNN+QuMLP  
# """
# class QNet(nn.Module):
#     def __init__(self):
#         super(QNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 64, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = CirModel(10)
#         #self.fc2 = nn.Linear(50, 10)
 
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         #print('x-conv1')
#         #print(x.shape)
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 1024)+0j
#         x = nn.functional.normalize(x)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         #x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

"""
up_sampling/down_sampling
"""
# class convQ_up(nn.Module):
#     def __init__(self, input=9):
#         super().__init__()
#         #self.weight = nn.Parameter(nn.init.uniform_(torch.empty(6*n_qubits), alpha=0., beta=2 * np.pi))
#         self.register = input # 线路比特数目
#         self.theta = torch.empty(6 * input + 3)
#         self.weight = nn.Parameter(nn.init.uniform_(self.theta, a = -1 * np.pi , b = 1. * np.pi))
#         #self.trash_dim  = trash_dim # 压缩比特数
    
#     def forward(self, input):
#         #input = self.up_samplingQ(input, y)
#         w = self.weight
#         cir = Circuit()
#         for which_R in range(0, self.register):
#             cir.rx(which_R, w[0+6*which_R])
#             cir.ry(which_R, w[1+6*which_R])
#             cir.rz(which_R, w[2+6*which_R])
            
#             cir.rx(which_R, w[6*self.register])
#             cir.ry(which_R, w[6*self.register+1])
#             cir.rz(which_R, w[6*self.register+2])
            
#             cir.rx(which_R, w[3+6*which_R])
#             cir.ry(which_R, w[4+6*which_R])
#             cir.rz(which_R, w[5+6*which_R])
            
#             if which_R < (self.register-1):
#                 cir.cnot(which_R, which_R + 1)
#             else:
#                 cir.cnot(which_R, 0)
            
#         Evolutian_op = torch.tensor(cir.to_unitary(), requires_grad = True).type(dtype=torch.complex64)
#         #print(***.shape)
#         out = Evolutian_op @ input @ Evolutian_op.conj().T
#         return out

# class deconvQ(nn.Module):
#     def __init__(self, input=8):
#         super().__init__()
#         #self.weight = nn.Parameter(nn.init.uniform_(torch.empty(6*n_qubits), alpha=0., beta=2 * np.pi))
#         self.register = input # 线路比特数目
#         self.theta = torch.empty(6 * input + 3)
#         self.weight = nn.Parameter(nn.init.uniform_(self.theta, a = -1 * np.pi , b = 1. * np.pi))
#         #self.trash_dim  = trash_dim # 压缩比特数
    
#     def forward(self, input):
#         w = self.weight
#         cir = Circuit()
#         for which_R in range(0, self.register):
#             cir.rx(which_R, w[0+6*which_R])
#             cir.ry(which_R, w[1+6*which_R])
#             cir.rz(which_R, w[2+6*which_R])
            
#             cir.rx(which_R, w[6*self.register])
#             cir.ry(which_R, w[6*self.register+1])
#             cir.rz(which_R, w[6*self.register+2])
            
#             cir.rx(which_R, w[3+6*which_R])
#             cir.ry(which_R, w[4+6*which_R])
#             cir.rz(which_R, w[5+6*which_R])
            
#             if which_R < (self.register-1):
#                 cir.cnot(which_R, which_R + 1)
#             else:
#                 cir.cnot(which_R, 0)
            
#         Evolutian_op = torch.tensor(cir.to_unitary(), requires_grad = True).type(dtype=torch.complex64)
#         #print(***.shape)
#         out = Evolutian_op @ input @ Evolutian_op.conj().T
#         return out
    
#     def down_samplingQ(self, x, y):
#         #x :: input_rho
#         #y :: trash dim
#         return ptrace(x, self.register - y, y)

"""
UNet Quantum Version 1.
"""
# class unetQ(nn.Module):
#     def __init__(self):
#         super(unetQ,self).__init__()
#         self.l1_convQ = convQ_up(input=9)
#         self.l2_convQ = convQ_up(input=8)
#         self.l3_convQ = convQ_up(input=7)
#         self.l4_convQ = convQ_up(input=6)
        
#         self.l5_convQ = convQ_up(input=4)
        
#         self.l6_convQ = convQ_up(input=6)
#         self.l7_convQ = convQ_up(input=7)
#         self.l8_convQ = convQ_up(input=8)
#         self.l9_convQ = convQ_up(input=9)
        
        
#         self.deconvQ1 = deconvQ(input=9)
#         self.deconvQ2 = deconvQ(input=8)
#         self.deconvQ3 = deconvQ(input=7)
#         self.deconvQ4 = deconvQ(input=6)
        
#         self.deconvQ5 = deconvQ(input=6)
#         self.deconvQ6 = deconvQ(input=7)
#         self.deconvQ7 = deconvQ(input=8)
#         self.deconvQ8 = deconvQ(input=9)
        
        
        
#     def forward(self,rho):
#         rho = up_samplingQ(rho, identityQ)
#         #print("input_rho,",rho.shape)
#         convq1 = self.l1_convQ(rho)
#         #print("convq1-9,",convq1.shape)
#         dn_cq1 = self.deconvQ1(convq1)
        
#         dnkeep1 = self.deconvQ1.down_samplingQ(dn_cq1, 2)
#         dntrash1 = self.deconvQ1.down_samplingQ(dn_cq1, 7)
        
#         rho2 = up_samplingQ(dnkeep1, identityQ)
#         print("input_rho2,",rho2.shape, "Is Hermitian? ::", IsHermitian(rho2))
#         convq2 = self.l2_convQ(rho2)
#         dn_cq2 = self.deconvQ2(convq2)
        
#         dnkeep2 = self.deconvQ2.down_samplingQ(dn_cq2, 2)
#         dntrash2 = self.deconvQ2.down_samplingQ(dn_cq2, 6)
        
#         rho3 = up_samplingQ(dnkeep2, identityQ)
#         print("input_rho3,",rho2.shape, "Is Hermitian? ::", IsHermitian(rho3))
#         convq3 = self.l3_convQ(rho3)
#         dn_cq3 = self.deconvQ3(convq3)
        
#         dnkeep3 = self.deconvQ3.down_samplingQ(dn_cq3, 2)
#         dntrash3 = self.deconvQ3.down_samplingQ(dn_cq3, 5)
        
#         rho4 = up_samplingQ(dnkeep3, identityQ)
#         print("input_rho4,",rho2.shape, "Is Hermitian? ::", IsHermitian(rho4))
#         convq4 = self.l4_convQ(rho4)
#         dn_cq4 = self.deconvQ4(convq4)
        
#         dnkeep4 = self.deconvQ4.down_samplingQ(dn_cq4, 2)
#         dntrash4 = self.deconvQ4.down_samplingQ(dn_cq4, 4)
        
#         rho5 = dnkeep4
#         convq5 = self.l5_convQ(rho5)
        
        
#         conv_kron1 = up_samplingQ(convq5, dntrash4)
#         convtq1 = self.l6_convQ(conv_kron1)
#         dn_cq6 = self.deconvQ5(convtq1)
#         convq6 =self.deconvQ5.down_samplingQ(dn_cq6,1)
        
#         conv_kron2 = up_samplingQ(convq6, dntrash3)
#         convtq2 = self.l7_convQ(conv_kron2)
#         dn_cq7 = self.deconvQ6(convtq2)
#         convq7 = self.deconvQ6.down_samplingQ(dn_cq7,1)
        
#         conv_kron3 = up_samplingQ(convq7, dntrash2)
#         convtp3 = self.l8_convQ(conv_kron3)
#         dn_cq8 = self.deconvQ7(convtp3)
#         convq8 = self.deconvQ7.down_samplingQ(dn_cq8,1)
        
#         conv_kron4 = up_samplingQ(convq8, dntrash1)
#         convtp4 = self.l9_convQ(conv_kron4)
#         dn_cq9 = self.deconvQ8(convtp4)
#         convq9 = self.deconvQ8.down_samplingQ(dn_cq9,1)
        
#         outq = convq9
#         return outq

