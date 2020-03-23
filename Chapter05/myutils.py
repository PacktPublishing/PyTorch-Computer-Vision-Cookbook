
import torch
from torch import nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_model_config(path2file):
    cfg_file = open(path2file, 'r')
    lines = cfg_file.read().split('\n')

    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] 
    
    blocks_list = []
    for line in lines:
        # start of a new block
        if line.startswith('['): 
            blocks_list.append({})
            blocks_list[-1]['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            value = value.strip()
            blocks_list[-1][key.rstrip()] = value.strip()

    return blocks_list


def create_layers(blocks_list):
    hyperparams = blocks_list[0]
    channels_list = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    
    for layer_ind, layer_dict in enumerate(blocks_list[1:]):
        modules = nn.Sequential()
        
        if layer_dict["type"] == "convolutional":
            filters = int(layer_dict["filters"])
            kernel_size = int(layer_dict["size"])
            pad = (kernel_size - 1) // 2
            bn=layer_dict.get("batch_normalize",0)    
            
            
            conv2d= nn.Conv2d(
                        in_channels=channels_list[-1],
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=int(layer_dict["stride"]),
                        padding=pad,
                        bias=not bn)
            modules.add_module("conv_{0}".format(layer_ind), conv2d)
            
            if bn:
                bn_layer = nn.BatchNorm2d(filters,momentum=0.9, eps=1e-5)
                modules.add_module("batch_norm_{0}".format(layer_ind), bn_layer)
                
                
            if layer_dict["activation"] == "leaky":
                activn = nn.LeakyReLU(0.1)
                modules.add_module("leaky_{0}".format(layer_ind), activn)
                
        elif layer_dict["type"] == "upsample":
            stride = int(layer_dict["stride"])
            upsample = nn.Upsample(scale_factor = stride)
            modules.add_module("upsample_{}".format(layer_ind), upsample) 
            

        elif layer_dict["type"] == "shortcut":
            backwards=int(layer_dict["from"])
            filters = channels_list[1:][backwards]
            modules.add_module("shortcut_{}".format(layer_ind), EmptyLayer())
            
        elif layer_dict["type"] == "route":
            layers = [int(x) for x in layer_dict["layers"].split(",")]
            filters = sum([channels_list[1:][l] for l in layers])
            modules.add_module("route_{}".format(layer_ind), EmptyLayer())
            
        elif layer_dict["type"] == "yolo":
            anchors = [int(a) for a in layer_dict["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]

            mask = [int(m) for m in layer_dict["mask"].split(",")]
            
            anchors = [anchors[i] for i in mask]
            
            num_classes = int(layer_dict["classes"])
            img_size = int(hyperparams["height"])
            
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module("yolo_{}".format(layer_ind), yolo_layer)
            
        module_list.append(modules)       
        channels_list.append(filters)

    return hyperparams, module_list        



class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        
        
class YOLOLayer(nn.Module):

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.grid_size = 0 
        
        
    def forward(self, x_in):
        batch_size = x_in.size(0)
        grid_size = x_in.size(2)
        devide=x_in.device
        
        prediction=x_in.view(batch_size, self.num_anchors, 
                             self.num_classes + 5, grid_size, grid_size)
        prediction=prediction.permute(0, 1, 3, 4, 2)
        prediction=prediction.contiguous()
        
        obj_score = torch.sigmoid(prediction[..., 4]) 
        pred_cls = torch.sigmoid(prediction[..., 5:]) 
        
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x_in.is_cuda)
            
        pred_boxes=self.transform_outputs(prediction) 
        
        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4),
                obj_score.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.num_classes),
            ), -1,)
        return output        
    
    
        
    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        self.stride = self.img_dim / self.grid_size
        
        self.grid_x = torch.arange(grid_size, device=device).repeat(1, 1, grid_size, 1 ).type(torch.float32)
        self.grid_y = torch.arange(grid_size, device=device).repeat(1, 1, grid_size, 1).transpose(3, 2).type(torch.float32)
        
        scaled_anchors=[(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]
        self.scaled_anchors=torch.tensor(scaled_anchors,device=device)
        
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
        
        
        
    def transform_outputs(self,prediction):
        device=prediction.device
        x = torch.sigmoid(prediction[..., 0]) # Center x
        y = torch.sigmoid(prediction[..., 1]) # Center y
        w = prediction[..., 2] # Width
        h = prediction[..., 3] # Height

        pred_boxes = torch.zeros_like(prediction[..., :4]).to(device)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        
        return pred_boxes * self.stride             