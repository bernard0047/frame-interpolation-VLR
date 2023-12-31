import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from model.loss import *
from model.warplayer import warp
import os
from config import *

    
class Model:
    def __init__(self, local_rank, use_perceptual_loss,device="cuda:0",use_style= False,freeze=False):
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH']
        self.net = multiscaletype(backbonetype(**backbonecfg),device, **multiscalecfg)
        self.name = MODEL_CONFIG['LOGNAME']
        self.use_perceptual_loss = use_perceptual_loss
        self.use_style = use_style
        self.device = device
        self.get_device()

        # train
        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)
        self.lap = LapLoss(use_perceptual_loss=self.use_perceptual_loss,use_style=self.use_style,device="cuda:0")
        if freeze:
           self.freeze_backbone()
        if local_rank != -1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank)
        
        

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def get_device(self):
        self.net.to(torch.device(self.device))

    def load_model(self, name=None, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }
        if rank <= 0 :
            if name is None:
                name = self.name
            # self.net.load_state_dict(convert(torch.load(f'infer_models/emavfi/{name}.pkl')))
            # self.net.load_state_dict(convert(torch.load('/home/arpitsah/Desktop/Fall-2023/VLR/project/frame-interpolation-VLR/baseline_ckpt/ours_small.pkl')))
            self.net.load_state_dict(convert(torch.load('/home/arpitsah/Desktop/Fall-2023/VLR/project/frame-interpolation-VLR/baseline_ckpt/ours_small_t.pkl')))
            # self.net.load_state_dict(convert(torch.load('/home/arpitsah/Desktop/Fall-2023/VLR/project/frame-interpolation-VLR/emavfi/ckpt/ours_small.pkl')))

            # self.net.load_state_dict(convert(torch.load(f'/home/xinyu/16824/project/frame-interpolation-VLR/emavfi/ckpt/ours_small_99_arpit.pkl')))
            # self.net.load_state_dict(convert(torch.load(f'/home/arpitsah/Desktop/Fall-2023/VLR/project/frame-interpolation-VLR/baseline_ckpt/ours_small_t.pkl', map_location=self.device)))

            
    def save_model(self, epoch,rank=0):
        if rank == 0:
            os.makedirs('ckpt',exist_ok = True)
            torch.save(self.net.state_dict(),f'ckpt/{self.name}_ours_small_t_300_{epoch}.pkl')

    @torch.no_grad()
    def hr_inference(self, img0, img1, TTA = False, down_scale = 1.0, timestep = 0.5, fast_TTA = False):
        '''
        Infer with down_scale flow
        Noting: return BxCxHxW
        '''
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)

            flow, mask = self.net.calculate_flow(imgs_down, timestep)

            flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
            mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)

            af, _ = self.net.feature_bone(img0, img1)
            pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
            return pred

        imgs = torch.cat((img0, img1), 1)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds = infer(input)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        if TTA == False:
            return infer(imgs)
        else:
            return (infer(imgs) + infer(imgs.flip(2).flip(3)).flip(2).flip(3)) / 2

    @torch.no_grad()
    def inference(self, img0, img1, TTA = False, timestep = 0.5, fast_TTA = False):
        imgs = torch.cat((img0, img1), 1)
        '''
        Noting: return BxCxHxW
        '''
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            _, _, _, preds = self.net(input, timestep=timestep)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        _, _, _, pred = self.net(imgs, timestep=timestep)
        if TTA == False:
            return pred
        else:
            _, _, _, pred2 = self.net(imgs.flip(2).flip(3), timestep=timestep)
            return (pred + pred2.flip(2).flip(3)) / 2

    @torch.no_grad()
    def multi_inference(self, img0, img1, TTA = False, down_scale = 1.0, time_list=[], fast_TTA = False):
        '''
        Run backbone once, get multi frames at different timesteps
        Noting: return a list of [CxHxW]
        '''
        assert len(time_list) > 0, 'Time_list should not be empty!'
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            af, mf = self.net.feature_bone(img0, img1)
            imgs_down = None
            if down_scale != 1.0:
                imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)
                afd, mfd = self.net.feature_bone(imgs_down[:, :3], imgs_down[:, 3:6])

            pred_list = []
            for timestep in time_list:
                if imgs_down is None:
                    flow, mask = self.net.calculate_flow(imgs, timestep, af, mf)
                else:
                    flow, mask = self.net.calculate_flow(imgs_down, timestep, afd, mfd)
                    flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
                    mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)
                
                pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
                pred_list.append(pred)

            return pred_list

        imgs = torch.cat((img0, img1), 1)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds_lst = infer(input)
            return [(preds_lst[i][0] + preds_lst[i][1].flip(1).flip(2))/2 for i in range(len(time_list))]

        preds = infer(imgs)
        if TTA is False:
            return [preds[i][0] for i in range(len(time_list))]
        else:
            flip_pred = infer(imgs.flip(2).flip(3))
            return [(preds[i][0] + flip_pred[i][0].flip(1).flip(2))/2 for i in range(len(time_list))]
    
    def update(self, imgs, gt, learning_rate=0, training=True,timestep = 0.5):
        

        
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()

        if training:
            flow, mask, merged, pred = self.net(imgs,timestep= timestep)
            
            loss_l1, loss_perc, loss_total = self.lap(pred, gt)
            loss_l1 = loss_l1.mean()
            loss_perc = loss_perc.mean()
            loss_total = loss_total.mean()

            for merge in merged:
                loss_l1_t, loss_perc_t, loss_total_t = self.lap(merge, gt)
                loss_l1 += loss_l1_t.mean() * 0.5
                loss_perc += loss_perc_t.mean() * 0.5
                loss_total += loss_total_t.mean() * 0.5


            self.optimG.zero_grad()
            loss_total.backward()
            self.optimG.step()
            return pred, loss_l1, loss_perc, loss_total
        else: 
            with torch.no_grad():
                flow, mask, merged, pred = self.net(imgs,timestep= timestep)
                return pred, 0
    def freeze_backbone(self):
        exclude_layers = ['unet','feature_bone.block2','feature_bone.block3','feature_bone.block1']
        for name,param in self.net.named_parameters():
            if any(exclude_layer in name for exclude_layer in exclude_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
