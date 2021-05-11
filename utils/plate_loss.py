# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class CosinLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self):
        super(CosinLoss, self).__init__()

    def forward(self, pred, target, mask):
        target_reshape = target.reshape(target.size(0), -1, 2)
        pred_reshape   = pred.reshape(pred.size(0), -1, 2)
        mask_reshape   = mask.reshape(mask.size(0), -1, 2)
        similarity = torch.cosine_similarity(target_reshape, pred_reshape, dim=-1)
        loss = 1 - similarity
        loss = loss * mask_reshape[:, :, 0]
        return loss.sum()
    
    
class WingLoss(nn.Module):
    def __init__(self, w=10, e=2):
        super(WingLoss, self).__init__()
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t, sigma=1):
        weight = torch.ones_like(t)
        weight[torch.where(t==-1)] = 0
        diff = weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        return y.sum()

class CrossIouLoss(nn.Module):
    def __init__(self):
        super(CrossIouLoss, self).__init__()
        self.eps = 1e-6
    def forward(self, pred, target, mask):
        target_reshape = target.reshape(target.size(0), -1, 4)
        pred_reshape   = pred.reshape(pred.size(0), -1, 4)
        mask_reshape   = mask.reshape(mask.size(0), -1, 4)
        total = torch.stack([pred_reshape, target_reshape], -1)
        #print('total: ', total.shape)
        l_max = total.max(dim=-1)[0].clamp(min=self.eps)
        l_min = total.min(dim=-1)[0]
        #print('l_max: ', l_max.shape)
        overlaps = l_min.sum(dim=-1)/l_max.sum(dim=-1)
        #print('overlaps: ', overlaps)
        #overlaps = overlaps.sum(-1)/total.size(1)
        loss = 1 - overlaps
        mask_reshape = mask_reshape[:,:,0]
        loss = loss * mask_reshape
        loss = loss.sum() / (torch.sum(mask_reshape) + self.eps)
        #print('loss: ', loss.shape)
        return loss

class LandmarksLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self):
        super(LandmarksLoss, self).__init__()
        self.loss_fcn = nn.SmoothL1Loss(reduction='sum')

    def forward(self, pred, truel, mask):
        loss = self.loss_fcn(pred*mask, truel*mask)
        return loss / (torch.sum(mask) + 1e-6)

class ComputePlateLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputePlateLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        self.landmarks_loss = LandmarksLoss()

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj, lmark = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors, tlandmarks, lmks_mask = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            #print('pi: ', pi.shape)
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                #print('tobj: ',tobj[b, a, gj, gi] )

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

                #landmarks loss
                #print('ps: ', ps.shape)
                plandmarks = ps[:,5:13].sigmoid() * 8. - 4.

                #print('anchors: ', anchors[i].shape)
                #print('plandmarks: ', plandmarks.shape)

                plandmarks[:, 0:2] = plandmarks[:, 0:2] * anchors[i]
                plandmarks[:, 2:4] = plandmarks[:, 2:4] * anchors[i]
                plandmarks[:, 4:6] = plandmarks[:, 4:6] * anchors[i]
                plandmarks[:, 6:8] = plandmarks[:, 6:8] * anchors[i]

                lmark += self.landmarks_loss(plandmarks, tlandmarks[i], lmks_mask[i])
            #print('tobj: ',tobj)
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lmark *= self.hyp['landmark']

        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lmark
        return loss * bs, torch.cat((lbox, lobj, lcls, lmark, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        #print('na: ', na , ' ---  ', 'nt: ', nt)
        tcls, tbox, indices, anch, landmarks, lmks_mask = [], [], [], [], [], []
        gain = torch.ones(15, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        #print(' targets: ', targets.shape)
        #print('  p: ', len(p))
        #print('  p: ', p[2].shape)
        

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            #print(i, ' - p: ', p[i].shape)
            #print(i, ' - p: ', gain.shape)
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]   # xyxy gain
            #print(i, ' ****  gain: ', gain.shape)
            #landmarks 8
            gain[6:14] = torch.tensor(p[i].shape)[[3, 2, 3, 2, 3, 2, 3, 2]]  # xyxy gain
            #temp = torch.tensor(p[i].shape)[[3, 2, 3, 2, 3, 2, 3, 2]]
            #print(temp)
            #temp = temp.to(targets.device)
            #gain[6:14] = temp


            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 14].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

            #landmarks
            lks = t[:,6:14]
            #print('t: ', t.shape)
            #print('lks: ', lks.shape)
            #lks_mask = lks > 0
            #lks_mask = lks_mask.float()
            lks_mask = torch.where(lks < 0, torch.full_like(lks, 0.), torch.full_like(lks, 1.0))
            #for index in range(lks_mask.shape[1] // 2):
            #    lks[:, [index*2, index*2+1]] =   lks[:, [index*2, index*2+1]] - gij
            lks[:, [0, 1]] = (lks[:, [0, 1]] - gij)
            lks[:, [2, 3]] = (lks[:, [2, 3]] - gij)
            lks[:, [4, 5]] = (lks[:, [4, 5]] - gij)
            lks[:, [6, 7]] = (lks[:, [6, 7]] - gij)

            #for cross_iou
            use_cross_iou = False
            if use_cross_iou:
                new_lks = lks.repeat(1, 2)
                lks_x1y1 = new_lks[:, 0:2]
                lks_x2y2 = new_lks[:, 2:4]
                lks_x1y1[lks_x1y1 < 0] = -0.2 * lks_x1y1[lks_x1y1 < 0]
                lks_x2y2[lks_x2y2 > 0] = 0.2 * lks_x2y2[lks_x2y2 > 0]
                lks_x2y2[lks_x2y2 < 0] = -1.0 * lks_x2y2[lks_x2y2 < 0]

                new_lks_mask = lks_mask.repeat(1, 2) 

            lmks_mask.append(lks_mask)
            landmarks.append(lks)

        return tcls, tbox, indices, anch, landmarks, lmks_mask
