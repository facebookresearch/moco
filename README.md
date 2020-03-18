## MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png" width="300">
</p>

This is a PyTorch implementation of MoCo unsupervised pre-training as described in the [arXiv paper](https://arxiv.org/abs/1911.05722):

```
@Article{he2019momentum,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```

This repo aims to be minimal modifications on the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Check the modifications by:
```
diff main_moco.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
diff main_lincls.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
```


### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).


### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```
This script uses all the default hyper-parameters as described in the paper.

***Note***: for 4-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): try `--lr 0.015 --batch-size 128` with 4 gpus. We got similar results using this setting.

The code includes a simple monitor of the (K+1)-way contrastive classification error during training. We provide this training curve for your reference. This is **neither** the validation monitor **nor** related to the transferring performance; it is only a weak monitor in case something goes badly wrong.
<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/71604707-c9055200-2b18-11ea-8cfa-dd6d4378e6cb.png" width="600">
</p>

### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

Here are the linear classification results on ImageNet using this repo with 8 NVIDIA V100 GPUs :
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">pre-train<br/>time</th>
<th valign="bottom">linear cls<br/>epochs</th>
<th valign="bottom">linear cls<br/>time</th>
<th valign="bottom">top-1<br/>acc. %</th>
<th valign="bottom">top-5<br/>acc. %</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">54 hours</td>
<td align="center">100</td>
<td align="center">12 hours</td>
<td align="center">60.8&plusmn;0.2</td>
<td align="center">83.1&plusmn;0.2</td>
</tr>
</tbody></table>

Here we run 5 trials (of pre-training and linear classification) and report mean&plusmn;std. The 5 trials have top-1 accuracy of 60.6, 60.6, 60.7, 60.9, 61.1.

***Note***: for faster prototyping, we suggest 100-epoch pre-training `--schedule 80 90 --epochs 100`. This reduces pre-training to 27 hours, with top-1 accuracy of about 59.0.


### Transferring to Object Detection

See [./detection](detection).
