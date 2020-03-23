
## MoCo: Transferring to Detection

The `train_net.py` script reproduces the object detection experiments on Pascal VOC and COCO.

### Instruction

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

1. Convert a pre-trained MoCo model to detectron2's format:
   ```
   python3 convert-pretrain-to-detectron2.py input.pth.tar output.pkl
   ```

1. Put dataset under "./datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

1. Run training:
   ```
   python train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./output.pkl
   ```

### Results

Below are the results on Pascal VOC 2007 test, fine-tuned on 2007+2012 trainval for 24k iterations using Faster R-CNN with a R50-C4 backbone:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">pretrain</th>
<th valign="bottom">AP50</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP75</th>
<!-- TABLE BODY -->
<tr><td align="left">ImageNet-1M, supervised</td>
<td align="center">81.3</td>
<td align="center">53.5</td>
<td align="center">58.8</td>
</tr>
<tr><td align="left">ImageNet-1M, MoCo v1, 200ep</td>
<td align="center">81.5</td>
<td align="center">55.9</td>
<td align="center">62.6</td>
</tr>
</tr>
<tr><td align="left">ImageNet-1M, MoCo v2, 200ep</td>
<td align="center">82.4</td>
<td align="center">57.0</td>
<td align="center">63.6</td>
</tr>
</tr>
<tr><td align="left">ImageNet-1M, MoCo v2, 800ep</td>
<td align="center">82.5</td>
<td align="center">57.4</td>
<td align="center">64.0</td>
</tr>
</tbody></table>

***Note:*** These results are means of 5 trials. Variation on Pascal VOC is large: the std of AP50, AP, AP75 is expected to be 0.2, 0.2, 0.4 in most cases. We recommend to run 5 trials and compute means.
