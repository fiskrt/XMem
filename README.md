# Filip's thesis @ CVL
## Weakly supervised Video Object Segmentation

Video object segmentation is a fundamental problem in computer vision used in a variety of application
across many fields. Over the past few years video object segmentation has witnessed rapid progress catalyzed
by increasingly large datasets. These datasets consisting of pixel-accurate masks with object association
between frames are especially labor-intensive and costly, prohibiting truly large-scale datasets. We
propose a video object segmentation model capable of being trained exclusively with bounding boxes, a
cheaper type of annotation. To achieve this, our method employs loss functions tailored for box-annotations
that leverages self-supervision through color similarity and spatio-temporal coherence.
We validate our approach against traditional fully-supervised methods and various other settings on YouTube-VOS
and DAVIS, achieving over 90% relative performance on $\mathcal{J}\\&\mathcal{F}$ in comparison to fully-supervised
models in the box-initialization setting, while scoring around 85% in the mask-initialization setting. We
also investigate practical aspects of our model, achieving a relative performance of 87% on longer term
videos with 1000s of frames. We also perform ablations both quantitatively and qualitatively and show 
visually how the loss function improves fine-detail along with failure cases. Moreover, our method is practical
with over 22 frames per second on the YouTubeVOS validation set.




### Base training code is from XMem

```bibtex
@inproceedings{cheng2022xmem,
  title={{XMem}: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model},
  author={Cheng, Ho Kei and Alexander G. Schwing},
  booktitle={ECCV},
  year={2022}
}
```
