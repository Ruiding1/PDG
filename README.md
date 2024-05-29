# Progressive Diversity Generation for Single Domain Generalization (PDG)

PDG extends the previous work "Diversity Probe" (ACM MM2023) by incorporating the $f$-diversity and the progressive generative framework to generate potential images from diverse perspectives.

Paper Link: https://dl.acm.org/doi/abs/10.1145/3581783.3612375

This paper appears in: IEEE Transactions on Multimedia

## Environment

- python=3.9.16
- torch==2.0.1
- torchvision==0.15.2
- munkres=1.1.4
- numpy==1.24.1
- opencv-python==4.7.0.72
- scikit-learn=1.2.2
- pandas==2.0.1

## Setting up the data

**Note**: You need to download the data if you wish to train your own model.

Download the digits dataset from this [link](https://pan.baidu.com/s/15XTZxbFY_JnTk_FZB4-1Jw )[BaiDuYunDisk] and its extracted code: `xcl3`. Please extract it inside the `data` directory

```shell
cd data
unzip digits.zip
cd ..
```

## Evaluating the model

Pretrained task model is available at this [link](https://pan.baidu.com/s/18bQCqtW5mCzeT-GByQar8w)[BaiDuYunDisk] and its extracted code:`2mz0`. Download and extract it in the `models_pth` directory.

In `train.py`:

- Specify the output directory to save the results in `--dir`.
- Turn on the evaluation in`--eval`
- Run `python train.py --dir SAVE_DIR --eval True`

## Acknowledgement

We thank the following authors for releasing their source code, data and models:

- [Progressive Domain Expansion Network for Single Domain Generalization](https://arxiv.org/abs/2103.16050)
- [Learning to Diversify for Single Domain Generalization](https://arxiv.org/abs/2108.11726)
