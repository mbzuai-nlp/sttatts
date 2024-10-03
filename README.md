<div align="center">

<h1> STTATTS </h1>
This repository contains the implementation of the paper:

**STTATTS**: Unified **S**peech-**T**o-**T**ext and **T**ext-**T**o**S**peech Model

<div>
    <a href='https://www.linkedin.com/in/toyinhawau/' >Hawau Olamide Toyin </a>&emsp;
    <a href='' target='_blank'>Hao Li </a>&emsp;
    <a href='https://linkedin.com/in/hanan-aldarmaki/' target='_blank'>Hanan Aldarmaki </a>&emsp;
</div>
<!-- <br> -->
<div>
     MBZUAI &emsp;
</div>
<!-- <br> -->
<i><strong><a href='' target='_blank'>EMNLP 2024 (findings)</a></strong></i>
<br>
</div>


### Checkpoints

Finetuned checkpoints are available for [Arabic]() and [English-small](https://huggingface.co/MBZUAI/STTATTS/blob/main/checkpoint_en_small.pt). To finetune on your dataset, download pretrained checkpoints,tokenizer and dict from [ArTST](https://github.com/mbzuai-nlp/ArTST/) and [SpeechT5](https://github.com/microsoft/SpeechT5/tree/main/SpeechT5).


### Finetune, Installation and Inference

See finetune scripts [here](./scripts/). Installation and Inference follows [ArTST](https://github.com/mbzuai-nlp/ArTST/) repo.


### Acknowledgements

STTATTS is built on [ArTST](https://aclanthology.org/2023.arabicnlp-1.5/) and [SpeechT5](https://arxiv.org/abs/2110.07205). If you use any of STTATTS models, please cite the papers:

```
@inproceedings{toyin2023artst,
  title={ArTST: Arabic Text and Speech Transformer},
  author={Toyin, Hawau and Djanibekov, Amirbek and Kulkarni, Ajinkya and Aldarmaki, Hanan},
  booktitle={Proceedings of ArabicNLP 2023},
  pages={41--51},
  year={2023}
}

@article{ao2021speecht5,
  title={Speecht5: Unified-modal encoder-decoder pre-training for spoken language processing},
  author={Ao, Junyi and Wang, Rui and Zhou, Long and Wang, Chengyi and Ren, Shuo and Wu, Yu and Liu, Shujie and Ko, Tom and Li, Qing and Zhang, Yu and others},
  journal={arXiv preprint arXiv:2110.07205},
  year={2021}
}
```


