Visual feature are extracted using a pretrained (on ImageNet) ResNet-152. Input Questions are tokenized, embedded and encoded with an LSTM. Image features and encoded questions are combined and used to compute multiple attention maps over image features. The attended image features and the encoded questions are concatenated and finally fed to a 2-layer classifier that outputs probabilities over the answers 



#### Experimental Results 

| method       | accuracy |
|--------------|----------|
| [VizWiz Paper][0] | 0.475    |
| Proposed Method   |**0.526**|



```


## Acknowledgment


- https://github.com/DenisDsh/VizWiz-VQA-PyTorch
- https://github.com/liqing-ustc/VizWiz_LSTM_CNN_Attention/
- https://github.com/Cadene/vqa.pytorch
- https://github.com/GT-Vision-Lab/VQA_LSTM_CNN
- https://github.com/Cyanogenoid/pytorch-vqa



[0]: http://vizwiz.org/data/
[1]: https://arxiv.org/abs/1704.03162
[2]: https://arxiv.org/pdf/1511.02274
[3]: https://arxiv.org/abs/1708.00584
[4]: https://arxiv.org/pdf/1505.00468v6.pdf

