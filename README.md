# DeepNSM -- Natural Scene Memorability

The open source codes of the paper:

> Jiaxin Lu, Mai Xu, Ren Yang, Zulin Wang, "Understanding and Predicting the Memorability of Outdoor Natural Scenes", in IEEE Transactions on Image Processing (T-IP), 2020. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9025769). 

If our paper and codes are useful for your research, please cite:
```
@article{lu2020understanding,
  title={Understanding and Predicting the Memorability of Outdoor Natural Scenes},
  author={Lu, Jiaxin and Xu, Mai and Yang, Ren and Wang, Zulin},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4927--4941},
  year={2020},
  publisher={IEEE}
}
```
## Dependency

TensorFlow (we tested on TensorFlow 1.12 and 1.15)

opencv-python

## How to use
Please first dowmload the pre-trained model [[Link]](https://drive.google.com/drive/folders/1Tpwv__MWHV0ul-627uQNbOuqyFWePJ-N?usp=sharing), and put the folder "Models" into the same directory as this code.

Then run "test_one_image.py" to predict the memerobility score of the input image, e.g.,

```
python test_one_image.py --img example.jpg
```

## Notice

As stated in our paper, this model is trained for outdoor natual scenes, i.e., the images which are only composed of outdoor natural scenes,
without any human, animal and man-made object. The pre-trained model may be not suitable for other kinds of images.

The aim of predicting memorability is to rank the image memorability as correctly as possible. Therefore, the absolute scores (e.g., 0.61) themselves are not meaningful , and only the rank matters, e.g., the image with score = 0.61 is more memorable than that with score = 0.40.

## LNSIM dataset

The proposed LNSIM dataset is available at https://github.com/JiaxinLu-home/Natural-Scene-Memorability-Dataset


## Contact

If you find any bug of our codes or have any question, please do not hesitate to contact:

Jiaxin Lu (lu-jia-xin@163.com)

Ren Yang (r.yangchn@gmail.com)


