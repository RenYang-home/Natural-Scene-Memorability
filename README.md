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

## How to use
Please first dowmload the pre-trained model. [[Link]](https://drive.google.com/drive/folders/1Tpwv__MWHV0ul-627uQNbOuqyFWePJ-N?usp=sharing)

Then run "test_one_image.py" to predict the memerobility score of the input image, e.g.,

```
python test_one_image.py --img example.jpg
```

## Notice

As stated in our paper, this model is trained for outdoor natual scenes, i.e., the images which are only composed of outdoor natural scenes,
without any human, animal and man-made object. The pre-trained model may be not suitable for other kinds of image.

The aim of predicting memorability is to rank the image memorability as correctly as possible. Therefore, the absolute score (e.g., 0.62) itself does not have clear meaning, and only the rank is meaningful, e.g., the image with score = 0.62 is more memorable than the image with score = 0.40. 


