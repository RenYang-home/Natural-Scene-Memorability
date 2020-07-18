import numpy as np
import scipy.io as sio
import shutil


def label_create(org_labels, index_list):
    img_name = []
    score = []
    index_list = list(index_list)
    index_list.sort()
    for i in range(len(index_list)):
        tmp = index_list[i]
        name_tmp = '%04d' % tmp + '.jpg'
        img_name.append(name_tmp)
        score_tmp = list(org_labels[tmp - 1])
        score_tmp = score_tmp[0]
        score.append(score_tmp)
    return img_name, score


def file_read(file_dst):
    f = open(file_dst)
    image_name = []
    gt_score = []
    for line in f:
        line = line.strip('\n')
        line = line.split(' ')
        image_name.append(line[0])
        gt_score.append(float(line[1]))
    f.close()
    return image_name, gt_score


def test_image_save():
    image_name, gt_score = file_read('./Data/gt_file.list')
    save_dir = './Data/test/'
    for i in range(len(image_name)):
        pic_name = image_name[i]
        shutil.copy('./Data/images/' + pic_name, save_dir + pic_name)


def main():
    path = './Data/'
    test_list = np.load(path + 'test_list.npy')
    hrs_file2 = sio.loadmat(path + 'labels/subject_hrs2.mat')
    hrs2 = hrs_file2['subject_hrs2']
    hrs2 = hrs2[0, :]
    img_name, score = label_create(hrs2, test_list)
    f = open(path + 'gt_file.list', 'w')
    for i in range(len(img_name)):
        print(i)
        pic_name = img_name[i]
        score1 = score[i]
        str1 = '{} {}\n'.format(pic_name, score1)
        f.write(str1)
    f.close()


if __name__ == '__main__':
    test_image_save()
