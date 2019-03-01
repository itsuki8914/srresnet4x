import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from model import *

DATASET_DIR = "data"
VAL_DIR ="val"
TEST_DIR = "test"
SAVE_DIR = "model"
SAVEIM_DIR ="pred"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if not os.path.exists(SAVEIM_DIR):
    os.makedirs(SAVEIM_DIR)

class BatchGenerator:
    def __init__(self, img_size, datadir):
        self.folderPath = datadir
        self.imagePath = glob.glob(self.folderPath+"/*.png")
        #self.orgSize = (218,173)
        self.imgSize = (img_size,img_size)
        assert self.imgSize[0]==self.imgSize[1]

    def augment(self,img):
        img = cv2.flip(img,1)
        return img

    def getBatch(self,nBatch,id,rand=0):
        x   = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],3), dtype=np.float32)
        for i,j in enumerate(id):

            img = cv2.imread(self.imagePath[j])
            dmin = min(img.shape[0],img.shape[1])
            img = img[int(0.5*(img.shape[0]-dmin)):int(0.5*(img.shape[0]+dmin)),int(0.5*(img.shape[1]-dmin)):int(0.5*(img.shape[1]+dmin)),:]
            if rand > 0.5:
                im = self.augment(img)
            img = cv2.resize(img,self.imgSize)
            x[i,:,:,:] = (img - 127.5) / 127.5

        return x

def loss_g(y, t):
    reg_collections = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    weight_decay = tf.reduce_sum(reg_collections)

    l2 = tf.nn.l2_loss(y-t)
    l1 = tf.reduce_sum(tf.abs(y-t)) *0.05
    wd = weight_decay

    loss = l2   + wd +l1

    return loss , l2 , wd


def training(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01,
                                       beta1=0.9,
                                       beta2=0.999)
    train_step = optimizer.minimize(loss)
    return train_step

def tileImage(imgs):
    d = int(math.sqrt(imgs.shape[0]-1))+1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.zeros((h*d,w*d,3),dtype=np.float32)
    for idx,img in enumerate(imgs):
        idx_y = int(idx/d)
        idx_x = idx-idx_y*d
        r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
    return r

def main():
    img_size = 96
    bs = 8
    val_size = 16
    start = time.time()

    dir = DATASET_DIR
    paths = os.listdir(dir)
    datalen = len(paths)
    print("train: {} pics".format(datalen))

    val = VAL_DIR
    vpaths = os.listdir(val)
    vallen = len(vpaths)
    print("val: {} pics".format(vallen))

    # loading images on training
    batch_x = BatchGenerator(img_size=img_size,datadir=DATASET_DIR)
    batch_t = BatchGenerator(img_size=img_size*4,datadir=DATASET_DIR)

    btval_x = BatchGenerator(img_size=img_size,datadir=VAL_DIR)
    btval_y = BatchGenerator(img_size=img_size*4,datadir=VAL_DIR)

    id = np.random.choice(range(datalen),bs)

    IN_ = tileImage(batch_x.getBatch(bs,id)[:4])
    IN_ = cv2.resize(IN_,(img_size*2*4,img_size*2*4))
    IN_ = (IN_ + 1)*127.5
    OUT_ = tileImage(batch_t.getBatch(bs,id)[:4])
    OUT_ = cv2.resize(OUT_,(img_size*4*2,img_size*4*2))
    OUT_ = (OUT_ + 1)*127.5

    Z_ = np.concatenate((IN_,OUT_), axis=1)
    cv2.imwrite("input.png",Z_)
    print("%s sec took sampling"%(time.time()-start))

    start = time.time()

    x = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
    t = tf.placeholder(tf.float32, [None, img_size*4, img_size*4, 3])

    y =buildSRGAN_g(x,nBatch=bs)
    print("modelbuid,{}".format(time.time()-start))

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("total parameters:",total_parameters)

    loss,l2,wd = loss_g(y, t)
    print("lossbuid,{}".format(time.time()-start))

    train_step = training(loss)
    print("optbuid,{}".format(time.time()-start))

    sess = tf.Session()

    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state('model')

    if ckpt: # checkpointがある場合
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print ("load " + last_model)
        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    print("%.4e sec took initializing"%(time.time()-start))

    hist =[]
    hist_v =[]

    start = time.time()

    for i in range(100000):
        # loading images on training
        id = np.random.choice(range(datalen),bs)
        rand = np.random.rand()
        batch_images_x = batch_x.getBatch(bs,id,rand)
        batch_images_t = batch_t.getBatch(bs,id,rand)

        tmp, yloss,l2_loss,wd_loss = sess.run([train_step,loss,l2,wd],
            feed_dict={
            x: batch_images_x,
            t: batch_images_t
        })
        print("in step %s loss = %.4e, L2loss = %.4e, WDloss = %.4e"
            %(i,yloss,l2_loss,wd_loss))

        if i %100 ==0:
            id = np.random.choice(range(vallen),val_size)
            btval_images_x = btval_x.getBatch(val_size,id)
            btval_images_y = btval_y.getBatch(val_size,id)
            val_loss = loss.eval(session=sess, feed_dict={
                x: btval_images_x,
                t: btval_images_y
            })

            val_loss = val_loss /val_size*bs
            hist.append(yloss)
            #print(val_loss)
            hist_v.append(val_loss)
            print("validation loss = %.4e " %val_loss)
            out = sess.run(y,feed_dict={
                x:btval_images_x})
            X_ = tileImage(btval_images_x[:4])
            X_ = cv2.resize(X_,(img_size*4*2,img_size*4*2))
            #print(X_.shape)
            Y_ = tileImage(out[:4])

            GT_ = tileImage(btval_images_y[:4])
            GT_ = cv2.resize(GT_,(img_size*4*2,img_size*4*2))
            GT_ = (GT_ + 1)*127.5

            X_ = (X_ + 1)*127.5
            Y_ = (Y_ + 1)*127.5
            Z_ = np.concatenate((X_,Y_,GT_), axis=1)
            cv2.imwrite("{}/{}.png".format(SAVEIM_DIR,i),Z_)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.title("Loss")
            plt.grid(which="both")
            plt.yscale("log")
            ax.plot(hist,label="loss_train")
            ax.plot(hist_v,label="loss_val")
            plt.xlabel('x{} step'.format(hist_freq), fontsize = 16)
            plt.ylabel('loss', fontsize = 16)
            plt.legend(loc = 'upper right')
            plt.savefig("hist.png")
            plt.close()

            print("%.4e sec took per 100steps" %(time.time()-start))
            start = time.time()

        if i%5000==0 and i!=0:

            loss_1k_old=0
            loss_1k_new=0
            if i >10000:
                loss_1k_old = np.mean(hist_v[-100:-50])
                loss_1k_new = np.mean(hist_v[-50:])
                print("old loss=%.4e , new loss=%.4e"%(loss_1k_old,loss_1k_new))
            if loss_1k_old*2 < loss_1k_new and i > 30000:
                break

            saver.save(sess,os.path.join(SAVE_DIR,"model.ckpt"),i)

if __name__ == '__main__':
    main()
