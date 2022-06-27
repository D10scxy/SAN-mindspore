#/usr/bin/env Python

import numpy as np
import random
import pickle as pkl

from VQA.vqa import VQA
from VQA.vqaEval import *
from mindspore import Tensor

def reshape_image_feat(batch_image_feat, num_region, region_dim):
    return np.reshape(np.asarray(batch_image_feat), (batch_image_feat.shape[0],
                                                     num_region, region_dim))

def preProcess(cfg):
    ################################
    # # process the train question #
    ################################
    vqa = VQA(cfg.trainAnnFile, cfg.trainQuesFile)
    train_question_ids = vqa.getQuesIds()
    train_questions = []
    train_answers = []

    random.shuffle(train_question_ids)
    train_question_ids = train_question_ids[:cfg.trainSize]
    train_image_ids = vqa.getImgIds(quesIds=train_question_ids)

    for idx, q_id in enumerate(train_question_ids):
        question = vqa.loadQuestion(q_id)
        question = processQuestion(question)
        answer = vqa.loadAnswer(q_id)
        answer = processAnswer(max(answer, key=answer.count))
        train_questions.append(question)
        train_answers.append(answer)
        if idx % 1000 == 0:
            print('finished processing %d in train' %(idx))

    print ('finished processing train')

    ############################
    # processing val questions #
    ############################
    vqa = VQA(cfg.valAnnFile, cfg.valQuesFile)
    val_question_ids = vqa.getQuesIds()
    val_questions = []
    val_answers = []

    random.shuffle(val_question_ids)
    val_question_ids = val_question_ids[:cfg.valSize]
    val_image_ids = vqa.getImgIds(quesIds=val_question_ids)
    
    for idx, q_id in enumerate(val_question_ids):
        question = vqa.loadQuestion(q_id)
        question = processQuestion(question)
        val_questions.append(question)
        answer = vqa.loadAnswer(q_id)
        answer = processAnswer(max(answer, key=answer.count))
        val_answers.append(answer)
        if idx % 1000 == 0:
            print('finished processing %d in train' %(idx))

    print('finished processing val')

    # process the image list and feature
    train_image_feature = []
    val_image_feature = []
    
    miss_img = 0
    mask = np.zeros([49, 2048])
    with open(cfg.trainFeatFile, "rb") as f:
        train_image = pkl.load(f).items()
        for i in train_image_ids:
            idx = str(i)
            while len(idx) < 12:
                idx = '0' + idx
            item = list(filter(lambda x: x[0].find(idx) > -1, train_image))
            if len(item) == 0:
                train_image_feature.append(mask)
                miss_img += 1
                continue
            item = item[0]
            ls = item[0].split('.')
            name = list(filter(lambda x: x.find(idx) > -1, ls))
            train_image_feature.append(item[1][ls.index(name[0])].T.reshape((49, 2048)).asnumpy())

    with open(cfg.valFeatFile, "rb") as f:
        val_image = pkl.load(f).items()
        for i in val_image_ids:
            idx = str(i)
            while len(idx) < 12:
                idx = '0' + idx
            item = list(filter(lambda x: x[0].find(idx) > -1, val_image))
            if len(item) == 0:
                train_image_feature.append(mask)
                miss_img += 1
                continue
            item = item[0]
            ls = item[0].split('.')
            name = filter(lambda x: x.find(idx) > -1, ls)
            val_image_feature.append(item[1][ls.index(list(name)[0])].T.reshape((49, 2048)).asnumpy())
    
    print('finished processing features. Miss Images: ', miss_img)

    return train_questions, train_answers, train_image_feature, val_questions, val_question_ids, val_image_feature, val_answers
