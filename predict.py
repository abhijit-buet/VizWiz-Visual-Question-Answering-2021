import argparse
import json
import os.path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from torch.autograd import Variable
from tqdm import tqdm

import models
from datasets import vqa_dataset


def predict_answers(model, loader, split):
    model.eval()
    predicted = []
    samples_ids = []

    tq = tqdm(loader)

    print("Evaluating...\n")

    for item in tq:
        v = item['visual']
        q = item['question']
        sample_id = item['sample_id']
        q_length = item['q_length']

        v = Variable(v.cuda())
        q = Variable(q.cuda())
        q_length = Variable(q_length.cuda())

        out = model(v, q, q_length)

        _, answer = out.data.cpu().max(dim=1)

        predicted.append(answer.view(-1))
        samples_ids.append(sample_id.view(-1).clone())

    predicted = list(torch.cat(predicted, dim=0))
    samples_ids = list(torch.cat(samples_ids, dim=0))

    print("Evaluation completed")

    return predicted, samples_ids


def create_submission(input_annotations, predicted, samples_ids, vocabs):
    answers = torch.FloatTensor(predicted)
    indexes = torch.IntTensor(samples_ids)
    ans_to_id = vocabs['answer']
    # need to translate answers ids into answers
    id_to_ans = {idx: ans for ans, idx in ans_to_id.items()}
    # sort based on index the predictions
    sort_index = np.argsort(indexes)
    sorted_answers = np.array(answers, dtype='int_')[sort_index]

    real_answers = []
    for ans_id in sorted_answers:
        ans = id_to_ans[ans_id]
        real_answers.append(ans)

    # Integrity check
    assert len(input_annotations) == len(real_answers)

    submission = []
    submission_answer = [] 
    for i in range(len(input_annotations)):
        pred = {}
        pred_ = {}


        pred['image'] = input_annotations[i]['image']
        pred['answer'] = real_answers[i]
        submission.append(pred)
        
        pred_['image'] = input_annotations[i]['image']
        pred_['answerable'] = 1.
        if real_answers[i] == "unanswerable":
            pred_['answerable'] = 0.0

        submission_answer.append(pred_)

    return submission, submission_answer


def main():
    # Load config yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', default='config/default.yaml', type=str,
                        help='path to a yaml config file')
    args = parser.parse_args()

    if args.path_config is not None:
        with open(args.path_config, 'r') as handle:
            config = yaml.load(handle)

    cudnn.benchmark = True

    # Generate dataset and loader
    print("Loading samples to predict from %s" % os.path.join(config['annotations']['dir'],
                                                              config['prediction']['split'] + '.json'))

    # Load annotations
    path_annotations = os.path.join(config['annotations']['dir'], config['prediction']['split'] + '.json')
    input_annotations = json.load(open(path_annotations, 'r'))

    # Data loader and dataset
    input_loader = vqa_dataset.get_loader(config, split=config['prediction']['split'])

    # Load model weights
    print("Loading Model from %s" % config['prediction']['model_path'])
    log = torch.load(config['prediction']['model_path'])

    # Num tokens seen during training
    num_tokens = len(log['vocabs']['question']) + 1
    # Use the same configuration used during training
    train_config = log['config']

    model = nn.DataParallel(models.Model(train_config, num_tokens)).cuda()

    dict_weights = log['weights']
    model.load_state_dict(dict_weights)

    predicted, samples_ids = predict_answers(model, input_loader, split=config['prediction']['split'])

    submission, submission_answer = create_submission(input_annotations, predicted, samples_ids, input_loader.dataset.vocabs)

    with open(config['prediction']['submission_file'], 'w') as fd:
        json.dump(submission, fd)

    with open(config['prediction']['submission_file_2'], 'w') as fd:
        json.dump(submission_answer, fd)

    print("Submission file saved in %s" % config['prediction']['submission_file'])


if __name__ == '__main__':
    main()
