import time

from rouge import Rouge
from rouge import FilesRouge  # compare entire file
import os
import argparse

parser = argparse.ArgumentParser(description='Rouge')
parser.add_argument('--gen_file', type=str, default='../save/test.data',
                    help='the hyp file')
parser.add_argument('--dataset', type=str, default='image_coco',
                    help='ref file')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    files_rouge = FilesRouge()

    # scores = files_rouge.get_scores(hyp_path, ref_path)  [行數必須要一樣，通常gen出來的都是9984, 因此對照的ref也要是9984]
    hyp_path = args.gen_file # generated output

    ref_path = "data/testdata/test_coco_ref.txt"  # real output
    if args.dataset == 'emnlp_news':
        ref_path = "data/testdata/test_emnlp_ref.txt"

    scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)  # 會直接算平均 有需要的話寫ignore
    print(scores)
    total_rouge = 0.0
    for i in scores:
        total_rouge += scores[i]['r']
    print("total rouge recall: ", total_rouge)
    total_rouge_f = 0.0
    for i in scores:
        total_rouge_f += scores[i]['f']
    print("total rouge f1-score: ", total_rouge_f)

    rouge_res = dict()
    rouge_res['score'] = scores
    rouge_res['score_recall'] = total_rouge
    rouge_res['score_f'] = total_rouge_f

    ds_name = 'imagecoco' if args.dataset == 'image_coco' else 'emnlpnews'
    logdir = os.path.join('..', 'experiments', time.strftime("%Y%m%d"), ds_name, 'rouge')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    file_name = time.strftime("%H%M%S") + '.txt'
    store = os.path.join(logdir, file_name)

    # store to file
    with open(store, 'w') as f:
        for k, v in rouge_res.items():
            f.write('%s : %s\n' % (k, v))