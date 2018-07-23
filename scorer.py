import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--pred', action='store', dest='pred',
                    help='Path to prediction output')
parser.add_argument('--gold', action='store', dest='gold',
                    help='Path to gold data')
opt = parser.parse_args()


def score_sentence(predline, goldline):
    predwrds = predline.split()
    goldwrds = goldline.split()

    bs = 0
    fullmatch = (predline == goldline)
    predbs = predwrds.count('B')
    goldbs = goldwrds.count('B')

    for ind, pw in enumerate(predwrds):
        gw = goldwrds[ind]

        if gw == 'B' and pw == 'B':
            # matched a bias word!
            bs += 1

    return bs, predbs, goldbs, fullmatch



def score():
    f_pred = open(opt.pred, 'r')
    f_gold = open(opt.gold, 'r')

    sent_matches = 0
    full_matches = 0
    total_gold_bs = 0
    total_pred_bs = 0

    total_sents = 0


    for predline in f_pred:
        goldline = f_gold.readline()
        total_sents += 1

        bs, predbs, goldbs, fullmatch = score_sentence(predline, goldline)

        sent_matches += (bs > 0 and 1 or 0)
        full_matches += (fullmatch and 1 or 0)
        total_gold_bs += goldbs
        total_pred_bs += predbs


    print('Matched at least 1 biased word in %d out of %d. %5.2f accuracy' %
          (sent_matches, total_sents, sent_matches/total_sents * 100.))

    print('Fully matched %d sentences (%5.2f)' % (full_matches, full_matches/total_sents*100))

    print('Total Bs in gold source: %d, in prediction: %d' % (total_gold_bs, total_pred_bs))


score()
