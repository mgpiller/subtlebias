import Levenshtein as l
import csv
import diff_match_patch
dmp = diff_match_patch.diff_match_patch()
import pandas

import pickle
import traceback


import spacy
my_tok = spacy.load('en')
from spacy.symbols import ORTH
my_tok.tokenizer.add_special_case(']]', [{ORTH: ']]'}])
my_tok.tokenizer.add_special_case('[[', [{ORTH: '[['}])
my_tok.tokenizer.add_special_case('/>', [{ORTH: '/>'}])
my_tok.tokenizer.add_special_case('__', [{ORTH: '__'}])
my_tok.tokenizer.add_special_case('===', [{ORTH: '==='}])
my_tok.tokenizer.add_special_case('==', [{ORTH: '=='}])
my_tok.tokenizer.add_special_case('}}', [{ORTH: '}}'}])


def my_spacy_tok(x): return [tok.text for tok in my_tok.tokenizer(x)]

def get_lexicon(lexfname):
    lex = []

    with open(lexfname) as lexfile:
        for l in lexfile:
            lex.append(l[:-1])
    return lex

def filter_edits(filename, lexicon):
    count = 0; invalid = 0; tot = 0; noseq = 0; sentmis = 0
    one_word = 0

    classification_data = []
    tagging_data = []
    oneword_tagging = []

    with open(filename) as csvfile:
        rdr = csv.reader(csvfile, delimiter='\t')

        for f in rdr:

            if f[3] == 'true':
                tot += 1
                # discard levenshtein distance of < 4
                try:
                    #if len(f) == 10:
                        before = f[6]
                        after = f[7]
                        ldist = l.distance(before, after)
                        if ldist > 3:
                            diff = dmp.diff_main(before, after)
                            dmp.diff_cleanupSemantic(diff)
                            #print(before, '->',after, diff)
                            # if len(f[6].split()) == 1 and any((c in f[6]) for c in lexicon) and \
                            #     not (any((c in f[6]) for c in ['{', '}', '[', ']', '(', ')', ':', '=', '"', "'"])) \
                            #     and 'http' not in f[6] and 'ref' not in f[6]\
                            #     and len(f[6].split()) == 1:
                            #     #print(f[6], '->', f[7])
                            if not (any((c in f[6]) for c in ['{', '}', '[', ']', ':', '='])) \
                                    and 'http' not in f[6] and 'ref' not in f[6]:
                                if len(f) < 9:
                                    sentmis += 1
                                    continue

                                bs = f[8].split()
                                tr = [1 if x == before else 0 for x in bs]
                                #tr = [before]
                                #tr = [x if x != before else '' for x in bs]
                                #training_data.append([bs, tr])
                                #print(tr)
                                rej = False
                                for dd in diff:
                                    if dd[0] != 0 and (dd[1].startswith('{') or dd[1].startswith('[') or dd[1].isdigit()):
                                        #print('REJECT: ', before, '->', after, diff)
                                        rej = True
                                if rej:
                                    continue
                                count += 1
                                #print('ACCEPT: ', before, '->', after, diff)
                                if len(f) == 10:
                                    classification_data.append([f[8], f[9]])
                                else:
                                    classification_data.append([f[8], ''])

                                source, target = get_tagged_seq(before, f[8])

                                if source:

                                    stc = " ".join(source)
                                    doc = my_tok(stc)
                                    lemmas = []
                                    poses = []
                                    grammar = []

                                    for token in doc:
                                        lemmas.append(token.lemma_)
                                        poses.append(token.tag_)
                                        grammar.append(token.dep_)

                                    tagging_data.append([source, target, lemmas, poses, grammar])

                                    if sum([1 for x in target if x == 'B']) == 1:
                                        one_word += 1
                                        oneword_tagging.append([source, target, lemmas, poses, grammar])
                                else:
                                    noseq += 1

                            else:
                                #print('REJECT: ',before, '->', after, diff)
                                pass
                        else:
                            #print('REJECT: ', before, '->', after, diff)
                            pass
                    #else:
                    #    invalid+=1
                except:
                    print('Exception:', f)
                    traceback.print_exc()

                    invalid += 1

    print('Found %d instances, invalid %d, no biased word found %d, before sentence missing %d, one_word %d' % (count, invalid, noseq, sentmis, one_word))
    df_class = pandas.DataFrame(classification_data, columns = ['yes', 'no'])
    df_tag = pandas.DataFrame(tagging_data, columns = ['src', 'tgt', 'lemmas', 'poses', 'grammar'])
    df_oneword_tag = pandas.DataFrame(oneword_tagging, columns=['src', 'tgt', 'lemmas', 'poses', 'grammar'])
    return df_class, df_tag, df_oneword_tag

def get_tagged_seq(bf, sf):
    sf = sf.replace("}}", " }} ").replace("]]", " ]] ").replace("{{", " {{ ").replace("[[", " [[ ").replace("/>", " /> ")\
            .replace("__", " __ ").replace("===", " === ").replace("==", " == ").replace("</ref>", " </ref> ").replace('"', ' " ')

    sl = my_spacy_tok(sf)
    bs = my_spacy_tok(bf)
    #remove punctuation from the ends
    if bs[0] in ['"', "{", '[', "[[", "]]", "/>", ",", '.', ')']:
        bs = bs[1:]
    if bs[-1] in ['"', "{", '[', "[[", "]]", "/>", ",", '.', ')']:
        bs = bs[:-1]

    out = []
    skip = 0


    for i, w in enumerate(sl):
        if skip:
            skip-=1
            out.append('B')
            continue
        if bs[0] == w and sl[i:i+len(bs)] == bs:
            skip = len(bs)-1
            out.append('B')
            continue

        out.append('_')

    if 'B' in out:
        if len(sl) != len(out):
            print("MISMATCH", sl, out)
        return sl, out
    else:
        return None,None

def output_sequences(oneword = False):
    lex = get_lexicon('/Users/Alex/data/bias-lexicon/bias-lexicon.txt')

    ftrain = open('train%s.tsv'%(oneword and '_oneword' or ''), 'w')
    ftest = open('test%s.tsv'%(oneword and '_oneword' or ''), 'w')

    for dset in ['test', 'train', 'dev']:
        df_class, df_tag, df_ow_tag = filter_edits('/Users/Alex/data/npov-edits/5gram-edits-%s.tsv' %dset, lex)
        #write_sequences('/Users/Alex/data/npov-edits/%s/data.txt'%dset, seqs)
        #pickle.dump(df_class, open(f'/Users/Alex/data/npov-edits/Classification_%s.pkl'%dset, 'wb'))
        pickle.dump(df_tag, open(f'/Users/Alex/data/npov-edits/Tagging_%s.pkl' % dset, 'wb'))

        f = ftrain if dset in ['train', 'dev'] else ftest
        if oneword:
            df = df_ow_tag
        else:
            df = df_tag

        for _,row in df.iterrows():
            f.write(" ".joint(row.src) + '\t' + " ".join(row.tgt) + "\n")

    ftrain.close()
    ftest.close()




def write_sequences(outfilename, seqs):
    with open(outfilename, 'w') as outf:
        for seq in seqs:
            outf.write(" ".join(seq[0]) + str(seq[1]).replace('[', '\t').replace(']', '\n').replace(',', ''))
            #outf.write(" ".join(seq[0]) + '\t%s\n'%seq[1][0])
            #outf.write(" ".join(seq[0]) + '\t' + " ".join(seq[1]) + '\n')

#output_sequences(True)
#lex = get_lexicon('/Users/Alex/data/bias-lexicon/bias-lexicon.txt')
#dset = 'test'
#df = filter_edits('/Users/Alex/data/npov-edits/5gram-edits-%s.tsv' %dset, lex)
#print( df)

def get_bias_set(file):
    hedges = set()
    with open("/Users/Alex/Downloads/bias_related_lexicons/%s" % file, 'r') as hedgefile:
        try:
            for l in hedgefile:
                if len(l) > 1 and l[0] not in ("#", ";"):
                    hedges.add(l[:-1])
        except:
            print("error", l)

    print(hedges)
    return hedges

import re

def parse_wiebe():
    strong = set()
    weak = set()

    with open("/Users/Alex/Downloads/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff",
              'r') as hedgefile:
        for l in hedgefile:
            ll = {k: v.strip('"') for k, v in re.findall(r'(\S+)=(".*?"|\S+)', l)}
            if ll['type'] == 'weaksubj':
                weak.add(ll['word1'])
            else:
                strong.add(ll['word1'])

    print(strong)
    print(weak)
    return strong, weak

def add_features():
    lex = get_lexicon('/Users/Alex/data/bias-lexicon/bias-lexicon.txt')
    hedges = get_bias_set("hedges_hyland2005.txt")
    assertives = get_bias_set("assertives_hooper1975.txt")
    factives = get_bias_set("factives_hooper1975.txt")
    implicatives = get_bias_set("implicatives_karttunen1971.txt")
    report_verbvs = get_bias_set("report_verbs.txt")
    strong_subjective, weak_subjective = parse_wiebe()
    negatives = get_bias_set("negative-words.txt")
    positives = get_bias_set("positive-words.txt")

    for dset in ['test', 'train', 'dev']:
        df_class, df_tag, df_ow_tag = filter_edits('/Users/Alex/data/npov-edits/5gram-edits-%s.tsv' %dset, lex)

        srcfile = open("/users/Alex/OpenNMT/data/npov/src-%s-seq-feat-data.txt"%dset, "w")
        tgtfile = open("/users/Alex/OpenNMT/data/npov/tgt-%s-seq-feat-data.txt"%dset, "w")

        srcfile_ow = open("/users/Alex/OpenNMT/data/npov/src-%s-seq-feat-data_ow.txt"%dset, "w")
        tgtfile_ow = open("/users/Alex/OpenNMT/data/npov/tgt-%s-seq-feat-data_ow.txt"%dset, "w")

        mismatch = 0
        matches = 0

        for ind, row in df_tag.iterrows():
            if len(row.src) != len(row.tgt):
                print('unequal lengths ', row.src, row.tgt)
                mismatch += 1

            if len(row.lemmas) != len(row.src):
                print('lemmas wrong length')
                print(row.lemmas)
                print(row.src)
                mismatch += 1

            #seqstc = str(ind) + ' '
            #tgtstc = str(ind) + ' '
            seqstc = ''
            tgtstc = ''

            spaces = 0
            wrds = 0
            for i,wrd in enumerate(row.lemmas):
                if len(row.src) > i:
                    if row.poses[i] == '_SP':
                        spaces += 1
                    else:
                        seqstc += row.src[i] + u"￨"
                        seqstc += wrd + u"￨"
                        seqstc += row.poses[i] + u"￨"
                        seqstc += row.grammar[i] + u"￨"
                        seqstc += (wrd in hedges and '1' or '0') + u"￨"
                        seqstc += (wrd in assertives and '1' or '0') + u"￨"
                        seqstc += (wrd in factives and '1' or '0') + u"￨"
                        seqstc += (wrd in implicatives and '1' or '0') + u"￨"
                        seqstc += (wrd in report_verbvs and '1' or '0') + u"￨"
                        seqstc += (wrd in strong_subjective and '1' or '0') + u"￨"
                        seqstc += (wrd in weak_subjective and '1' or '0') + u"￨"
                        seqstc += (wrd in negatives and '1' or '0') + u"￨"
                        seqstc += (wrd in positives and '1' or '0') + ' '
                        tgtstc += row.tgt[i] + ' '
                        if row.tgt[i] == 'B':
                            wrds += 1
                # else:
                #     if row.poses[i] == '_SP':
                #         spaces += 1
                #         #seqstc += ' '

            srcfile.write(seqstc + '\n')
            tgtfile.write(tgtstc + '\n')
            if wrds == 1:
                srcfile_ow.write(seqstc + '\n')
                tgtfile_ow.write(tgtstc + '\n')

            if (len(tgtstc.split()) != len(seqstc.split())):
                #print('src tgt mismatch',len(tgtstc.split()), len(seqstc.split()), spaces)
                if spaces + len(tgtstc.split()) != len(seqstc.split()):
                    print('mismatch not spaces', seqstc, tgtstc)
            else:
                matches += 1

        print('%d mismatches, %d matches' % (mismatch, matches))
        srcfile.close()
        tgtfile.close()
        srcfile_ow.close()
        tgtfile_ow.close()


def write_featured_seq(df):
    srcfile = open("/users/Alex/data/npov-edits/src-test-seq-feat-data.txt", "w")
    tgtfile = open("/users/Alex/data/npov-edits/tgt-train-seq-feat-data.txt", "w")

    for _,row in df.iterrows():

        srcline = ''
        for ind, wrd in enumerate(row.src):
            srcseq = [wrd,row.lemmas[ind],row.poses[ind],row.grammar[ind],row.hedges[ind],row.assertives[ind],
                      row.factives[ind],row.implicatives[ind],row.report[ind],row.strong[ind],row.weak[ind],
                      row.negative[ind],row.positive[ind]]
            srcline += ("|".join(srcseq) + " ")
        srcline += "\n"

        srcfile.write(srcline)
        tgtfile.write(" ".join(row.tgt))

    srcfile.close()
    tgtfile.close()

add_features()
#write_featured_seq(df)
