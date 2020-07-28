import argparse
from os import path
from glob import glob
from string import punctuation

from tqdm import tqdm
from segtok.tokenizer import web_tokenizer, split_contractions


class Convert(object):
    def __init__(self, datasetdir, listoffilters):
        self.datasetdir = datasetdir
        self.datasetid = self.__get_datasetid__()
        self.lang = self.__get_language__()
        self.filters = self.__get_filters__(listoffilters)

    def build_result(self, results_dir):
        appname = self.__get_appname__(results_dir)
        conversor = self.__get_conversor__(appname)

        ptx = path.join(results_dir, self.datasetid, '*')

        listofresults = glob(ptx)

        toreturn = []
        for resultdoc in tqdm(sorted(listofresults), desc=appname, position=4):
            docid = self.__get_docid__(resultdoc)
            if docid not in self.qrels:
                print('[WARNING] Documento %s not fount in qrels' % docid)
                continue
            gt = self.qrels[docid]
            seen = set()
            result = []
            keyphrases = self.__readfile__(resultdoc).split('\n')
            if len(keyphrases) == 0:
                idkw = 'uk00'
                gt['--'] = (idkw, False)
            else:
                for weight, kw in conversor(keyphrases):
                    kw_key = self.__get_filtered_key__(kw)
                    if kw_key not in gt:
                        idkw = ('uk%d' % len(gt))
                        isrel = False
                        gt[kw] = (idkw, False)
                    else:
                        idkw, isrel = gt[kw_key]
                    if idkw not in seen:
                        seen.add(idkw)
                        result.append(idkw)
            self.qrels[docid] = gt
            toreturn.append((docid, result))
        return (appname, toreturn)

    def save_in_trec_format(self, output_path, appname, results):
        output_file = path.join(output_path,
                                "%s_%s.out" % (self.datasetid, appname))
        with open(output_file, 'w') as outfile:
            for (docid, result) in results:
                for i, instance in enumerate(result):
                    outfile.write("%s Q0 %s %d %d %s\n" %
                                  (docid, instance, (i + 1),
                                   (len(result) - i), appname))

    def save_qrel(self, output_path):
        output_file = path.join(output_path, "%s.qrel" % self.datasetid)
        with open(output_file, 'w') as outfile:
            for docid in self.qrels:
                for (idkw, isrel) in [
                    (idkw, isrel)
                        for (idkw, isrel) in self.qrels[docid].values()
                        if isrel
                ]:
                    outfile.write("%s\t0\t%s\t1\n" % (docid, idkw))

    def build_ground_truth(self):
        keysfiles = glob(path.join(self.datasetdir, 'keys', '*'))
        self.qrels = {}
        for keyfile in tqdm(keysfiles,
                            desc='Building Ground Truth (%s)' % self.datasetid,
                            position=2):
            docid = self.__get_docid__(keyfile)
            gt = {}
            keysunfiltered = self.__readfile__(keyfile).split('\n')
            for goldkey in keysunfiltered:
                gold_key = self.__get_filtered_key__(goldkey)
                if gold_key not in gt:
                    gt[gold_key] = ('k%d' % len(gt), True)
            self.qrels[docid] = gt
        return self.qrels

    # UTILS
    def __get_filters__(self, listoffilters):
        filters = []
        for filter_name in listoffilters:
            if filter_name == 'none':
                filters.append(self.__none_filter__)
            if filter_name == 'stem':
                if self.lang == 'polish':
                    from stems.polishstem import PolishStemmer
                    self.stem = PolishStemmer()
                    filters.append(self.__polish_stem__)
                elif self.lang == 'english':
                    from nltk.stem import PorterStemmer
                    self.stem = PorterStemmer()
                    filters.append(self.__nltk_stem__)
                elif self.lang == 'portuguese':
                    from nltk.stem import RSLPStemmer
                    self.stem = RSLPStemmer()
                    filters.append(self.__nltk_stem__)
                else:
                    from nltk.stem.snowball import SnowballStemmer
                    self.stem = SnowballStemmer(self.lang)
                    filters.append(self.__nltk_stem__)
        return filters

    def __get_filtered_key__(self, key):
        key_filtered = self.__simple_filter__(key)
        for termfilter in self.filters:
            key_filtered = termfilter(key_filtered)
        return key_filtered

    def __get_datasetid__(self):
        return path.split(path.realpath(self.datasetdir))[1]

    def __get_docid__(self, dockeypath):
        return path.basename(dockeypath).replace('.txt', '').replace(
            '.key', '').replace('.out', '').replace('.phrases', '')

    def __readfile__(self, filepath):
        with open(filepath, encoding='utf8') as infile:
            content = infile.read()
        return content

    def __get_language__(self):
        return self.__readfile__(path.join(self.datasetdir,
                                           'language.txt')).replace('\n', '')

    def __get_appname__(self, resultdir):
        return '_'.join([
            config for config in path.dirname(resultdir).split(path.sep)[-2:]
            if config != 'None'
        ])

    # FILTERS
    def __simple_filter__(self, word):
        term = word.lower()
        for p in punctuation:
            term = term.replace(p, ' ')
        term = ' '.join([w for w in split_contractions(web_tokenizer(term))])
        return term.strip()

    def __none_filter__(self, word):
        return word

    def __polish_stem__(self, word):
        return ' '.join(
            self.stem.stemmer_convert(
                [w for w in split_contractions(web_tokenizer(word))]))

    def __nltk_stem__(self, word):
        return ' '.join([
            self.stem.stem(w) for w in split_contractions(web_tokenizer(word))
        ])

    # CONVERSORS
    def __get_conversor__(self, method):
        if method.lower().startswith('rake') or method.lower().startswith(
                'yake') or method.lower().startswith(
                    'ibm') or method.lower().startswith('pke'):
            return self.__sorted_numericList__
        return self.__non_numericList__

    def __non_numericList__(self, listofkeys):
        return [(100. / (1. + i), kw) for i, kw in enumerate(listofkeys)
                if len(kw) > 0]

    def __sorted_numericList__(self, listofkeys):
        toreturn = []
        for key in listofkeys:
            parts = key.rsplit(' ', 1)
            if len(key) > 0 and len(parts) > 1:
                kw, weight = parts
                try:
                    weight = float(weight)
                except:
                    weight = 0.
                toreturn.append((weight, kw))
        return toreturn


parser = argparse.ArgumentParser()
required_args = parser.add_argument_group('required arguments')
required_args.add_argument('-d',
                           '--datasetdir',
                           type=str,
                           nargs='+',
                           help='',
                           required=True)
required_args.add_argument('-r',
                           '--results',
                           type=str,
                           nargs='+',
                           help='',
                           required=True)

parser.add_argument('-o',
                    '--output',
                    type=str,
                    nargs='?',
                    help='Output path.',
                    default='./output/')
parser.add_argument('-f',
                    '--filter',
                    type=str,
                    nargs='+',
                    help='Filter method.',
                    default=['none'],
                    choices=['none', 'stem'])

args = parser.parse_args()
for datasetdir in tqdm(args.datasetdir, position=1, desc='Datasets'):
    conv = Convert(datasetdir, listoffilters=args.filter)
    conv.build_ground_truth()

    for results in tqdm(args.results, position=3, desc=conv.datasetid):
        (appname, results_processed) = conv.build_result(results)
        conv.save_in_trec_format(args.output, appname, results_processed)
    conv.save_qrel(args.output)

# trec_eval -q -c -M1000 official_qrels submitted_results
