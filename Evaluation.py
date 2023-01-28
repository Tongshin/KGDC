from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from nlgeval import NLGEval
import spacy

class EvaluationrRes():
    def __init__(self, bleu1 = None, bleu2 = None, bleu3 = None, bleu4 = None, distinct1 = None, distinct2 = None, rouge1 = None
                 , rouge2 = None, rougeL = None, meteor = None, AVG = None, EXT = None, GREEDY = None, F1 = None):
        self.bleu1, self.bleu2, self.bleu3, self.bleu4 = bleu1, bleu2, bleu3, bleu4
        self.distinct1, self.distinct2 = distinct1, distinct2
        self.rouge1, self.rouge2, self.rougeL = rouge1, rouge2, rougeL
        self.meteor = meteor
        self.AVG, self.EXT, self.GREEDY = AVG, EXT, GREEDY
        self.F1 = F1
    
    def show_results(self):
        bleu = "BLEU-1: {}\tBLEU-2: {}\tBLEU-3: {}\tBLEU-4: {}".format(self.bleu1, self.bleu2, self.bleu3, self.bleu4)
        dist = "DIST-1: {}\tDIST-2: {}".format(self.distinct1, self.distinct2)
        rouge_ = "ROUGE-1: {}\tROUGE-2: {}\tROUGE:{}".format(self.rouge1, self.rouge2, self.rougeL)
        meteor = "METEOR: {}".format(self.meteor)
        embedding = "AVG: {}\tEXTR:{}\tGREEDY:{}".format(self.AVG, self.EXT, self.GREEDY)
        f1 = "F1: {}".format(self.F1)
        print(bleu)
        print(dist)
        print(rouge_)
        print(meteor)
        print(f1)
        print(embedding)

def calculate_bleu(references, hypothesis, smoothing_function = SmoothingFunction().method1):
    """
    样例生成完毕后，计算所有的样例的bleu值
    :param references: List of lists of references
    :param hypothesis: List of hypothesis
    :return: BLEU1-4 score * 100
    """
    BLEU1 = corpus_bleu(references, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing_function) * 100
    BLEU2 = corpus_bleu(references, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function) * 100
    BLEU3 = corpus_bleu(references, hypothesis, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smoothing_function) * 100
    BLEU4 = corpus_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function) * 100
    return BLEU1, BLEU2, BLEU3, BLEU4

def calculate_distinct(gens):
    '''
    计算生成的回复的distinct-1、distinct-2
    :param gens: list(list(int or str))
    :return: DISTINCT1-2 * 100
    '''
    
    one_grams = []
    for line in gens:
        one_grams.extend(line)
    
    two_grams = []
    
    for gen in gens:
        for i in range(len(gen) - 1):
            #two_grams.append(gen[i:i+2])
            two_grams.append("{}-{}".format(gen[i], gen[i+1]))
    #print(one_grams)
    DISTINCT1 = len(list(set(one_grams))) / len(one_grams) * 100
    DISTINCT2 = len(list(set(two_grams))) / len(two_grams) * 100

    return DISTINCT1, DISTINCT2

def calculate_rouge(references, hypothesis):
    '''
    计算rouge
    :param references: list(str)
    :param hypothesis: list(str)
    :return: ROUGE1-2, ROUGE-L (*100)
    '''
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypothesis, references, avg=True)
    
    ROUGE1 = rouge_scores['rouge-1']['r'] * 100
    ROUGE2 = rouge_scores['rouge-2']['r'] * 100
    ROUGEL = rouge_scores['rouge-l']['r'] * 100

    return ROUGE1, ROUGE2, ROUGEL
'''
def calculate_meteor(hypothesis, references):
    """
    Calculate_meteor
    :param references: List of lists of references
    :param hypothesis: List of hypothesis
    :return: METEOR score
    """
    return meteor_score(references=references, hypothesis = hypothesis)
'''

def calculate_F1(hypothesis, references, url = 'en_core_web_sm', stopwords_file = './stopwords.txt'):
    """
    Calculate F1
    :param references: List of lists of references
    :param hypothesis: List of hypothesis
    :return: F1
    """
    assert len(references) == len(hypothesis)

    nlp = spacy.load(url)

    with open(stopwords_file, 'r', encoding = 'utf8') as f:
        stopwords = set([x.strip() for x in f.readlines()])
    
    hyp_ents = []
    ref_ents = []

    for ref_list in references:

        ents = []

        for ref_sent in ref_list:
            sent_ents = nlp(ref_sent).ents

            ent_text = []

            for ent in sent_ents:
                if ent.text.lower() in stopwords:
                    continue
                ent_text.append(ent.text.lower())
            
            ents.extend(ent_text)
        
        ref_ents.append(ents)
    
    for hyp_sent in hypothesis:
        sent_ents = nlp(hyp_sent).ents
        ent_text = []
        for ent in sent_ents:
            if ent.text.lower() in stopwords:
                continue
            ent_text.append(ent.text.lower())
        
        hyp_ents.append(ent_text)
    
    ref_results = []
    
    for x in ref_ents:
        ref_dic = {}
        for text in x:
            if text not in ref_dic.keys():
                ref_dic[text] = 1
            else:
                ref_dic[text] += 1
        ref_results.append(ref_dic)
    
    hyp_results = []
    for y in hyp_ents:
        hyp_dic = {}
        for text in y:
            if text not in hyp_dic.keys():
                hyp_dic[text] = 1
            else:
                hyp_dic[text] += 1
        hyp_results.append(hyp_dic)
    
    
    sum_A, sum_B, sum_IN = 0, 0, 0
    for i in range(len(hyp_results)):
        sum_A += len(ref_ents[i])
        sum_B += len(hyp_ents[i])

        for key in ref_results[i].keys():
            if key in hyp_results[i].keys():
                #print(key)
                sum_IN += min(ref_results[i][key], hyp_results[i][key])

    if sum_A == 0 or sum_B == 0:
        print("Divide 0! Set F1 = 0.")
        return 0

    P = sum_IN / sum_B
    R = sum_IN / sum_A
    if P + R == 0:
        F1 = "NAN"
    else:
        F1 = 2 * P * R / (P + R)

    return F1

def calculate_meteor_and_embedding_based_metrics(hypothesis, references):
    """
    Calculate_meteor
    :param references: List of lists of references
    :param hypothesis: List of hypothesis
    :return: METEOR score and Embe
    """
    n = NLGEval(no_skipthoughts = True)
    scores = n.compute_metrics(ref_list = references, hyp_list = hypothesis)

    return scores["METEOR"], scores["EmbeddingAverageCosineSimilarity"], scores["VectorExtremaCosineSimilarity"], scores["GreedyMatchingScore"]