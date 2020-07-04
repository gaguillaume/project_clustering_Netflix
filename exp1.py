import glob
import math
import pandas as pd


def train_bimodal_naive_Bayes(list_reviews_train_pos,list_reviews_train_neg):    
    reviews_train_pos=""
    for l in list_reviews_train_pos:
        with open(l) as a:
            reviews_train_pos = reviews_train_pos + a.read()            
    word_train_pos = set(reviews_train_pos.split())
    reviews_train_neg=""
    for l in list_reviews_train_neg:
        with open(l) as a:
            reviews_train_neg = reviews_train_neg + a.read()
    word_train_neg = set(reviews_train_neg.split())
    word_train = set(list(word_train_neg)+list(word_train_pos))
    prior_pos = len(list_reviews_train_pos)/(len(list_reviews_train_pos)+len(list_reviews_train_neg))
    prior_neg = 1 - prior_pos
    logprior = (math.log(prior_pos), math.log(prior_neg))
    tot_weigh_pos = 0
    for l in word_train:
        tot_weigh_pos = tot_weigh_pos + reviews_train_pos.count(l)
##    tot_weigh_pos = 5769021
    tot_weigh_neg = 0
    for l in word_train:
        tot_weigh_neg = tot_weigh_neg + reviews_train_neg.count(l)
##    tot_weigh_neg = 5089215
    loglikehood_pos = {}
    for w in word_train_pos:
        loglikehood_pos[w] = math.log(reviews_train_pos.count(w)/tot_weigh_pos)
    loglikehood_neg = {}
    for w in word_train_neg:
        loglikehood_neg[w] = math.log(reviews_train_neg.count(w)/tot_weigh_neg)
    loglikehood = (loglikehood_pos,loglikehood_neg)
    return logprior, loglikehood

 

def test(text_to_test, logprior, loglikehood):
    logprior_pos,logprior_neg = logprior
    sumlikehood = [logprior_pos,logprior_neg]                                     
    text_review_to_test = ""  
    with open(text_to_test) as  a:
        text_review_to_test = a.read()
    list_text_review_to_test = text_review_to_test.split()
    for word in list_text_review_to_test:
        if word in loglikehood[0]:
            loglikehood_pos_w = loglikehood[0][word]
        else: 
            loglikehood_pos_w = 0
        if word in loglikehood[1]:
            loglikehood_neg_w = loglikehood[1][word]
        else: 
            loglikehood_neg_w = 0
        sumlikehood[0]=sumlikehood[0]+ loglikehood_pos_w
        sumlikehood[1]=sumlikehood[1]+ loglikehood_neg_w
    if sumlikehood[0]>sumlikehood[1]:
        return "positive"#, sumlikehood
    else:
        return "negative"#, sumlikehood

def report(list_reviews_test_pos,list_reviews_test_neg,logprior, loglikehood):
    true_pos, false_pos, true_neg, false_neg = 0,0,0,0
    res_actual = []
    res_expected = []
    for text_pos in list_reviews_test_pos:
        res_actual.append(1)
        if test(text_pos, logprior, loglikehood) == "positive":
            true_pos = true_pos + 1
            res_expected.append(1)
        else:
            false_pos = false_pos + 1
            res_expected.append(0)
    for  text_neg in list_reviews_test_neg:
        res_actual.append(0)
        if test(text_neg, logprior, loglikehood) == "negative":
            true_neg = true_neg + 1
            res_expected.append(0)
        else:
            false_neg = false_pos + 1
            res_expected.append(1)
    precision_pos = true_pos/(true_pos+false_pos)
    recall_pos = true_pos/(true_pos+false_neg)
    accuracy_pos = (true_pos+true_neg)/(true_pos+false_pos+true_neg+false_neg)
    precision_neg = true_neg/(true_neg+false_neg)
    recall_neg = true_neg/(true_neg+false_pos)
    accuracy_neg = (true_neg+true_pos)/(true_neg+false_neg+true_pos+false_pos)   
    return precision_pos, recall_pos, accuracy_pos, precision_neg, recall_neg, accuracy_neg, res_expected, res_actual


# Experiment 1 english


list_reviews_en_train_neg = glob.glob("./datas/movie-reviews-en/train/neg/*.txt")
list_reviews_en_train_pos = glob.glob("./datas/movie-reviews-en/train/pos/*.txt")

logprior, loglikehood = train_bimodal_naive_Bayes(list_reviews_en_train_pos,list_reviews_en_train_neg)    

list_reviews_en_test_neg = glob.glob("./datas/movie-reviews-en/test/neg/*.txt")
list_reviews_en_test_pos = glob.glob("./datas/movie-reviews-en/test/pos/*.txt")


precision_en, recall_en, accuracy_en, precision_neg_en, recall_neg_en, accuracy_neg_en, res_expected_en, res_actual_en = report(list_reviews_en_test_pos,list_reviews_en_test_neg,logprior, loglikehood)



print("Confusion matrix test for english data:")
print("Precison, recall and accuracy for positive class :")
print(precision_en, " ; ", recall_en, "; ", accuracy_en)
print("Precison, recall and accuracy for negative class :")
print(precision_neg_en, " ; ", recall_neg_en, "; ", accuracy_neg_en)

# resultat : 0.33 0.32673267326732675 0.39461883408071746

y_actu_en = pd.Series(res_actual_en, name='Actual')
y_pred_en = pd.Series(res_expected_en, name='Predicted')

print("Confusion matrix test for english data:")
df_confusion_en = pd.crosstab(y_actu_en, y_pred_en)

print(df_confusion_en)


# Experiment 1 french


list_reviews_fr_train_neg = glob.glob("./datas/movie-reviews-fr/train/neg/*.txt")
list_reviews_fr_train_pos = glob.glob("./datas/movie-reviews-fr/train/pos/*.txt")

logprior, loglikehood = train_bimodal_naive_Bayes(list_reviews_fr_train_pos,list_reviews_fr_train_neg)    

list_reviews_fr_test_neg = glob.glob("./datas/movie-reviews-fr/test/neg/*.txt")
list_reviews_fr_test_pos = glob.glob("./datas/movie-reviews-fr/test/pos/*.txt")


precision_fr, recall_fr, accuracy_fr, precision_neg_fr, recall_neg_fr, accuracy_neg_fr, res_expected_fr, res_actual_fr = report(list_reviews_fr_test_pos,list_reviews_fr_test_neg,logprior, loglikehood)



print("Confusion matrix test for french data:")
print("Precison, recall and accuracy for positive class :")
print(precision_fr, " ; ", recall_fr, "; ", accuracy_fr)
print("Precison, recall and accuracy for negative class :")
print(precision_neg_fr, " ; ", recall_neg_fr, "; ", accuracy_neg_fr)

# resultat : 0.61 0.6039603960396039 0.49032258064516127

y_actu_fr = pd.Series(res_actual_fr, name='Actual')
y_pred_fr = pd.Series(res_expected_fr, name='Predicted')

print("Confusion matrix test for french data:")
df_confusion_fr = pd.crosstab(y_actu_fr, y_pred_fr)

print(df_confusion_fr)


