import nltk 
from nltk4russian.tagger import PMContextTagger
from nltk4russian.util import read_corpus_to_nltk
from nltk.tag.brill import *
import nltk.tag.brill_trainer as bt
import opencorpora

#brill templates
Template._cleartemplates() 
templates = nltk.tag.brill.fntbl37()

#чтение подкорпуса Open Corpora + фильтрация пустых предложений
corpus = opencorpora.CorpusReader('annot.opcorpora.no_ambig.xml')
sents_OC = list(filter(lambda x: len(x), corpus.iter_tagged_sents()))

#чтение подкорпуса НКРЯ (media1.tab, можно подключить другие из папки nltk4russian/data):
with open('media1.tab', encoding='utf-8') as media:
    sents_RNC = list(read_corpus_to_nltk(media))

#чтение подкорпуса LENTA
with open('LENTA_RNC.txt', encoding='utf-8') as LENTA:
    sents1 = list(read_corpus_to_nltk(LENTA))

#чтение подкорпуса VK
with open('VK_RNC.txt', encoding='utf-8') as VK:
    sents2 = list(read_corpus_to_nltk(VK))

#чтение подкорпуса JZ
with open('JZ_RNC.txt', encoding='utf-8') as JZ:
    sents3 = list(read_corpus_to_nltk(JZ))
    
tagger = PMContextTagger(sents_RNC) #выбираем обучающий корпус sents_RNC или sents_OC
tagger = bt.BrillTaggerTrainer(tagger, templates, trace=3)
tagger = tagger.train(sents_RNC, max_rules=400) #задаем max кол-во правил
tagger.print_template_statistics(printunused=False) #таблица статистических параметров

#читаем и разбиваем на токены файл, который размечаем обученным теггером (взят VK_TEST без разметки)
inFile = nltk.word_tokenize(open('VK_TEST.txt', mode='r', encoding='utf-8').read())

#вывод в файл tagged_text.txt размеченного текста
with open('tagged_text.txt', mode='w', encoding='utf-8') as tagged:
    print(tagger.tag(inFile), file=tagged)

#вывод в файл tagger_result.txt результатов оценки на разных подкорпусах, списка правил)
with open('tagger_result.txt', mode='w', encoding='utf-8') as result:
    print('Оценка результатов по выборке LENTA: ', tagger.evaluate(sents1), file=result)
    print('Оценка результатов по выборке VK: ', tagger.evaluate(sents2), file=result)
    print('Оценка результатов по выборке JZ: ', tagger.evaluate(sents3), file=result)
    print('\n', 'Перечень выведенных правил: ', file=result)
    for i in range(len(tagger.rules())):
        print(str(i+1)+'.', tagger.rules()[i], file=result)
    print('Итого (правил): ', len(tagger.rules()), file=result)
