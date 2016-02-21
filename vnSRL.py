import lib
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("input", help="input file name")
parser.add_argument("ilp", help="integer linear programming post-processing")
parser.add_argument("embedding", help="word embedding file")
parser.add_argument("output", help="output file name")
args = parser.parse_args()

corpusFile = args.input
ilp = int(args.ilp)
embedding = args.embedding
outputFileName = args.output

modelFile = 'model.pkl'
encFile = 'enc.pkl'
leFeatureFile = 'leFeature.pkl'
leLabelFile = 'leLabel.pkl'
listFeatureName = ['voice', 'position', 'phrase type', 'function tag']

startTime = datetime.now()

print 'Running Vietnamese Semantic Role Labelling Toolkit'

print 'Loading Parameters'

model, enc, listLE, leLabel = lib.readingParameterFromFile(modelFile, encFile, leFeatureFile,
                                                           leLabelFile)
wordEmbeddingGlove, wordEmbeddingSkipGram = lib.importWordEmbedding()
if embedding == 'skipgram':
    wordEmbedding = wordEmbeddingSkipGram
elif embedding == 'glove':
    wordEmbedding = wordEmbeddingGlove

print 'Reading Data'

corpus = lib.readTestData(corpusFile)
listTag, listWord = lib.convertData(corpus)
listTree = lib.dataToTree(listTag, listWord)
listOfListPredicate = lib.getPredicate(listTree)

listChunkVer = lib.getListChunkVer(listOfListPredicate)

corpus, listTree, listChunkVer, listOfListPredicate = lib.removeSenNoPredicate(corpus, listTree,
                                                                               listChunkVer, listOfListPredicate)

print 'Extracting Arguments'

listTree, listPredicate = lib.chunkingTest(listTree, listOfListPredicate)

listWord = lib.getWord(listTree)

print 'Creating Features'

listFeature, listCount, listTree, listPredicate, setPhraseType, setFunctionType, listWord = \
    lib.getFeatureTest(listTree, listPredicate, listLE, listWord, listChunkVer)

listWordEmbeddingPredicate = lib.createWordEmbeddingPredicateTest(listFeature, wordEmbeddingSkipGram)
listWordEmbeddingHeadword = lib.createWordEmbeddingHeadwordTest(listFeature, wordEmbeddingSkipGram)
listFeatureSVM, listPredicateType = lib.createTestData(listFeature, listWordEmbeddingPredicate,
                                                       listWordEmbeddingHeadword, listFeatureName, enc, listLE)

print 'Running Classifier'

listLabel = lib.semanticRoleClassifier(listFeatureSVM, listCount, model, listPredicateType, ilp)

print 'Writing To File'

lib.output2File(listLabel, listPredicate, listChunkVer, corpus, listCount, listWord, leLabel, outputFileName)

endTime = datetime.now()
print "Running time: "
print (endTime - startTime)
