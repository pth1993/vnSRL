#!/usr/bin/python
# -*- coding: utf8 -*-
"""
Building a Semantic Role Labelling system for Vietnamese
Python libary
"""

import numpy as np
from ete2 import Tree
import pandas as pd
from pulp import *
import pickle
import codecs
import os

def convertData(listSentence):
    """Get list of words and tags (brackets and syntactic labels)
    """
    listTag = []
    listWord = []
    for record in listSentence:
        temp = []
        recordNew = []
        recordWord = []
        backword = []
        backbackword = []
        for word in record:
            if word == '(':
                if(backword != ')'):
                    recordNew.append(word)
                else:
                    recordNew.pop()
                    recordNew.append(',')
            elif(word == ")"):
                recordNew.append(temp.pop())
                recordNew.append(word)
            else:
                if (backword == '(') or (backword == ')'):
                    temp.append(word)
                elif (backbackword == '(') or (backbackword == ')'):
                    recordWord.append(word)
                else:
                    tempWord = recordWord.pop()
                    tempWord = tempWord + ' ' + word
                    recordWord.append(tempWord)
            backbackword = backword
            backword = word
        listTag.append(recordNew)
        listWord.append(recordWord)
    return listTag, listWord

def dataToTree(listTag, listWord):
    """Get list of bracketed syntactic trees
    Each node's name is a syntactic label.
    Each leaf has a feature named word which stores 1 word of sentence
    """
    listTree = []
    for i in range(len(listTag)):
        tempString = ''.join(listTag[i])
        tempString = tempString + ';'
        tempTree = Tree(tempString, format = 1)
        for leaf in tempTree:
            leaf.add_features(word = listWord[i].pop(0))
        listTree.append(tempTree)
    return listTree

def importWordEmbedding():
    """Get word embedding vector from file
    """
    with open(os.path.join('embedding', 'glove.pkl'), 'rb') as input:
        wordEmbeddingGlove = pickle.load(input)
    with open(os.path.join('embedding', 'skipgram.pkl'), 'rb') as input:
        wordEmbeddingSkipGram = pickle.load(input)
    return wordEmbeddingGlove, wordEmbeddingSkipGram

def isSType(ss):
    """Return True if string "ss" is a sentence label
    False if not
    """
    if len(ss) > 0:
        if ss[0] == 'S':
            return True
    return False

def collect(node):
    """Collect words at leaves of subtree rooted at "node"
    create a constituent and add feature "headWord" which stores head word type.
    """
    leaves = node.get_leaves()
    headWordType = leaves[0].name
    temp = ''
    for leaf in leaves:
        temp += leaf.word + ','
    temp = temp.rstrip(',')
    node.add_features(word = temp)
    node.add_features(headWord = headWordType)
    for child in node.get_children():
        child.detach()

def isPhraseType(ss):
    """Return True if string "ss" is a phrase label
    		  False if not
    """
    specialTag = ['MDP', 'UCP', 'WHAP', 'WHNP', 'WHPP', 'WHRP', 'WHVP', 'WHXP']
    for tag in specialTag:
        if ss.find(tag) != -1:
            return True
    if len(ss) > 1:
        if ss[1] == 'P':
            return True
    return False

def phraseType(tag):
    """Get phrase type
    """
    return tag.split('-')[0]

def process(node):
    """Process creating constituent
    """
    children = node.get_children()
    if len(children) > 1 and isPhraseType(children[0].name):
        same = True
        for child in children:
            if phraseType(child.name) != phraseType(children[0].name):
                same = False
                break
        diff = True
        n = 0
        for child in children:
            if n == 0:
                n += 1
                continue
            if child.name == children[0].name:
                diff = False
        if same and diff:
            for child in children:
                collect(child)
        else:
            collect(node)
    else:
        collect(node)

def reformTag1(ss):
    """Get phrase type. (replacing sentence labels by "S")
    """
    for d in string.digits:
        ss = ss.replace(d, '')
    s = ss.split('-')
    temp = ''.join(s[0])
    if temp[0] == 'S':
        temp = 'S'
    if temp == 'VY-H':
        temp = 'Vy'
    return temp

def getTagFunction(ss):
    """Get function tag of syntactic label.
    """
    for d in string.digits:
        ss = ss.replace(d, '')
    s = ss.split('-')
    if len(s) == 2:
        return s[1]
    elif len(s) > 2:
        if s[2] == '':
            return s[1]
        else:
            if s[2] == 'TPC' or s[2] == 'SUB':
                return s[2]
            return s[1]
    return u'None'

def labelEncoderData(listFeature, listLE):
    """Get lists of features after encoding
    """
    listFeature = np.transpose(listFeature)
    for i in range(len(listFeature)):
        listFeature[i] = listLE[i].transform(listFeature[i])
    listFeature = np.transpose(listFeature)
    return listFeature

def convertToDataFrame(listFeature):
    """Convert to pandas format
    """
    listFeaturePD = pd.DataFrame(listFeature)
    listFeaturePD.columns = ['predicate', 'path tree', 'phrase type', 'position', 'voice', 'head word', 'sub category',
                             'path tree reduce', 'distance', 'head word type', 'function tag', 'predicate type']
    return listFeaturePD

def readTestData(filename):
    """Read data from file and store in a list
    """
    f = codecs.open(os.path.join('data/input', filename), "r", "utf-8")
    corpus = []
    for sent in f.readlines():
        sent = sent.split()
        sentNew = list(sent)
        for i in range(len(sent)):
            if sent[i] in [u'.', u',', u'?', u'!', u':', u'-', u'"', u'...', u';', u'/']:
                sentNew[i-1] = u'vietsrl'
                sentNew[i] = u'vietsrl'
                sentNew[i+1] = u'vietsrl'
                sentNew[i+2] = u'vietsrl'
                sent[i+1] = u'vietsrl'
        sentNew = filter(lambda a: a != u'vietsrl', sentNew)
        corpus.append(sentNew)
    return corpus

def getPredicate(listTree):
    """Get predicate node in tree
    """
    listOfListPredicate = []
    for tree in listTree:
        listPredicateV = tree.search_nodes(name=u'V-H')
        listPredicateA = tree.search_nodes(name=u'A-H')
        listPredicate = listPredicateV + listPredicateA
        listOfListPredicate.append(listPredicate)
    return listOfListPredicate

def chunkingTest(listTree, listOfListPredicate):
    """Get list of tree after applying Constituent Extraction Algorithm
    """
    newListTree = []
    newListPredicate = []
    for i in range(len(listTree)):
        tree = listTree[i]
        idx = 0
        for predicate in listOfListPredicate[i]:
            predicate.add_features(index = idx)
            tempTree = tree.copy()
            currentNode = tempTree.search_nodes(index = idx)[0]
            newListPredicate.append(currentNode)
            currentNode.add_features(rel = True)
            while not isSType(currentNode.name):
                for sister in currentNode.get_sisters():
                    process(sister)
                currentNode = currentNode.up
            newListTree.append(currentNode)
            idx += 1
    return newListTree, newListPredicate

def getPath(node, predicateNode):
    """Get path from argument candidate to predicate
    """
    ancestor = predicateNode.get_common_ancestor(node)
    path = []
    currentNode = node
    while currentNode != ancestor:
        path.append(reformTag1(currentNode.name))
        path.append('1')
        currentNode = currentNode.up
    path.append(reformTag1(ancestor.name))
    temp = []
    currentNode = predicateNode
    while currentNode != ancestor:
        temp.append(reformTag1(currentNode.name))
        temp.append('0')
        currentNode = currentNode.up
    path.extend(temp[::-1])
    return ''.join(path), len(path)/2+1

def getHalfPath(path):
    """Get path from argument candidate to common ancestor
    """
    temp = ''
    for c in path:
        if c != '0':
            temp += c
        else:
            break
    return temp

def getPhraseType(node):
    """Get phrase type of argument candidate
    """
    return reformTag1(node.name)

def getFunctionType(node):
    """Get function tag of argument candidate
    """
    return getTagFunction(node.name)

def getPosition(node, predicateNode):
    """Get position of argument candidate
    Denote 0 if it is before predicate
    Denote 1 if it is after predicate
    """
    for leaf in node.get_leaves():
        if leaf == node:
            return 0
        if leaf == predicateNode:
            return 1

def getVoice(tree):
    """Get voice of sentence
    Denote 0 if it is passive voice
    Denote 1 if it is active voice
    """
    if len(tree.search_nodes(word = u'bị')) > 0:
        for node in tree.search_nodes(word = u'bị'):
            if node.name == 'V-H':
                for sister in node.get_sisters():
                    if sister.name == 'SBAR':
                        return 0
    if len(tree.search_nodes(word = u'được')) > 0:
        for node in tree.search_nodes(word = u'được'):
            if node.name == 'V-H':
                for sister in node.get_sisters():
                    if sister.name == 'SBAR':
                        return 0
    return 1

def getHeadWord(node):
    """Get head word of argument candidate
    """
    return node.word.split(',')[0].strip()

def getHeadWordType(node):
    """Get head word's syntactic label of argument candidate
    """
    return node.headWord

def getSubCategorization(predicateNode):
    """Get minimum subtree that consists predicate
    """
    ancestor = predicateNode.up
    subtree = ancestor.copy()
    for node in subtree.traverse("postorder"):
        node.name = reformTag1(node.name)
    return subtree.write(format = 8)

def getFeatureTest(listTree, listPredicate, listLE, listWord, listChunkVer):
    """Return list of features for each argument candidate
    """
    setPhraseType = list(listLE[2].classes_)
    setFunctionType = list(listLE[3].classes_)
    listFeature = []
    listCount = []
    listWordNew = []
    listChunkVerTemp = []
    for i in range(len(listTree)):
        tree = listTree[i]
        predicate = listPredicate[i]
        words = listWord[i]
        i = 0
        j = 0
        voice = getVoice(tree)
        subcate = getSubCategorization(predicate)
        wordNew = []
        rel = predicate.word
        for leaf in tree:
            if leaf == predicate:
                continue
            feature = []
            phraseType = getPhraseType(leaf)
            functionType = getFunctionType(leaf)
            if(phraseType in setPhraseType) and (functionType in setFunctionType):
                relNew = rel.lower()
                feature.append(relNew)
                path, distance = getPath(leaf, predicate)
                feature.append(path)
                feature.append(phraseType)
                feature.append(getPosition(leaf, predicate))
                feature.append(voice)
                feature.append(getHeadWord(leaf).lower())
                feature.append(subcate)
                feature.append(getHalfPath(path))
                feature.append(distance)
                feature.append(getHeadWordType(leaf).split('-')[0])
                feature.append(functionType)
                feature.append(predicate.name.split('-')[0])
                listFeature.append(feature)
                wordNew.append(words[j])
                i += 1
            j += 1
        if len(wordNew) > 0:
            listWordNew.append(wordNew)
        listCount.append(i)
    for i in range(len(listCount)):
        if listCount[i] == 0:
            listTree[i] = 0
            listPredicate[i] = 0
            listChunkVerTemp.append(0)
        else:
            listChunkVerTemp.append(1)
    count = 0
    for i in range(len(listChunkVer)):
        temp = sum(listChunkVerTemp[count:count+listChunkVer[i]])
        count += listChunkVer[i]
        listChunkVer[i] = temp
    listCount = filter(lambda a: a != 0, listCount)
    listTree = filter(lambda a: a != 0, listTree)
    listPredicate = filter(lambda a: a != 0, listPredicate)
    return listFeature, listCount, listTree, listPredicate, setPhraseType, setFunctionType, listWordNew

def readingParameterFromFile(modelFile, encFile, leFeatureFile, leLabelFile):
    """Read model's parameter and word embedding from file
    """
    with open(os.path.join('model', modelFile), 'rb') as input:
        model = pickle.load(input)
    with open(os.path.join('model', encFile), 'rb') as input:
        enc = pickle.load(input)
    with open(os.path.join('model', leFeatureFile), 'rb') as input:
        listLE = pickle.load(input)
    with open(os.path.join('model', leLabelFile), 'rb') as input:
        leLabel = pickle.load(input)
    return model, enc, listLE, leLabel

def createWordEmbeddingPredicateTest(listFeature, wordEmbedding):
    """Create word embedding for predicate feature
    """
    listWordEmbedding = []
    temp1 = np.asarray(listFeature)
    temp2 = temp1[:,0]
    listError = []
    listVecError = []
    for item in temp2:
        try:
            temp3 = wordEmbedding[item]
        except KeyError:
            listError.append(item)
            temp3 = 0
        listWordEmbedding.append(temp3)
    for listTemp in listError:
        count = 0
        temp4 = [0]*50
        listTemp = listTemp.split()
        for word in listTemp:
            word = word.replace("_", " ")
            try:
                temp5 = wordEmbedding[word]
                temp4 = [temp5+temp4 for temp5,temp4 in zip(temp5, temp4)]
                count += 1
            except KeyError:
                pass
        if(count>0):
            temp4 = np.asarray(temp4)/count
        else:
            temp4 = [0.1]*50
        listVecError.append(temp4)
    k = 0
    for i in range(len(listWordEmbedding)):
        if(listWordEmbedding[i]==0):
            listWordEmbedding[i] = listVecError[k]
            k += 1
    listWordEmbedding = np.asarray(listWordEmbedding)
    return listWordEmbedding

def createWordEmbeddingHeadwordTest(listFeature, wordEmbedding):
    """Create word embedding for head word feature
    """
    listWordEmbedding = []
    temp1 = np.asarray(listFeature)
    temp2 = temp1[:,5]
    listError = []
    listVecError = []
    for item in temp2:
        try:
            temp3 = wordEmbedding[item]
        except KeyError:
            listError.append(item)
            temp3 = 0
        listWordEmbedding.append(temp3)
    for listTemp in listError:
        count = 0
        temp4 = [0]*50
        listTemp = listTemp.split()
        for word in listTemp:
            word = word.replace("_", " ")
            try:
                temp5 = wordEmbedding[word]
                temp4 = [temp5+temp4 for temp5,temp4 in zip(temp5, temp4)]
                count += 1
            except KeyError:
                pass
        if(count>0):
            temp4 = np.asarray(temp4)/count
        else:
            temp4 = [0.1]*50
        listVecError.append(temp4)
    k = 0
    for i in range(len(listWordEmbedding)):
        if(listWordEmbedding[i]==0):
            listWordEmbedding[i] = listVecError[k]
            k += 1
    listWordEmbedding = np.asarray(listWordEmbedding)
    return listWordEmbedding

def createTestData(listFeature, listWordEmbeddingPredicate, listWordEmbeddingHeadword, listFeatureName, enc, listLE):
    """Create data for SVM classifier
    """
    listPredicateType = (np.asarray(listFeature))[:, 11]
    listFeature = convertToDataFrame(listFeature)
    listFeature = listFeature.loc[:,listFeatureName]
    listFeature = np.asarray(listFeature)
    listFeature = labelEncoderData(listFeature, listLE)
    listFeature = listFeature.astype(int)
    listFeatureSVM = enc.transform(listFeature).toarray()
    listFeatureSVM = np.concatenate((listFeatureSVM, listWordEmbeddingPredicate, listWordEmbeddingHeadword), axis=1)
    return listFeatureSVM, listPredicateType

def classificationSVMTest(listFeatureSVM, model):
    """Classify by SVM
    """
    listLabelPredict = model.predict(listFeatureSVM)
    densityMatrix = model.decision_function(listFeatureSVM)
    return listLabelPredict, densityMatrix

def ilpSolving(densityMatrix, predicateType):
    """Integer Linear Programming Solver
    """
    shape = np.shape(densityMatrix)
    if(shape[1]!=27):
        densityMatrix = np.insert(densityMatrix,11,-10,axis=1)
    shape = np.shape(densityMatrix)
    prob = LpProblem("SRL", LpMaximize)
    numItem = shape[0]*shape[1]
    densityList = np.reshape(densityMatrix, numItem)
    index = range(0,numItem)
    #create cost dict
    costs = dict(zip(index, densityList))
    #create constrain 1 dict: each argument can take only one type
    tempDict1 = []
    count = 0
    for i in range(shape[0]):
        tempList = [0]*numItem
        tempList[(count):(count+shape[1])] = [1]*shape[1]
        tempDict = dict(zip(index, tempList))
        tempDict1.append(tempDict)
        count += shape[1]
    #create constrain 2 dict: each type appears only one in sentence
    tempDict2 = []
    for i in range(shape[1]):
        if(i==0 or i==1 or i==2 or i==3 or i==4):
            tempList = [0]*numItem
            for j in range(shape[0]):
                tempList[i+j*shape[1]] = 1
            tempDict = dict(zip(index, tempList))
            tempDict2.append(tempDict)
    if(predicateType!=u'V' and predicateType!=u'VP' and predicateType!=u'Vb'):
        tempDict3 = []
        for i in range(shape[1]):
            if(i==2 or i==3 or i==4):
                tempList = [0]*numItem
                for j in range(shape[0]):
                    tempList[i+j*shape[1]] = 1
                tempDict = dict(zip(index, tempList))
                tempDict3.append(tempDict)
    #create variables
    vars = LpVariable.dicts("Var",index,0,1,LpInteger)
    #objective function
    prob += lpSum([costs[i]*vars[i] for i in index]), "Objective Function"
    #constrain 1
    for j in range(shape[0]):
        prob += lpSum([tempDict1[j][i] * vars[i] for i in index]) == 1.0
    #constrain 2
    for j in range(len(tempDict2)):
        prob += lpSum([tempDict2[j][i] * vars[i] for i in index]) <= 1.0
    if(predicateType!=u'V' and predicateType!=u'VP' and predicateType!=u'Vb'):
        for j in range(len(tempDict3)):
            prob += lpSum([tempDict3[j][i] * vars[i] for i in index]) == 0.0
    #solving
    prob.solve()
    listLabel = []
    listVariable = []
    for v in prob.variables():
        listVariable.append(v.varValue)
        if(v.varValue == 1):
            count = int(v.name[4:])
            listLabel.append(count)
    listLabel = sorted(listLabel)
    listLabel = np.asarray(listLabel)
    listLabel = listLabel%27
    listLabel.tolist()
    return listLabel

def semanticRoleClassifier(listFeatureSVM, listNumArgPerSen, model, listPredicateType, ilp):
    """Label semantic role for each argument candidate
    """
    listLabelPredict, densityMatrix = classificationSVMTest(listFeatureSVM, model)
    listLabelILP = []
    listPredicateType = np.asarray(listPredicateType)
    if ilp == 1:
        count = 0
        for item in listNumArgPerSen:
            predicateType = listPredicateType[count]
            tempMatrix = densityMatrix[count:(count+item), :]
            listLabelTemp = ilpSolving(tempMatrix, predicateType)
            listLabelILP.append(listLabelTemp)
            count += item
    elif ilp == 0:
        count = 0
        for item in listNumArgPerSen:
            listLabelTemp = listLabelPredict[count:(count+item)]
            listLabelILP.append(listLabelTemp)
            count += item
    return listLabelILP

def getWord(listTree):
    """Get word for each leaves node in trees
    """
    listWord = []
    for i in range(len(listTree)):
        leaves = listTree[i].get_leaves()
        if(len(leaves) > 0):
            predicate = listTree[i].search_nodes(rel = True)[0]
            words = []
            for leave in leaves:
                if leave != predicate:
                    word = leave.word.replace(',', ' ')
                    words.append(word)
            listWord.append(words)
    return listWord

def getListChunkVer(listOfListPredicate):
    """Get number of chunk tree version for each tree
    """
    listChunkVer = []
    for item in listOfListPredicate:
        listChunkVer.append(len(item))
    return listChunkVer

def removeSenNoPredicate(corpus, listTree, listChunkVer, listOfListPredicate):
    """Remove sentences that don't have predicate
    """
    for i in range(len(listChunkVer)):
        if listChunkVer[i] == 0:
            corpus[i] = 0
            listTree[i] = 0
    listChunkVer = filter(lambda a: a != 0, listChunkVer)
    listOfListPredicate = filter(lambda a: a != [], listOfListPredicate)
    corpus = filter(lambda a: a != 0, corpus)
    listTree = filter(lambda a: a != 0, listTree)
    return corpus, listTree, listChunkVer, listOfListPredicate

def output2File(listLabel, listPredicate, listChunkVer, corpus, listCount, listWord, leLabel, filename):
    """Writing output of system to file
    """
    f = codecs.open(os.path.join('data/output', filename), "w", "utf-8")
    count1 = 0
    none = leLabel.transform([u'None'])[0]
    for i in range(len(listChunkVer)):
        f.write(' '.join(corpus[i]))
        f.write('\n')
        for j in range(listChunkVer[i]):
            f.write('- Predicate: %s - ' % listPredicate[count1 + j].word)
            for k in range(listCount[count1 + j]):
                if listLabel[count1 + j][k] != none:
                    f.write('%s: ' % leLabel.inverse_transform(listLabel[count1 + j][k]))
                    f.write('%s - ' % listWord[count1 + j][k])
            f.write('\n')
        count1 += listChunkVer[i]
        f.write('\n')
