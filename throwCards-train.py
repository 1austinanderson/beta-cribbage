
# coding: utf-8

# In[1]:


import sys
from collections import namedtuple, Counter
from itertools import combinations, starmap
from numpy import *
import random
import itertools

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import *

import pickle

import numpy as np
import collections
import time
import operator
SUITS = ["1", "2", "3", "4"]
RANKS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]
VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9 , 10, 10, 10, 10]
val_dict = dict(zip(RANKS, VALUES))


#from pathos.multiprocessing import ProcessingPool as Pool
from operator import add


# In[2]:


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def scoreHand(hand):
    _handRanks = [x[:-2] for x in hand]
    _handSuits = [x[-1:] for x in hand]
    #print(_handSuits)
    score = 0

    ## pairs, trips, quads
    duplicates = {x:y for x, y in collections.Counter(_handRanks).items() if y > 1}
    for v in duplicates:
        if duplicates[v] == 2:
            score = score + 2
        elif duplicates[v] == 3:
            score = score + 6
        elif duplicates[v] == 4:
            score = score + 12

    #print("Score from pairs: "+str(score))
    handCombinations = []
    for i in range(2, 5):
        els = [list(x) for x in itertools.combinations(_handRanks, i)]
        handCombinations.extend(els)

    ## 15s
    fifteenScore = 0
    for combo in handCombinations:
        comboScore = 0
        for i in range(0, len(combo)):
            cardValue = val_dict[combo[i]]
            comboScore = comboScore + cardValue

        if comboScore == 15:
            #print("Fifteen for 2: " + str(combo))
            score = score + 2
            fifteenScore = fifteenScore + 2

    #print("Score from 15s: "+str(fifteenScore))

    # flush
    if (sum(x == _handSuits[0] for x in _handSuits))==5:
        score = score + 5
    #    print("flush 5")
    elif (sum(x == _handSuits[0] for x in _handSuits[0:4]))==4:
        score = score + 4
    #    print("flush 4")

    # Nob can be calculated here, but I'm not really sure we need that .. yet?

    ## runs ... oh god why

    numerRanks = [int(x[:-2]) for x in hand]
    
    if sum(numerRanks)*1.0 == ((max(numerRanks)*(max(numerRanks)+1)/2-min(numerRanks)*(min(numerRanks)-1)/2)) and len(_handRanks)==len(set(numerRanks)):
    #    print("run of 5")
        score = score + 5
    else:
        y=0
        for i in itertools.combinations(numerRanks, 4):
            if sum(i)*1.0 == ((max(i)*(max(i)+1)/2-min(i)*(min(i)-1)/2)) and len(i)==len(set(i)):
                #            print("run of 4")
                score = score + 4
                y+=1
        if y==0:
            for i in itertools.combinations(numerRanks, 3):
                if sum(i)*1.0 == ((max(i)*(max(i)+1)/2-min(i)*(min(i)-1)/2)) and len(i)==len(set(i)):
                    #            print("run of 3")
                    score = score + 3
                    y+=1
            

    return score


def index(subseq, seq):
    """Return an index of `subseq`uence in the `seq`uence.

    Or `-1` if `subseq` is not a subsequence of the `seq`.

    The time complexity of the algorithm is O(n*m), where

        n, m = len(seq), len(subseq)

    >>> index([1,2], range(5))
    1
    >>> index(range(1, 6), range(5))
    -1
    >>> index(range(5), range(5))
    0
    >>> index([1,2], [0, 1, 0, 1, 2])
    3
    """
    i, n, m = -1, len(seq), len(subseq)
    try:
        while True:
            i = seq.index(subseq[0], i + 1, n - m + 1)
            if subseq == seq[i:i + m]:
                return i
    except ValueError:
        return -1




# In[3]:


def calcMarginsFromModelMod(_dealtToMe, _theyKept, _theyThrew, _flipCard, _whoseCrib, modelIn):
    
    scores=[]
     
    for t in range(len(_dealtToMe)):
        
        subScore=0
        
        dealtToMe=_dealtToMe[t]
        theyKept=_theyKept[t]
        theyThrew=_theyThrew[t]

        whoseCrib=_whoseCrib[t]
 

        suits = '1 2 3 4'.split()
        ranks = '2 3 4 5 6 7 8 9 10 11 12 13 1'.split()
        deck  = [r + ' ' + s for s in suits for r in ranks]

        for j in range(len(dealtToMe)):
            deck.remove(dealtToMe[j])

        for j in range(len(theyKept)):
            deck.remove(theyKept[j])

        for j in range(len(theyThrew)):
            deck.remove(theyThrew[j])
      


        aiHands = [[] for _ in range(15)]
        aiKeeps = [[] for _ in range(15)]
        aiThrows = [[] for _ in range(15)]
        
        i=0
        for subset in itertools.combinations(dealtToMe, 2):
            y = dealtToMe[:]
            y.remove(subset[0])
            y.remove(subset[1])
            aiKeeps[i]=list(y)
            aiThrows[i]=list(subset)
            aiHands[i]=list(y)+list(subset)
            i+=1

        numericalHands=[]
        for q in range(len(aiHands)):
            hand=aiHands[q]
            numericalHands+=[[whoseCrib]+[int(i.split(' ', 1)[0]) for i in hand]+[int(i.split(' ', 1)[1]) for i in hand] ]
        #print(numericalHands)
        aiouts=modelIn.predict(numericalHands)
        index, value = max(enumerate(aiouts), key=operator.itemgetter(1))

        aiwin=numericalHands[index]

        aiKept=aiKeeps[index]
        aiThrew=aiThrows[index]

        for j in range(len(deck)):
            flipCard=deck[j]
            all1=aiKept+[flipCard]
            all2=theyKept+[flipCard]
            allc=aiThrew+theyThrew+[flipCard]

            aiscore1=scoreHand(all1)
            aiscore2=scoreHand(all2)
            aiscorec=scoreHand(allc)

            subScore+=aiscore1-aiscore2+whoseCrib*aiscorec
        scores+=[subScore/len(deck)]
    return scores
    
    


# In[4]:


def calcMarginsFromSinglePlay(_iKept, _iThrew, _theyKept, _theyThrew, _whoseCrib, modelIn):
    
    score=0

    suits = '1 2 3 4'.split()
    ranks = '2 3 4 5 6 7 8 9 10 11 12 13 1'.split()
    deck  = [r + ' ' + s for s in suits for r in ranks]

    for j in range(len(_iKept)):
        deck.remove(_iKept[j])

    for j in range(len(_iThrew)):
        deck.remove(_iThrew[j])

    for j in range(len(_theyKept)):
        deck.remove(_theyKept[j])

    for j in range(len(_theyThrew)):
        deck.remove(_theyThrew[j])

    for j in range(len(deck)):
        flipCard=deck[j]
        all1=_iKept+[flipCard]
        all2=_theyKept+[flipCard]
        allc=_iThrew+_theyThrew+[flipCard]

        iscore1=scoreHand(all1)
        iscore2=scoreHand(all2)
        iscorec=scoreHand(allc)

        score+=iscore1-iscore2+_whoseCrib*iscorec

    return score/len(deck)
    
    


# In[5]:


def calcMarginsFromModel(_dealtToMe, _theyKept, _theyThrew, _flipCard, _whoseCrib, modelIn):
    
    scores=[]
    for t in range(len(_dealtToMe)):
        
        dealtToMe=_dealtToMe[t]
        theyKept=_theyKept[t]
        theyThrew=_theyThrew[t]
        flipCard=_flipCard[t]
        whoseCrib=_whoseCrib[t]
        
        aiHands = [[] for _ in range(15)]
        aiKeeps = [[] for _ in range(15)]
        aiThrows = [[] for _ in range(15)]
        
        i=0
        for subset in itertools.combinations(dealtToMe, 2):
            y = dealtToMe[:]
            y.remove(subset[0])
            y.remove(subset[1])
            aiKeeps[i]=list(y)
            aiThrows[i]=list(subset)
            aiHands[i]=list(y)+list(subset)
            i+=1

        numericalHands=[]
        for q in range(len(aiHands)):
            hand=aiHands[q]
            numericalHands+=[[whoseCrib]+[int(i.split(' ', 1)[0]) for i in hand]+[int(i.split(' ', 1)[1]) for i in hand] ]
        #print(numericalHands)
        aiouts=modelIn.predict(numericalHands)
        index, value = max(enumerate(aiouts), key=operator.itemgetter(1))

        aiwin=numericalHands[index]

        aiKept=aiKeeps[index]
        aiThrew=aiThrows[index]


        all1=aiKept+[flipCard]
        all2=theyKept+[flipCard]
        allc=aiThrew+theyThrew+[flipCard]

        aiscore1=scoreHand(all1)
        aiscore2=scoreHand(all2)
        aiscorec=scoreHand(allc)

        scores+=[aiscore1-aiscore2+whoseCrib*aiscorec]
    return scores
    
    


# In[6]:


def getScoresEZ(subplay):
    scores=[]
    for i in range(len(subplay['crib'])):
        scores+=[scoreHand(subplay['iKept'][i]+[subplay['flip']])+subplay['whoseCrib']*scoreHand(subplay['crib'][i]+[subplay['flip']])]
    return scores


# In[7]:


def getScoresMod(subplayNew):
    scores=[]
    
    suits = '1 2 3 4'.split()
    ranks = '2 3 4 5 6 7 8 9 10 11 12 13 1'.split()
    deck  = [r + ' ' + s for s in suits for r in ranks]

    for j in range(6):
        deck.remove(subplayNew['dealtToMe'][0][j])

    for j in range(6):
        deck.remove(subplayNew['dealtToThem'][0][j])
    
    for i in range(len(subplayNew['crib'])):
        thisScore=0
        
        


        
        for j in range(len(deck)):
            all1=subplayNew['iKept'][i]+[deck[j]]
            allc=subplayNew['crib'][i]+[deck[j]]
            thisScore+=scoreHand(all1)+subplayNew['whoseCrib']*scoreHand(allc)
        
        scores+=[thisScore/len(deck)]
    return scores


# In[8]:


def toCardText(cardList):
    
    numerizedList=[int(i.split(' ', 1)[0]) for i in cardList]+[int(i.split(' ', 1)[1]) for i in cardList] 

    reada=['K' if x==13 else x for x in numerizedList]
    reada=['Q' if x==12 else x for x in reada]
    reada=['J' if x==11 else x for x in reada]
    reada=['T' if x==10 else x for x in reada]
    reada=['A' if x==1 else x for x in reada]

    readb=['h' if x==1 else x for x in numerizedList]
    readb=['c' if x==2 else x for x in readb]
    readb=['d' if x==3 else x for x in readb]
    readb=['s' if x==4 else x for x in readb]
    
    returnString=""
    for i in range(len(cardList)):
        returnString+=str(reada[i])+str(readb[i+len(cardList)])+" "
    return returnString


# In[9]:


def dealHand(x):
    suits = '1 2 3 4'.split()
    ranks = '2 3 4 5 6 7 8 9 10 11 12 13 1'.split()
    deck  = [r + ' ' + s for s in suits for r in ranks]
    n_players = 2
    hand_size = 6

    random.shuffle(deck)
    #deals = deal(deck, n_players, hand_size)
    deals=[[],[]]
    deals[0] = deck[0:6]
    return(deals[0])


# In[10]:


def dealHands(x):
    suits = '1 2 3 4'.split()
    ranks = '2 3 4 5 6 7 8 9 10 11 12 13 1'.split()
    deck  = [r + ' ' + s for s in suits for r in ranks]
    n_players = 2
    hand_size = 6

    random.shuffle(deck)
    #deals = deal(deck, n_players, hand_size)
    deals=[[],[]]
    deals[0] = deck[0:6]
    deals[1] = deck[6:12]
    flip=deck[13]
    return(deals[0],deals[1],flip)


# In[11]:


def getTrainingData(hand):
    suits = '1 2 3 4'.split()
    ranks = '2 3 4 5 6 7 8 9 10 11 12 13 1'.split()
    deck  = [r + ' ' + s for s in suits for r in ranks]
    n_players = 2
    hand_size = 6

    random.shuffle(deck)
    #deals = deal(deck, n_players, hand_size)
    deals=[[],[]]
    deals[0] = deck[0:6]
    deals[1] = deck[6:12]
    
    whoseCrib=random.choice([1,-1])
    
    
    
    
    
    
    numericalHands=[]
    for i in range(len(subplay['dealt'])):
        hand=subplay['dealt'][i]
        numericalHands+=[[whoseCrib]+[int(i.split(' ', 1)[0]) for i in hand]+[int(i.split(' ', 1)[1]) for i in hand] ]
    return(numericalHands)


# In[12]:


def randomThrow(hand):
    theThrows= random_combination(hand, 2)
    return theThrows


# In[13]:


def makeNumeric(subplay):
    numericalHands=[]
    for i in range(len(subplay['dealtToMe'])):
        hand=subplay['dealtToMe'][i]
        numericalHands+=[[subplay['whoseCrib']]+[int(i.split(' ', 1)[0]) for i in hand]+[int(i.split(' ', 1)[1]) for i in hand] ]
    return(numericalHands)


# In[14]:


def modelThrows(hands):
    a=model.predict(np.array(hands))
    return(a)


# In[15]:


def enumerateHand(hand):
    p1hand = [[] for _ in range(15)]
    p1keep = [[] for _ in range(15)]
    p1throw = [[] for _ in range(15)]
    i=0
    for subset in itertools.combinations(hand, 2):
        y = hand[:]
        y.remove(subset[0])
        y.remove(subset[1])
        p1keep[i]=list(y)
        p1throw[i]=list(subset)
        p1hand[i]=list(y)+list(subset)
        i+=1
    return {'hand': p1hand, 'kept': p1keep, 'thrown':p1throw}


# In[16]:


def getScores(subplay):
    scores=[]
    for i in range(len(subplay['crib'])):
        scores+=[scoreHand(subplay['myhand'][i])+subplay['whoseCrib']*scoreHand(subplay['crib'][i])]
    return scores


# In[17]:


def findBestPlays(predictedScores):
    indexx, value = max(enumerate(predictedScores[0:15]), key=operator.itemgetter(1))
    return indexx


# In[18]:


# slower option I used before: from pathos.multiprocessing import ProcessingPool as Pool
### Training data generation is CPU-bound - on my 16-core server it took 8 hours to fully generate 600k hands
from multiprocessing import Pool
pool=Pool(processes=29)  #use (2 * num_cpus) - 3


# In[19]:


### Load some training data (code to generate below)

### Pregenerated training data: https://drive.google.com/open?id=1yZqy9p2yRDZRciAj5ll5I1QS2VvGggkN
### (~400 MB)

with open('trainingData.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
     train_x, train_y, test_x, test_y = pickle.load(f)


# In[20]:


###Set up the neural network architecture

hands_input = Input(shape=(13,), dtype='float32', name='hands_input')


# We stack a deep densely-connected network on top
x = Dense(240, activation='relu')(hands_input)
x = Dense(240, activation='relu')(x)
x = Dense(240, activation='relu')(x)
x = Dense(240, activation='relu')(x)
x = Dense(120, activation='relu')(x)
x = Dense(120, activation='relu')(x)
x = Dense(120, activation='relu')(x)
x = Dense(120, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

main_output = Dense(1, activation='linear', name='main_output')(x)

modelt = Model(inputs=[hands_input], outputs=[main_output])
modelt.compile(loss='mean_squared_error', optimizer='adam')


# In[21]:


### Load benchmark data - 300 hands I played against the machine

with open('benchmark2.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    dealtToMeList, theyKeptList, theyThrewList, iKeptList, iThrewList, flipCardList, whoseCribList = pickle.load(f)


# In[22]:


### Train Train Train!

### Each epoch is 120 seconds on my 16-core server
### Probably a little/lot faster with a GPU

### Every 4 epochs you can see how it performs on the benchmark data
### For benchmark.pkl human is -100 or so? (didn't save human throws, DOH!)
### For benchmark2.pkl human is -13. 2-sigma is +/- 70 or so

modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[23]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[24]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[25]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[26]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[28]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[29]:


### Saves the neural network weights for use elsewhere

from keras.models import load_model

modelt.save('bestThrowingWeights.h5')


# In[ ]:


###Sets up a smalll set to get the model started
###Plays against random opponent throws - not worth investing too much time

p =pool.map(dealHands, range(20000))
myHands=[]
theirHands=[]
flips=[]
for a in range(len(p)):
    myHands+=[p[a][0]]
    theirHands+=[p[a][1]]
    flips+=[p[a][2]]

allPossibilities=pool.map(enumerateHand, myHands)
inTheCrib=pool.map(randomThrow, theirHands)


subplays=[]
trainingHands=[]
for i in range(len(allPossibilities)):
    subplays+=[
        {
            'dealtToMe': allPossibilities[i]['hand'],
            'iKept': allPossibilities[i]['kept'],
            'crib': [a + b for a, b in zip(allPossibilities[i]['thrown'], 15*[list(inTheCrib[i])])],
            'whoseCrib': random.choice([1,-1]),
            'flip': flips[i]
        }
    ]
    
trainingLabels=pool.map(getScoresEZ,subplays)
trainingData=pool.map(makeNumeric,subplays)

train_y=[j for i in trainingLabels for j in i]
print("y: ",len(train_y))
train_x=[j for i in trainingData for j in i]
print("x: ",len(train_x))


# In[ ]:


hands_input = Input(shape=(13,), dtype='float32', name='hands_input')

#allins = keras.layers.concatenate([hands_input])

# We stack a deep densely-connected network on top
x = Dense(120, activation='relu')(hands_input)
x = Dense(120, activation='relu')(x)
x = Dense(120, activation='relu')(x)
x = Dense(120, activation='relu')(x)
x = Dense(120, activation='relu')(x)
x = Dense(120, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

main_output = Dense(1, activation='linear', name='main_output')(x)

modelt = Model(inputs=[hands_input], outputs=[main_output])
modelt.compile(loss='mean_squared_error', optimizer='adam')
modelt.fit(train_x, train_y, epochs=16, batch_size=1500)


# In[ ]:


### Now generate the real training data. 600k = 23000 sec. Mostly on "getScoresMod"
### It averages across all 40 possible flip cards, which takes a while but removes a lot of variance

### Pregenerated training data: https://drive.google.com/open?id=1yZqy9p2yRDZRciAj5ll5I1QS2VvGggkN
### with open('trainingData.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
###      train_x, train_y, test_x, test_y = pickle.load(f)



numExamples=600000
startTime=time.time()
p=pool.map(dealHands, range(numExamples))
myHands=[]
theirHands=[]
flips=[]
for a in range(len(p)):
    myHands+=[p[a][0]]
    theirHands+=[p[a][1]]
    flips+=[p[a][2]]

allPossibilities=pool.map(enumerateHand, myHands)
theirPossibilities=pool.map(enumerateHand, theirHands)
print("hands enumerated",time.time()-startTime)

opponentPlays=[]
for i in range(len(theirPossibilities)):
    opponentPlays+=[
        {
            'dealtToMe': theirPossibilities[i]['hand'],
            'whoseCrib': random.choice([-1,1])
        }
    ]
print("dict built",time.time()-startTime)

predictThis=pool.map(makeNumeric,opponentPlays)
print("made numeric hands",time.time()-startTime)
predict_x=[j for i in predictThis for j in i]
print("predict ready",time.time()-startTime)


# In[ ]:


outputStuff=modelt.predict(predict_x)
print("predict done",time.time()-startTime)


# In[ ]:


jj=[]
qq=np.split(outputStuff,numExamples)
print("split predictions",time.time()-startTime)
jj=pool.map(findBestPlays, qq)
#apply this to Pool
print("pulled best plays",time.time()-startTime)
bestThrows=[]
for i in range(len(theirPossibilities)):
    bestThrows+=[theirPossibilities[jj[i]]['thrown']]

inTheCrib=[j for i in bestThrows for j in i]
print("crib set",time.time()-startTime)


# In[ ]:


subplays=[]
trainingHands=[]
for i in range(len(allPossibilities)):
    subplays+=[
        {
            'dealtToMe': allPossibilities[i]['hand'],
            'dealtToThem': theirPossibilities[i]['hand'],
            'iKept': allPossibilities[i]['kept'],
            'crib': [a + b for a, b in zip(allPossibilities[i]['thrown'], 15*[list(inTheCrib[i])])],
            'whoseCrib': -1*opponentPlays[i]['whoseCrib'],
            'flip': flips[i]
        }
    ]
    
print("subplays ready",time.time()-startTime)
trainingLabels=pool.map(getScoresMod,subplays)
trainingData=pool.map(makeNumeric,subplays)
print("subplays done",time.time()-startTime)
train_y=[j for i in trainingLabels for j in i]
print("y: ",len(train_y))
train_x=[j for i in trainingData for j in i]
print("x: ",len(train_x))
print("ready to train",time.time()-startTime)


# In[ ]:


### Smaller test set

p=pool.map(dealHands, range(6000))
myHands=[]
theirHands=[]
flips=[]
for a in range(len(p)):
    myHands+=[p[a][0]]
    theirHands+=[p[a][1]]
    flips+=[p[a][2]]
allPossibilities=pool.map(enumerateHand, myHands)
theirPossibilities=pool.map(enumerateHand, theirHands)

opponentPlays=[]
for i in range(len(theirPossibilities)):
    opponentPlays+=[
        {
            'dealtToMe': theirPossibilities[i]['hand'],
            'whoseCrib': random.choice([-1,1])
        }
    ]

predictThis=pool.map(makeNumeric,opponentPlays)
predict_x=[j for i in predictThis for j in i]

outputStuff=modelt.predict(predict_x)

jj=[]
qq=np.split(outputStuff,6000)
jj=pool.map(findBestPlays, qq)
#apply this to Pool
bestThrows=[]
for i in range(len(theirPossibilities)):
    bestThrows+=[theirPossibilities[jj[i]]['thrown']]

inTheCrib=[j for i in bestThrows for j in i]
subplays=[]
trainingHands=[]
for i in range(len(allPossibilities)):
    subplays+=[
        {
            'dealtToMe': allPossibilities[i]['hand'],
            'dealtToThem': theirPossibilities[i]['hand'],
            'iKept': allPossibilities[i]['kept'],
            'crib': [a + b for a, b in zip(allPossibilities[i]['thrown'], 15*[list(inTheCrib[i])])],
            'whoseCrib': -1*opponentPlays[i]['whoseCrib'],
            'flip': flips[i]
        }
    ]
trainingLabels=pool.map(getScoresMod,subplays)
trainingData=pool.map(makeNumeric,subplays)

test_y=[j for i in trainingLabels for j in i]
print("y: ",len(test_y))
test_x=[j for i in trainingData for j in i]
print("x: ",len(test_x))


# In[ ]:


### Shows a few example hands w/ AI choices

for _ in range(10):
    myHand=dealHand(0)
    
    p1hand = [[] for _ in range(15)]
    p1keep = [[] for _ in range(15)]
    p1throw = [[] for _ in range(15)]
    i=0
    for subset in itertools.combinations(myHand, 2):
        y = myHand[:]
        y.remove(subset[0])
        y.remove(subset[1])
        p1keep[i]=list(y)
        p1throw[i]=list(subset)
        p1hand[i]=list(y)+list(subset)
        i+=1
    
    numericalHands=[]
    for i in range(len(p1hand)):
        hand=p1hand[i]
        numericalHands+=[[-1]+[int(i.split(' ', 1)[0]) for i in hand]+[int(i.split(' ', 1)[1]) for i in hand] ]
    #print(numericalHands)
    outs=modelt.predict(numericalHands)
    index, value = max(enumerate(outs), key=operator.itemgetter(1))
    win=numericalHands[index]
    
    read=['K' if x==13 else x for x in win]
    read=['Q' if x==12 else x for x in read]
    read=['J' if x==11 else x for x in read]
    read=['T' if x==10 else x for x in read]
    read=['A' if x==1 else x for x in read]
    
    read2=['h' if x==1 else x for x in win]
    read2=['c' if x==2 else x for x in read2]
    read2=['d' if x==3 else x for x in read2]
    read2=['s' if x==4 else x for x in read2]
    #isMine='(My Crib)'
    #if sing['x_data'][index][0]==-1:
    isMine='(Their Crib)'
    print(isMine +"  "+str(read[1]) +str(read2[7])+" "+ str(read[2])+str(read2[8])+" "+ str(read[3])+str(read2[9])+" "+ str(read[4]) +str(read2[10])+" "+"| " + str(read[5])+str(read2[11])+" "+ str(read[6])+str(read2[12]))


# In[ ]:


crib=1
humanScores=[]
aiScores=[]
dealtToMeList=[]
dealtToThemList=[]
iKeptList=[]
iThrewList=[]
theyKeptList=[]
theyThrewList=[]
flipCardList=[]
whoseCribList=[]
numOfGames=0
suits = '1 2 3 4'.split()
ranks = '2 3 4 5 6 7 8 9 10 11 12 13 1'.split()
deck  = [r + ' ' + s for s in suits for r in ranks]
n_players = 2
hand_size = 6

random.shuffle(deck)
#deals = deal(deck, n_players, hand_size)
dealtToMe = deck[0:6]
crib=crib*-1

print(crib)
print(toCardText(dealtToMe))
print(" 1  2  3  4  5  6")


# In[ ]:



leftOvers=[1,2,3,4,5,6]
leftOvers.remove(throw[0])
leftOvers.remove(throw[1])
leftOvers

iThrew=[dealtToMe[throw[0]-1],dealtToMe[throw[1]-1]]
iKept=[dealtToMe[leftOvers[0]-1],dealtToMe[leftOvers[1]-1],dealtToMe[leftOvers[2]-1],dealtToMe[leftOvers[3]-1]]

iKeptList+=[iThrew]
iThrewList+=[iKept]

dealtToThem = deck[6:12]

theirHands = [[] for _ in range(15)]
theirKeeps = [[] for _ in range(15)]
theirThrows = [[] for _ in range(15)]
i=0
for subset in itertools.combinations(dealtToThem, 2):
    y = dealtToThem[:]
    y.remove(subset[0])
    y.remove(subset[1])
    theirKeeps[i]=list(y)
    theirThrows[i]=list(subset)
    theirHands[i]=list(y)+list(subset)
    i+=1

numericalHands=[]
for q in range(len(theirHands)):
    hand=theirHands[q]
    numericalHands+=[[-1*crib]+[int(i.split(' ', 1)[0]) for i in hand]+[int(i.split(' ', 1)[1]) for i in hand] ]
#print(numericalHands)
outs=modelt.predict(numericalHands)
index, value = max(enumerate(outs), key=operator.itemgetter(1))

win=numericalHands[index]

theyKept=theirKeeps[index]
theyThrew=theirThrows[index]

flipCard=deck[13]

all1=iKept+[flipCard]
all2=theyKept+[flipCard]
allc=iThrew+theyThrew+[flipCard]

score1=scoreHand(all1)
score2=scoreHand(all2)
scorec=scoreHand(allc)

read1=['K' if x==13 else x for x in win]
read1=['Q' if x==12 else x for x in read]
read1=['J' if x==11 else x for x in read]
read1=['T' if x==10 else x for x in read]
read1=['A' if x==1 else x for x in read]

read2=['h' if x==1 else x for x in win]
read2=['c' if x==2 else x for x in read2]
read2=['d' if x==3 else x for x in read2]
read2=['s' if x==4 else x for x in read2]
#isMine='(My Crib)'
#if sing['x_data'][index][0]==-1:
isMine='(Their Crib)'

cribMid=(crib==1)*toCardText(theyThrew)+(crib==-1)*toCardText(iThrew)

print("me:   ",toCardText(iKept)," =",score1)
print("                  ",(crib==1)*toCardText(iThrew))
print("     ",toCardText([flipCard]),"        ",cribMid," =",scorec)
print("                  ",(crib==-1)*toCardText(theyThrew))
print("them:   ",toCardText(theyKept)," =",score2)

numOfGames+=1
huSc=calcMarginsFromSinglePlay(iKept,iThrew,theyKept,theyThrew,crib,modelt)
humanScores+=[huSc]

aiHands = [[] for _ in range(15)]
aiKeeps = [[] for _ in range(15)]
aiThrows = [[] for _ in range(15)]
i=0

for subset in itertools.combinations(dealtToMe, 2):
    y = dealtToMe[:]
    y.remove(subset[0])
    y.remove(subset[1])
    aiKeeps[i]=list(y)
    aiThrows[i]=list(subset)
    aiHands[i]=list(y)+list(subset)
    i+=1

numericalHands=[]
for q in range(len(aiHands)):
    hand=aiHands[q]
    numericalHands+=[[crib]+[int(i.split(' ', 1)[0]) for i in hand]+[int(i.split(' ', 1)[1]) for i in hand] ]
#print(numericalHands)
aiouts=modelt.predict(numericalHands)
index, value = max(enumerate(aiouts), key=operator.itemgetter(1))

aiwin=numericalHands[index]

aiKept=aiKeeps[index]
aiThrew=aiThrows[index]

aiall1=aiKept+[flipCard]
aiall2=theyKept+[flipCard]
aiallc=aiThrew+theyThrew+[flipCard]

aiscore1=scoreHand(aiall1)
aiscore2=scoreHand(aiall2)
aiscorec=scoreHand(aiallc)


aiSc=calcMarginsFromSinglePlay(aiKept,aiThrew,theyKept,theyThrew,crib,modelt)
aiScores+=[aiSc]
dealtToMeList+=[dealtToMe]
dealtToThemList+=[dealtToThem]
theyKeptList+=[theyKept]
theyThrewList+=[theyThrew]
flipCardList+=[flipCard]
whoseCribList+=[crib]

print()
print("     human: ", sum(humanScores),"points ","+/- ",2*sqrt(np.var(humanScores))*sqrt(len(humanScores)))
print("           ",numOfGames, " hands")

print()


print()
print("     ai:    ", sum(aiScores),"points ")
print()
print("             diff=",sum(humanScores)-sum(aiScores)," +/- ",2*sqrt(np.var(np.array(humanScores)-np.array(aiScores)))*sqrt(len(humanScores)))

print()

print(" ")
print(" ")
suits = '1 2 3 4'.split()
ranks = '2 3 4 5 6 7 8 9 10 11 12 13 1'.split()
deck  = [r + ' ' + s for s in suits for r in ranks]
n_players = 2
hand_size = 6

random.shuffle(deck)
#deals = deal(deck, n_players, hand_size)
dealtToMe = deck[0:6]
crib=crib*-1

print(crib)
print(toCardText(dealtToMe))
print(" 1  2  3  4  5  6")


# In[ ]:


# next time save the cards so we can plot performance vs training time
throw=[2,5]


# In[ ]:


lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[ ]:



print(dealtToMeList)
print(theyKeptList)
print(theyThrewList)
print(flipCardList)
print(whoseCribList)
print(iThrewList)
print(iKeptList)


# In[ ]:


print(sum(humanScores[::2])/len(humanScores[::2]))
print(sum(humanScores[1::2])/len(humanScores[1::2]))


# In[ ]:


runningTotal=[]
runningVar=[]
runningMarHi=[]
runningMarLo=[]
zippo=[]
for i in range(1,len(humanScores)):
    runningTotal+=[sum(humanScores[:i])-sum(aiScores[:i])]
    runningVar+=[2*sqrt(np.var(np.array(humanScores[:i])-np.array(aiScores[:i])))*sqrt(len(humanScores[:i]))]
    runningMarHi+=[sum(humanScores[:i])-sum(aiScores[:i])+2*sqrt(np.var(np.array(humanScores[:i])-np.array(aiScores[:i])))*sqrt(len(humanScores[:i]))]
    runningMarLo+=[sum(humanScores[:i])-sum(aiScores[:i])-2*sqrt(np.var(np.array(humanScores[:i])-np.array(aiScores[:i])))*sqrt(len(humanScores[:i]))]                                                                                   
    zippo+=[0]


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(len(zippo)),zippo)
a=plt.plot(range(len(runningMarHi)),runningMarHi,'r')
b=plt.plot(range(len(runningTotal)),runningTotal,'b')
c=plt.plot(range(len(runningMarLo)),runningMarLo,'r')

plt.fill_between(range(len(runningMarLo)), runningMarLo, runningMarHi, color='red', alpha='0.5')

red_patch = mpatches.Patch(color='blue', label='Human minus AI scores')
blue_patch = mpatches.Patch(color='red', label='95% confidence')

plt.legend(handles=[red_patch, blue_patch],loc=3)
plt.title('Beta Cribbage performance')
plt.xlabel('cribbage hands')



plt.show()


# In[ ]:


2*sqrt(np.var(np.array(humanScores[:i])-np.array(aiScores[:i])))


# In[ ]:


print(aiThrows)


# In[ ]:


asd[5:]


# In[ ]:


print(humanScores)
print(aiScores)


# In[ ]:


### How to train the model


# In[ ]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])


# In[ ]:


lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)


# In[ ]:


sum(lis)


# In[ ]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[ ]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[ ]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[ ]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[ ]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[ ]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[ ]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[ ]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[ ]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[ ]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[ ]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[ ]:


modelt.fit(train_x, train_y, epochs=4, batch_size=2048,validation_data=[test_x,test_y])
lis=calcMarginsFromModelMod(dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList, modelt)
sum(lis)


# In[ ]:


subplays=[]
trainingHands=[]
for i in range(len(allPossibilities)):
    subplays+=[
        {
            'dealtToMe': allPossibilities[i]['hand'],
            'dealtToThem': theirPossibilities[i]['hand'],
            'iKept': allPossibilities[i]['kept'],
            'crib': [a + b for a, b in zip(allPossibilities[i]['thrown'], 15*[list(inTheCrib[i])])],
            'whoseCrib': -1*opponentPlays[i]['whoseCrib']
        }
    ]


# In[ ]:



trainingLabels=pool.map(getScoresMod,subplays)


# In[ ]:


print()
print("     human: ", sum(humanScores),"points ","+/- ",2*sqrt(np.var(humanScores))*sqrt(len(humanScores)))
print("           ",numOfGames, " hands")

print()


print()
print("     ai:    ", sum(aiScores),"points ")
print()
print("             diff=",sum(humanScores)-sum(aiScores)," +/- ",2*sqrt(np.var(np.array(humanScores)-np.array(aiScores)))*sqrt(len(humanScores)))

print()



# In[ ]:


with open('trainingData.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([train_x, train_y, test_x, test_y], f)


# In[ ]:


with open('trainingData.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    train_x, train_y, test_x, test_y = pickle.load(f)



# In[ ]:


with open('benchmark2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([dealtToMeList, theyKeptList, theyThrewList, iKeptList, iThrewList, flipCardList, whoseCribList], f)


# In[ ]:


with open('benchmark2.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    dealtToMeList, theyKeptList, theyThrewList, iKeptList, iThrewList, flipCardList, whoseCribList = pickle.load(f)


# In[ ]:


with open('benchmark.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    dealtToMeList, theyKeptList, theyThrewList, flipCardList, whoseCribList = pickle.load(f)


# In[ ]:


dealtToMeList[66]


# In[ ]:


from keras.models import load_model

modelt.save('bestThrowingWeights.h5')  # creates a HDF5 file 'my_model.h5'

