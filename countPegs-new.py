
# coding: utf-8

# In[1]:


### Haven't touched this model in a while
### Might not execute right!

import numpy as np
from multiprocessing import Pool
#from pathos.multiprocessing import ProcessingPool as Pool
import random
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
import time

import os
os.system("taskset -p 0xffffffff %d" % os.getpid())


# In[2]:


os.sched_getaffinity(0)


# In[3]:


def pegScore(plays,who):
    thisCard=0
    if len(plays)==1:
        #print('1')
        return 0
    if plays[-2:]==[0,0]:
        #print("go",who[-1])
        theGo=who[-1]
        if sum(plays)>=31:
            theGo=0
        return pegScore(plays[:-1],who[:-1])+theGo
    playruns=list(filter(lambda a: a != 0, plays[:-1]))+[plays[-1]]
    playsums=[min(x,10) for x in plays]
    playsums=list(filter(lambda a: a != 0, playsums[:-1]))+[playsums[-1]]
    #print(len(playsums))
    #print(plays)
    #print(who)
    if len(playsums)==2 and playsums[-1]>0:
        #print('2')
        if playruns[-1]==playruns[-2]:
            thisCard+=2
        if(sum(playsums)==15):
            thisCard+=2
    elif len(playsums)==3 and playsums[-1]>0:
        #print('3')
        if playruns[-1]==playruns[-2]==playruns[-3]:
            thisCard+=6
        elif playruns[-1]==playruns[-2]:
            thisCard+=2
        if sum(playruns)*1.0 == ((max(playruns)*(max(playruns)+1)/2-min(playruns)*(min(playruns)-1)/2)) and len(playruns)==len(set(playruns)):
            thisCard+=3
        if(sum(playsums)==15):
            thisCard+=2
    elif len(playsums)>=4 and playsums[-1]>0:
        #print('4')
        if playruns[-1]==playruns[-2]==playruns[-3]==playruns[-4]:
            thisCard+=12
        elif playruns[-1]==playruns[-2]==playruns[-3]:
            thisCard+=6
        elif playruns[-1]==playruns[-2]:
            thisCard+=2
        if sum(playruns[-7:])*1.0 == ((max(playruns[-7:])*(max(playruns[-7:])+1)/2-min(playruns[-7:])*(min(playruns[-7:])-1)/2)) and len(playruns[-7:])==len(set(playruns[-7:]))==7:
            thisCard+=7
        elif sum(playruns[-6:])*1.0 == ((max(playruns[-6:])*(max(playruns[-6:])+1)/2-min(playruns[-6:])*(min(playruns[-6:])-1)/2)) and len(playruns[-6:])==len(set(playruns[-6:]))==6:
            thisCard+=6
        elif sum(playruns[-5:])*1.0 == ((max(playruns[-5:])*(max(playruns[-5:])+1)/2-min(playruns[-5:])*(min(playruns[-5:])-1)/2)) and len(playruns[-5:])==len(set(playruns[-5:]))==5:
            thisCard+=5
        elif sum(playruns[-4:])*1.0 == ((max(playruns[-4:])*(max(playruns[-4:])+1)/2-min(playruns[-4:])*(min(playruns[-4:])-1)/2)) and len(playruns[-4:])==len(set(playruns[-4:]))==4:
            thisCard+=4
        elif sum(playruns[-3:])*1.0 == ((max(playruns[-3:])*(max(playruns[-3:])+1)/2-min(playruns[-3:])*(min(playruns[-3:])-1)/2)) and len(playruns[-3:])==len(set(playruns[-3:]))==3:
            thisCard+=3
        if(sum(playsums)==15):
            thisCard+=2
        if(sum(playsums)==31):
            thisCard+=2
    #print("outp",plays)
    #print("outw",who)
    #print("this",who[-1]*thisCard)
    return pegScore(plays[:-1],who[:-1])+who[-1]*thisCard


# In[4]:


def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(10) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.full(mask.shape, -1, dtype="float32")
    out[mask] = np.concatenate(data)
    return out


# In[5]:


class GameNode(object):
    "Generic tree node."
    def __init__(self, hand1,hand2,plays,whos,nextPlay,nodeID):
        self.nodeID = nodeID
        self.hand1 = hand1
        self.hand2 = hand2
        self.plays = plays
        self.whos = whos
        self.nextPlay = nextPlay
        self.children = []
        self.runningCount=sum(np.minimum(plays,10).tolist())
        self.isDone = 0
        self.isPredicted = 0
        self.isExpanded = 0
        self.score = -999
        self.predScore = -999
    def __repr__(self):
        return "GameNode object "+str(self.nodeID)+" predScore:"+str(self.predScore)
    
    def expand(self):
        if len(self.plays)>1:
            if self.plays[-2:]==[0,0]:
                self.isExpanded = 1
                return 0
        if len(self.plays)>12:
            self.isExpanded = 1
            return "whoa!"
        if self.nextPlay==1:
            newhand1=[]
            newhand2=[]
            newplays=[]
            newwhos=[]
            for i in range(len(self.hand1)):
                nextCard=self.hand1[i]
                if nextCard!=0 and self.runningCount+min(10,nextCard)<=31:
                    h1=self.hand1[:]
                    newplays+=[self.plays+[nextCard]]
                    newwhos+=[self.whos+[1]]
                    h1[i]=0
                    newhand1+=[h1]
                    newhand2+=[self.hand2]
            if len(newhand1)==0:
                newplays+=[self.plays+[0]]
                newwhos+=[self.whos+[1]]
                newhand1+=[self.hand1]
                newhand2+=[self.hand2]
            for i in range(len(newhand1)):
                self.children+=[GameNode(newhand1[i],newhand2[i],newplays[i],newwhos[i],-1,self.nodeID*10+i+1)]
        else:
            newhand1=[]
            newhand2=[]
            newplays=[]
            newwhos=[]
            for i in range(len(self.hand2)):
                nextCard=self.hand2[i]
                if nextCard!=0 and self.runningCount+min(10,nextCard)<=31:
                    h2=self.hand2[:]
                    newplays+=[self.plays+[nextCard]]
                    newwhos+=[self.whos+[-1]]
                    h2[i]=0
                    newhand1+=[self.hand1]
                    newhand2+=[h2]
            if len(newhand1)==0:
                newplays+=[self.plays+[0]]
                newwhos+=[self.whos+[-1]]
                newhand1+=[self.hand1]
                newhand2+=[self.hand2]
            for i in range(len(newhand1)):
                self.children+=[GameNode(newhand1[i],newhand2[i],newplays[i],newwhos[i],1,self.nodeID*10+i+1)]
        self.isExpanded = 1
        for i in range(len(self.children)):
            self.children[i].expand()
        return "expanded"+str(self.plays)
    
    def updatePredScore(self,returnData):
        if self.nodeID in returnData:
            self.predScore=returnData[self.nodeID]
        for i in self.children:
            i.updatePredScore(returnData)
    
    def getAllPossibilities(self):
        allHand1=[]
        allHand2=[]
        allWhos=[]
        allPlays=[]
        for i in range(len(self.children)):
            childsPlay=self.children[i].getAllPossibilities()
            for j in range(len(childsPlay['hand1'])):
                allHand1+=[childsPlay['hand1'][j]]
                allHand2+=[childsPlay['hand2'][j]]
                allWhos+=[childsPlay['whos'][j]]
                allPlays+=[childsPlay['plays'][j]]
        allHand1+=[self.hand1]
        allHand2+=[self.hand2]
        allWhos+=[self.whos]
        allPlays+=[self.plays]
        return {'hand1': allHand1, 'hand2': allHand2, 'whos': allWhos, 'plays': allPlays}

    def getNaiveScore(self):
        if self.score > -999:
            return(self.score)
        return(self.calcNaiveScore())


    def calcNaiveScore(self):
        if len(self.plays)>1:
            if self.plays[-2:]==[0,0]:
                self.score=pegScore(self.plays,self.whos)
                self.isScored = 1
                return(self.score)
        for child in self.children:
            kidScore=child.getNaiveScore()
            if kidScore>self.score:
                self.score=kidScore
        return(self.score)

    
    def getSmartScore(self):
        if self.score > -999:
            return(self.score)
        return(self.calcSmartScore())


    def calcSmartScore(self):
        if len(self.plays)>1:
            if self.plays[-2:]==[0,0]:
                self.score=pegScore(self.plays,self.whos)
                self.isScored = 1
                return(self.score)
        if self.nextPlay==-1 and len(self.children)>1:
            bestGuessIndex=0
            bestGuessValue=-999
            for i in range(len(self.children)):
                if self.children[i].predScore>bestGuessValue:
                    bestGuessValue=self.children[i].predScore
                    bestGuessIndex=i
            self.score=self.children[bestGuessIndex].getSmartScore()
        else:
            for child in self.children:
                kidScore=child.getSmartScore()
                if kidScore>self.score:
                    self.score=kidScore
        return(self.score)
    
    
    def getAllNaiveScores(self):
        allNaiveScores=[]
        for i in range(len(self.children)):
            childsPlay=self.children[i].getAllNaiveScores()
            for j in range(len(childsPlay['naiveScores'])):
                allNaiveScores+=[childsPlay['naiveScores'][j]]
        allNaiveScores+=[self.score]
        return {'naiveScores': allNaiveScores}
    
    def getAllSmartScores(self):
        allSmartScores=[]
        for i in range(len(self.children)):
            childsPlay=self.children[i].getAllSmartScores()
            for j in range(len(childsPlay['smartScores'])):
                allSmartScores+=[childsPlay['smartScores'][j]]
        if(self.nextPlay==1 and len(self.children)>1):
            for i in range(len(self.children)):
                allSmartScores+=[self.children[i].getSmartScore()]
        return {'smartScores': allSmartScores}
    
    def getOwnDecisions(self):
        allHand1=[]
        allHand2=[]
        allWhos=[]
        allPlays=[]
        allNodeIDs=[]
        for i in range(len(self.children)):
            childsPlay=self.children[i].getOwnDecisions()
            for j in range(len(childsPlay['hand1'])):
                allHand1+=[childsPlay['hand1'][j]]
                allHand2+=[childsPlay['hand2'][j]]
                allWhos+=[childsPlay['whos'][j]]
                allPlays+=[childsPlay['plays'][j]]
                allNodeIDs+=[childsPlay['nodeIDs'][j]]
        if(self.nextPlay==1 and len(self.children)>1):
            for i in range(len(self.children)):
                allHand1+=[self.children[i].hand1]
                allHand2+=[self.children[i].hand2]
                allWhos+=[self.children[i].whos]
                allPlays+=[self.children[i].plays]
                allNodeIDs+=[self.children[i].nodeID]
        #elif(self.nextPlay==-1 and len(self.children)>1):
            #maxIndex=0
            #bestValue=-999
            #for i in range(len(self.children)):
            #    if self.children[i].predScore > bestValue:
            #        maxIndex=i
            #allHand1+=[self.children[maxIndex].hand1]
            #allHand2+=[self.children[maxIndex].hand2]
            #allWhos+=[self.children[maxIndex].whos]
            #allPlays+=[self.children[maxIndex].plays]
            #allNodeIDs+=[self.children[maxIndex].nodeID]
        #elif self.nextPlay==1:
        #    for i in range(len(self.children)):
        #        allHand1+=[self.children[i].hand1]
        #        allHand2+=[self.children[i].hand2]
        #        allWhos+=[self.children[i].whos]
        #        allPlays+=[self.children[i].plays]
        #        allNodeIDs+=[self.children[i].nodeID]
        return {'hand1': allHand1, 'hand2': allHand2, 'whos': allWhos, 'plays': allPlays, 'nodeIDs': allNodeIDs}


    
    def getOpponentDecisions(self):
        allHand1=[]
        allHand2=[]
        allWhos=[]
        allPlays=[]
        allNodeIDs=[]
        for i in range(len(self.children)):
            childsPlay=self.children[i].getOpponentDecisions()
            for j in range(len(childsPlay['hand1'])):
                allHand1+=[childsPlay['hand1'][j]]
                allHand2+=[childsPlay['hand2'][j]]
                allWhos+=[childsPlay['whos'][j]]
                allPlays+=[childsPlay['plays'][j]]
                allNodeIDs+=[childsPlay['nodeIDs'][j]]
        if(self.nextPlay==-1 and len(self.children)>1):
            for i in range(len(self.children)):
                allHand1+=[self.children[i].hand1]
                allHand2+=[self.children[i].hand2]
                allWhos+=[self.children[i].whos]
                allPlays+=[self.children[i].plays]
                allNodeIDs+=[self.children[i].nodeID]
        return {'hand1': allHand1, 'hand2': allHand2, 'whos': allWhos, 'plays': allPlays, 'nodeIDs': allNodeIDs}

    def getGamePlays(self):
        if len(self.children)>1:
            bestGuessIndex=0
            bestGuessValue=-999
            for i in range(len(self.children)):
                if self.children[i].predScore>bestGuessValue:
                    bestGuessValue=self.children[i].predScore
                    bestGuessIndex=i
            return self.children[bestGuessIndex].getGamePlays()
        elif len(self.children)==1:
            return self.children[0].getGamePlays()
        else:
            return(self.plays)
 


# In[6]:


def newGame(x):
    
    #os.system("taskset -p 0xffffffff %d" % os.getpid())
    suits = '1 2 3 4'.split()
    ranks = '2 3 4 5 6 7 8 9 10 11 12 13 1'.split()
    deck  = [r + ' ' + s for s in suits for r in ranks]
    n_players = 2
    hand_size = 6

    random.shuffle(deck)
    #deals = deal(deck, n_players, hand_size)
    deals=[[],[]]
    deals[0] = list(map(int,[x[:-2] for x in deck[0:4]]))
    deals[1] = list(map(int,[x[:-2] for x in deck[4:8]]))

    return GameNode(deals[0],deals[1],[],[],random.choice([1,-1]),1)


# In[7]:


def fillInNaive(gameNode):
    #os.system("taskset -p 0xffffffff %d" % os.getpid())
    gameNode.expand()
    gameNode.calcNaiveScore()
    #print("done fill in - ", os.getpid())
    return gameNode


# In[8]:


def fillInSmart(gameNode):
    #os.system("taskset -p 0xffffffff %d" % os.getpid())
    gameNode.expand()
    #print("done fill in - ", os.getpid())
    return gameNode


# In[9]:


def getAll(gameNode):
    #os.system("taskset -p 0xffffffff %d" % os.getpid())
    alls=gameNode.getAllPossibilities()
    #print("done alls - ", os.getpid())
    return(alls)


# In[10]:


def getNaives(gameNode):
    #os.system("taskset -p 0xffffffff %d" % os.getpid())
    naives=gameNode.getAllNaiveScores()
    #print("done naives - ", os.getpid())
    return(naives)


# In[11]:


def getSmarts(inputDict):
    #os.system("taskset -p 0xffffffff %d" % os.getpid())
    gameNode=inputDict['node']
    gameNode.expand()
    gameNode.updatePredScore(inputDict['update'])
    gameNode.calcSmartScore()
    smarts=gameNode.getAllSmartScores()
    #print("done smarts - ", os.getpid())
    return(smarts)


# In[12]:


def getOpponents(gameNode):
    #os.system("taskset -p 0xffffffff %d" % os.getpid())
    gameNode.expand()
    opponents=gameNode.getOpponentDecisions()
    #print("done opponents - ", os.getpid())
    return(opponents)


# In[13]:


def getOwns(inputDict):
    #os.system("taskset -p 0xffffffff %d" % os.getpid())
    gameNode=inputDict['node']
    gameNode.expand()
    gameNode.updatePredScore(inputDict['update'])
    gameNode.calcSmartScore()
    owns=gameNode.getOwnDecisions()
    #print("done owns - ", os.getpid())
    return(owns)


# In[14]:


def updateOpponent(inputDict):
    #os.system("taskset -p 0xffffffff %d" % os.getpid())
    gameNode=inputDict['node']
    gameNode.updatePredScore(inputDict['update'])
    gameNode.calcSmartScore()
    #print("done update - ", os.getpid())
    return gameNode


# In[15]:


pool=Pool(processes=24)


# In[16]:


myHands=pool.map(newGame, range(400))

filledInNaive=pool.map(fillInNaive, myHands)

train_x_dict=pool.map(getAll, filledInNaive)
train_y_dict=pool.map(getNaives, filledInNaive)

train_xh1=[]
train_xp=[]
train_xw=[]

for i in range(len(train_x_dict)):
    train_xh1+=train_x_dict[i]['hand1']
    train_xp+=train_x_dict[i]['plays']
    train_xw+=train_x_dict[i]['whos']

train_y=[]

for i in range(len(train_y_dict)):
    train_y+=train_y_dict[i]['naiveScores']


# In[17]:


myHands=pool.map(newGame, range(100))


test_x_dict=pool.map(getAll, filledInNaive)
test_y_dict=pool.map(getNaives, filledInNaive)

test_xh1=[]
test_xp=[]
test_xw=[]

for i in range(len(test_x_dict)):
    test_xh1+=test_x_dict[i]['hand1']
    test_xp+=test_x_dict[i]['plays']
    test_xw+=test_x_dict[i]['whos']

test_y=[]

for i in range(len(test_y_dict)):
    test_y+=test_y_dict[i]['naiveScores']


# In[18]:



hand_input = Input(shape=(4,), dtype='float32', name='hand_input')
plays_input = Input(shape=(10,), dtype='float32', name='plays_input')
whos_input = Input(shape=(10,), dtype='float32', name='whos_input')

allins = keras.layers.concatenate([hand_input, plays_input, whos_input])

# We stack a deep densely-connected network on top
x = Dense(120, activation='relu')(allins)
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

model = Model(inputs=[hand_input, plays_input, whos_input], outputs=[main_output])


# In[19]:


model.compile(optimizer='adam', loss='mean_squared_error')
datadict= {"train_x": [np.array(train_xh1),numpy_fillna(train_xp),numpy_fillna(train_xw)], "train_y": [np.array(train_y)], "test_set": [[np.array(test_xh1),numpy_fillna(test_xp),numpy_fillna(test_xw)], [np.array(test_y)]] }


# In[20]:


model.fit(datadict["train_x"], datadict["train_y"],validation_data=datadict["test_set"],epochs=8, batch_size=1500)


# In[ ]:


startTime = time.time()
myHands=pool.map(newGame, range(80000))
print("milestone",time.time()-startTime)
#filledInSmart=pool.map(fillInSmart, myHands)
#print("milestone",time.time()-startTime)
opponentDecisions=pool.map(getOpponents, myHands)
print("milestone",time.time()-startTime)


# In[ ]:


predict_xh1=opponentDecisions[0]['hand2']
predict_xp=opponentDecisions[0]['plays']
predict_xw=opponentDecisions[0]['whos']
predict_treelens=[len(opponentDecisions[0]['whos'])]

for i in range(1,len(opponentDecisions)):
    predict_xh1+=opponentDecisions[i]['hand2']
    predict_xp+=opponentDecisions[i]['plays']
    predict_xw+=opponentDecisions[i]['whos']
    predict_treelens+=[predict_treelens[-1]+len(opponentDecisions[i]['whos'])]

print("milestone",time.time()-startTime)
# fix with the for j in i in x for i in j thing


# In[ ]:


opponentBests=model.predict([np.array(predict_xh1),numpy_fillna(predict_xp),-1*numpy_fillna(predict_xw)])
print("milestone",time.time()-startTime)


# In[ ]:


resultsByTree=np.split(np.squeeze(opponentBests), predict_treelens)
print("milestone",time.time()-startTime)


# In[ ]:


feedbackForNodes=[]
for i in range(len(opponentDecisions)):
    feedbackForNodes+=[dict(zip(opponentDecisions[i]['nodeIDs'], resultsByTree[i]))]
print("milestone",time.time()-startTime)


# In[ ]:


toPool=[]
for i in range(len(opponentDecisions)):
    toPool+=[{'node': myHands[i], 'update': feedbackForNodes[i]}]
print("milestone",time.time()-startTime)
#smartOpponent=pool.map(updateOpponent, toPool)
#print("milestone",time.time()-startTime)


# In[ ]:


train_x_dict=pool.map(getOwns, toPool)
print("milestone",time.time()-startTime)
train_y_dict=pool.map(getSmarts, toPool)
print("milestone",time.time()-startTime)

train_xh1=[]
train_xp=[]
train_xw=[]

for i in range(len(train_x_dict)):
    train_xh1+=train_x_dict[i]['hand1']
    train_xp+=train_x_dict[i]['plays']
    train_xw+=train_x_dict[i]['whos']

print("milestone",time.time()-startTime)
train_y=[]

for i in range(len(train_y_dict)):
    train_y+=train_y_dict[i]['smartScores']
print("milestone",time.time()-startTime)


# In[ ]:


myHands=pool.map(newGame, range(3000))
print("milestone",time.time()-startTime)
#filledInNaive=pool.map(fillInNaive, myHands)
#print("milestone",time.time()-startTime)
opponentDecisions=pool.map(getOpponents, myHands)
print("milestone",time.time()-startTime)


# In[ ]:


predict_xh1=opponentDecisions[0]['hand2']
predict_xp=opponentDecisions[0]['plays']
predict_xw=opponentDecisions[0]['whos']
predict_treelens=[len(opponentDecisions[0]['whos'])]

for i in range(1,len(opponentDecisions)):
    predict_xh1+=opponentDecisions[i]['hand2']
    predict_xp+=opponentDecisions[i]['plays']
    predict_xw+=opponentDecisions[i]['whos']
    predict_treelens+=[predict_treelens[-1]+len(opponentDecisions[i]['whos'])]
    
print("milestone",time.time()-startTime)
opponentBests=model.predict([np.array(predict_xh1),numpy_fillna(predict_xp),-1*numpy_fillna(predict_xw)])

print("milestone",time.time()-startTime)
resultsByTree=np.split(np.squeeze(opponentBests), predict_treelens)

print("milestone",time.time()-startTime)
feedbackForNodes=[]
for i in range(len(opponentDecisions)):
    feedbackForNodes+=[dict(zip(opponentDecisions[i]['nodeIDs'], resultsByTree[i]))]
    
print("milestone",time.time()-startTime)
toPool=[]
for i in range(len(opponentDecisions)):
    toPool+=[{'node': myHands[i], 'update': feedbackForNodes[i]}]

print("milestone",time.time()-startTime)
test_x_dict=pool.map(getOwns, toPool)
print("milestone",time.time()-startTime)
test_y_dict=pool.map(getSmarts, toPool)
print("milestone",time.time()-startTime)
test_xh1=[]
test_xp=[]
test_xw=[]

for i in range(len(test_x_dict)):
    test_xh1+=test_x_dict[i]['hand1']
    test_xp+=test_x_dict[i]['plays']
    test_xw+=test_x_dict[i]['whos']

print("milestone",time.time()-startTime)
test_y=[]

for i in range(len(test_y_dict)):
    test_y+=test_y_dict[i]['smartScores']
print("milestone",time.time()-startTime)


# In[ ]:



hand_input = Input(shape=(4,), dtype='float32', name='hand_input')
plays_input = Input(shape=(10,), dtype='float32', name='plays_input')
whos_input = Input(shape=(10,), dtype='float32', name='whos_input')

allins = keras.layers.concatenate([hand_input, plays_input, whos_input])

# We stack a deep densely-connected network on top
x = Dense(240, activation='relu')(allins)
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

model = Model(inputs=[hand_input, plays_input, whos_input], outputs=[main_output])

model.compile(optimizer='adam', loss='mean_squared_error')
datadict= {"train_x": [np.array(train_xh1),numpy_fillna(train_xp),numpy_fillna(train_xw)], "train_y": [np.array(train_y)], "test_set": [[np.array(test_xh1),numpy_fillna(test_xp),numpy_fillna(test_xw)], [np.array(test_y)]] }




# In[ ]:


model.fit(datadict["train_x"], datadict["train_y"],validation_data=datadict["test_set"],epochs=64, batch_size=4096)


# In[66]:


myBests=model.predict(datadict['test_set'][0])


# In[ ]:



test_treelens=[len(test_x_dict[0]['whos'])]
for i in range(len(test_x_dict)):
    test_treelens+=[test_treelens[-1]+len(test_x_dict[0]['whos'])]

resultsByTree=np.split(np.squeeze(myBests), test_treelens)

feedbackForNodes=[]
for i in range(len(test_x_dict)):
    feedbackForNodes+=[dict(zip(test_x_dict[i]['nodeIDs'], resultsByTree[i]))]
    
toPool=[]
for i in range(len(test_x_dict)):
    toPool+=[{'node': myHands[i], 'update': feedbackForNodes[i]}]

playedGames=pool.map(updateOpponent, toPool)


# In[ ]:


for i in range(0,20):
    playedGames[i].expand()
    playedGames[i].updatePredScore(toPool[i]['update'])
    playedGames[i].calcSmartScore()
    print(playedGames[i].hand1)
    print("    ",playedGames[i].getGamePlays())
    print(playedGames[i].hand2)
    print()


# In[35]:


startTime=time.time()
pegScore(playedGames[3].getGamePlays(),[1,-1,1,-1,1,-1,1])
print("milestone",time.time()-startTime)


# In[36]:


np.stack((datadict['test_set'][1][0][0:100], myBests[0:100]), axis=-1).tolist()


# In[26]:


pool.restart()


# In[49]:


import pickle
len(pickle.dumps(listOfHands))


# In[29]:


listOfHands=[]
for i in range(100):
    listOfHands+=[pool.map(newGame, range(10))]


# In[30]:


startTime = time.time()
q=pool.map(getSmartList,listOfHands)
print("milestone",time.time()-startTime)


# In[20]:


def getSmartList(x):
    for i in range(len(x)):
        x[i].expand()
    return x[i]


# In[29]:


pool=Pool(processes=24)


# In[28]:


pool.terminate()


# In[73]:


playedGames[0].expand()

