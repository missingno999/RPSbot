

##!!!SECURITY NOTIFICATION!!!
##There is no sanitization performed on hard coded data passed to eval(). This means that if an attacker changes a variable's value at the register level, they could execute arbitrary code.
##Use at your own risk

print("importing")

import random
import torch
from torch import nn
from abc import ABC, abstractmethod, abstractproperty
import __main__
from os import mkdir, getcwd, listdir
from os.path import isfile, exists
import threading
from datetime import datetime
from collections import deque

print("import complete\n\n")

##The node of a linked list
class memNode():
    def __init__(self, value):
        self.Value=value
        self.nextNode=None
    def setNextNode(self, nextNod):
        self.nextNode=nextNod

#The linked list class representing bot move memory
#Global used: memoryLength
class memChain():
    def __init__(self, memoryLength):
        self.memLength=memoryLength
        self.head=memNode(torch.tensor([[0., 1., 0.]]))
        nodey=self.head
        for i in range(self.memLength-1):
            nexty=memNode(torch.tensor([[0., 1., 0.]]))
            nodey.setNextNode(nexty)
            nodey=nexty
        self.tail=nodey
    def update(self, newVal):
        self.tail.setNextNode(memNode(newVal))
        self.head=self.head.nextNode
        self.tail=self.tail.nextNode
    def getContents(self):
        tense=self.head.Value
        nodey=self.head
        for i in range(self.memLength-1):
            tense=torch.cat((tense,nodey.nextNode.Value),1)
            nodey=nodey.nextNode
        return tense

#This is an abstract class meant to be used as the basis for users adding their own trainer classes
#globals used:trainerInputSem,primaryBotThreadID,secondaryBotThreadID
class RPStrainer(ABC):
    ##This function computes what move to play this turn
    ##playerPrevMove=an int representing what the trainer played last turn
    ##BotPrevMove=an int representing what the AI played last turn
    ##outcome=an int representing whether last round ended in a tie, win for the trainer, or win for the AI
    ##Returns: an int representing what move the AI should play this round
    @abstractmethod
    def computeMove(self, playerPrevMove, BotPrevMove,outcome):
        pass
    #The name of the class. As a string, exactly
    @property
    @abstractmethod
    def ___Name___(self):
        pass
    #The very first move the bot will play. Returns an int representing said move.
    def computeFirstMove(self):
        return self.randThrow()
    #The function that the program actually calls to get a move from the trainer.
    #Meant as an interface to simplify things from the user's perspective.
    def getNextMove(self, playerPrevMove, BotPrevMove,outcome):
        hashh={'r':0,'p':1, 's':2}
        if(playerPrevMove!=None):
            goods=self.computeMove(hashh.get(playerPrevMove),hashh.get(BotPrevMove),outcome)
        else:
            goods=self.computeFirstMove()
        if goods==0:
            return 'r'
        elif goods==1:
            return 'p'
        elif goods==2:
            return 's'
        else:
            print("Error in trainer logic, aborting run")
            return "exit"
    #Utility function. The AI sometimes thinks in terms of "change", so this function gives user-made trainers easy access to said change
    def getChange(self,firstMove, secondMove):
        return (secondMove-firstMove+3)%3 #0=tie, 1=upgrade, 2=downgrade
    #Utility function that just returns a random throw.
    def randThrow(self):
        randy=random.random()
        if randy<=(1.0/3.0):
            return 0
        elif randy<=(2.0/3.0):
            return 2
        else:
            return 1
    ##A way for user to SAFELY provide user-input to their trainers.
    ##Due to the AI running in concurrent threads during botVbot mode, using this function prevents the bots from talking over each other. It also allows the user to see which bot is requesting input.
    ##inputPrompt=the string that will be used as the prompt when requesting input.
    ##Returns: a string equivalent to whatever the user input.
    def getUserInput(self, inputPrompt):
        global botVbotBoolean
        global trainerInputSem
        if(botVbotBoolean):
            trainerInputSem.acquire()
            if secondaryBotThreadID==threading.get_native_id():
                print("SECONDARY BOT INPUT REQUESTED")
            elif primaryBotThreadID==threading.get_native_id():
                print("PRIMARY BOT INPUT REQUESTED")
            daGoods=input(inputPrompt)
            trainerInputSem.release()
            return daGoods
            #SecondarybotthreadId HAS to be checked first. Initialization of these variables means that when secondaryBot checks it's ID, it is possible that primaryID == secondaryID
        else:
            return input(inputPrompt)
    #If the user has the trainer track some variables, this is the function called to reinitialize those variables.
    #Called at the start of every game.
    def preGameAssignment(self):
        return 0 

#Default trainer class.
class defaultTrainer(RPStrainer):
    def ___Name___(self):
        return "defaultTrainer"
    def computeMove(self, playerPrevMove, BotPrevMove,outcome):
        randy=random.random()
        if randy<=(1.0/3.0):
            return 0
        elif randy<=(2.0/3.0):
            return 2
        else:
            return 1

device="cpu"

##Loss function. Arguably a modified version of cross entropy loss.
class correctLinearLoss(nn.Module):
    def __init__(self):
         super(correctLinearLoss, self).__init__()
    def forward(self,input_,target):
        return ((input_[0][torch.argmax(target)]-0.6)**2)

##The net
class NeuralNetwork(nn.Module):
    def __init__(self, memoryLength, layer1hiddenStateSize, layer2hiddenStateSize,memType):
        super().__init__()
        self.hiddenState0=torch.rand(1,layer2hiddenStateSize)#torch.tensor([[0., 0., 0.]])
        self.hiddenState1=torch.rand(1,layer1hiddenStateSize)#torch.tensor([[1., 1., 1.]])

        self.recurrent1 = nn.RNN(memoryLength*3+3+(memType*3),layer1hiddenStateSize,nonlinearity='tanh')
        self.recurrent2 = nn.RNN(memoryLength*3+3+(memType*3),layer1hiddenStateSize,nonlinearity='tanh')
        self.recurrent3 = nn.RNN((memoryLength*3+3+(memType*3)+(layer1hiddenStateSize*2)),layer2hiddenStateSize,nonlinearity='tanh')
        self.soft=nn.Softmax(dim=1)
        self.lin=nn.Linear(layer2hiddenStateSize,layer1hiddenStateSize)
        self.lin2=nn.Linear(layer2hiddenStateSize,3)

    def forward(self, x):
        self.hiddenState0 = self.hiddenState0.detach()
        self.hiddenState1 = self.hiddenState1.detach()
        self.hiddenState1=self.lin(self.hiddenState0)
        output, i=self.recurrent1(x,self.hiddenState1)
        output1, i=self.recurrent2(x,self.hiddenState1)
        output=self.soft(output)
        output1=self.soft(output1)
        output3=torch.cat((output,output1,x),1)
        output3, self.hiddenState0=self.recurrent3(output3,self.hiddenState0)
        output3=self.lin2(output3)
        output3=self.soft(output3)
        #print(self.hiddenState0)
        #print(output3)
        return output3


#Dictionary maps sissors/paper/rock to tensor for loss calulation
#Of important note is that the trainer and AI map throws in reverse to each other. This shouldn't affect anything, but it is worth noting.
movedict={"s": torch.tensor([[1., 0., 0.]]),
          "p": torch.tensor([[0., 1., 0.]]),
          "r": torch.tensor([[0., 0., 1.]])}
#Dictionary maps bot output to string representing what the bot PREDICTS the player will throw
botDict={0: "s",
           1: "p",
           2: "r"}
reverseBotDict={"s":0,
           "p":1,
           "r":2}
#Dictionary maps bot's prediction to the correct choice to beat said prediction
upDict={"s": "r",
           "p": "s",
           "r": "p"}

#The thread IDs for the two bots
primaryBotThreadID=0
secondaryBotThreadID=0

botVbotAmbassador=0 #The two bots need to be able to trade throws with each other. This variable holds throws that are in transit.
primaryAmbassador=threading.Semaphore(0) #These two semaphores are used to synchronize when the bots make throws, accessthe botVbotAmbassador, or compute loss
secondaryAmbassador=threading.Semaphore(0)
initializerSem=threading.Semaphore(0) #Only one bot resets the ambassador semaphores. Errors occure if one bot accesses a semaphore before it gets reset. This semaphore is used to prevent that.
trainerInputSem=threading.Semaphore(1) #Semaphore for when a trainer calls for input.

#The actual AI
class daBot():
    trainerClass=defaultTrainer() #The trainer
    botRefreshCyclePeriod=-30 #Negative values means the feature is disabled
    automationTrainingDuration=-50
    dataCollection=False
    collectionDiscardInterval=0
    lossMetric=1
    memoryLength=4
    layer1hiddenStateSize=9
    layer2hiddenStateSize=5
    memoryType=0 #0 remembers moves, 1 remembers change
    delimiter=""

    isTrainer=False #False for primary bot, true for secondary
    botVbotDuration=-100

    def __init__(self,isT=False):
        self.isTrainer=isT
    
    def initialize(self): #Called at the start of every game
        global secondaryBotThreadID, primaryBotThreadID
        self.memory=memChain(self.memoryLength) #Player move history

        #All AI objects are reinitialized to clear out any gradients, hidden states, or memory in general. "Training" versions are for the model used in bot refresh
        self.model = NeuralNetwork(self.memoryLength, self.layer1hiddenStateSize, self.layer2hiddenStateSize, self.memoryType).to(device)
        self.model.train()
        self.loss_fn = correctLinearLoss()
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=0.07)
        
        self.modelInTraining = NeuralNetwork(self.memoryLength, self.layer1hiddenStateSize, self.layer2hiddenStateSize, self.memoryType).to(device)
        self.modelInTraining.train()
        self.loss_fnTraining=correctLinearLoss()
        self.optimizerInTrainig = torch.optim.Adagrad(self.modelInTraining.parameters(), lr=0.07)

        self.trainingCountDown=self.automationTrainingDuration-1#Countdown on the number of rounds to train for
        self.dualTrainingStart=False #Bot Refresh begins training the second bot at the start of the second period, and then refreshes the bot every following period. This bool tells the AI to refresh
        self.outcome=0 #0=tie, 1=player won, 2=bot won. Used specifically by the trainer. I don't think it's used anywhere else.
        self.userHistory=torch.tensor([[0., 0., 0.]]) #The user's last move. Sort of a legacy variable, but one that I refuse to change.
        self.outcomeHistory=torch.tensor([[0., 0., 0.]]) #Exactly like outcome, but as a vector. Used by AI as a feature vector
        self.changeHistory=torch.tensor([[0., 0., 0.]]) #Tracks the change for the previous round
        #Rounds and score keeping
        self.numRounds=0
        self.numBotWins=0
        self.numPlayerWins=0
        self.numTies=0

        #Score keeping, but for when the player sets a "checkpoint". Allows total score and a split to be recorded and displayed simultaneously
        self.scoreCheckPoint=False
        self.numRoundsC=0
        self.numBotWinsC=0
        self.numPlayerWinsC=0
        self.numTiesC=0
        
        self.botVbotCountDown=self.botVbotDuration
        #Threads stuff. Only used in botVbot games
        global botVbotAmbassador, primaryAmbassador, secondaryAmbassador
        if self.isTrainer:
            secondaryBotThreadID=threading.get_native_id()
            botVbotAmbassador=0
            primaryAmbassador=threading.Semaphore(0)
            secondaryAmbassador=threading.Semaphore(0)
            initializerSem.release()
        elif self.botVbotDuration>0:
            initializerSem.acquire()
            primaryBotThreadID=threading.get_native_id()
        self.trainerClass.preGameAssignment()
        self.matchUpDict={"(r,r)":[0],"(r,p)":[0],"(r,s)":[0], #Used for tracking the number of matchups for data collection mode
                          "(p,r)":[0],"(p,p)":[0],"(p,s)":[0],
                          "(s,r)":[0],"(s,p)":[0],"(s,s)":[0]}
        self.dataSplitsStorage=[]#Stores the current round at each split in data collection mode

    ###Updates the parameters of the input model. Returns the output prediction of the model
    ##modelX=the model being trained
    ##optimizerX=the optimizer to use
    ##loss_fnX=the loss function for the current model
    ##userMove=char representing the user's move for this round
    ##botPrevMove=char representing the bot's previous move
    ##Returns: a tensor representing the AI's output (interpretation of output differs based on loss metric)
    def modelUpdate(self,modelX,optimizerX,loss_fnX,userMove, botPrevMove):
        #This is just one tensor being passed to the model after a lot of concatenation
        output=modelX(torch.cat((self.memory.getContents(),(self.outcomeHistory if not self.memoryType else torch.cat((self.outcomeHistory,self.userHistory),1))),1))
        loss=0
        if self.lossMetric==1:#throwPrediction: output is interpreted as rock, paper, or scissors
            loss=loss_fnX(output,movedict.get(upDict.get(userMove[0])))
        elif self.lossMetric==2:#directionMirroring: output is interpreted as the change between the bot's previous move and current move. Get's converted into the Rock Paper Scissors move space back in the main loop
            targetMove = nn.functional.softmax(movedict.get(userMove)[-1], dim=0).data
            TM_2 = torch.max(targetMove, dim=0)[1].item()
            originMove = nn.functional.softmax(self.userHistory[-1], dim=0).data
            OM_2 = torch.max(originMove, dim=0)[1].item()
            loss=loss_fnX(output,movedict.get(botDict.get(((TM_2-OM_2)+3)%3)))
        elif self.lossMetric==3: #directionPrediction: output interpreted as change
            targetMove = nn.functional.softmax(movedict.get(upDict.get(userMove))[-1], dim=0).data #the correct move the bot should have made
            TM_2 = torch.max(targetMove, dim=0)[1].item()
            originMove = nn.functional.softmax(movedict.get(botPrevMove)[-1], dim=0).data #the previous move the bot did make
            OM_2 = torch.max(originMove, dim=0)[1].item()
            loss=loss_fnX(output,movedict.get(botDict.get(((TM_2-OM_2)+3)%3)))
        elif self.lossMetric==4 or self.lossMetric==5:#PM Outcome and BM Outcome. Output is... weird. Very abstract
            daValue=None
            if(self.lossMetric==4):#PM Outcome
                daValue=self.userHistory
            else:#BM Outcome
                daValue=movedict.get(botPrevMove)
            targetMove=movedict.get(upDict.get(userMove[0])) #Don't undo this. The correct bot output is the move that BEATS the player's move.
            originMove = nn.functional.softmax(daValue[-1], dim=0).data
            OM_2 = torch.max(originMove, dim=0)[1].item()
            if self.outcomeHistory[0][1]==1: #This is where changes for when the player won the previous round are made
                targetMove=torch.tensor([[targetMove[0][0].item(),targetMove[0][2].item(),targetMove[0][1].item()]]) #Flip the mappings for paper & rock
                if OM_2<2: #This is meant to change the shift direction. 
                    OM_2=not OM_2
            targetMove = nn.functional.softmax(targetMove[-1], dim=0).data
            TM_3 = torch.max(targetMove, dim=0)[1].item()
            loss=loss_fnX(output, movedict.get(botDict.get((TM_3+OM_2)%3)))
        else:#PM Outcome 2#
            targetMove=movedict.get(upDict.get(userMove[0]))
            outcomeTarget = nn.functional.softmax(self.outcomeHistory[-1], dim=0).data
            OT_2 = torch.max(A2, dim=0)[1].item()
            if self.userHistory[0][1]==1:
                targetMove=torch.tensor([[targetMove[0][0].item(),targetMove[0][2].item(),targetMove[0][1].item()]])
            targetMove = nn.functional.softmax(targetMove[-1], dim=0).data
            TM_2 = torch.max(targetMove, dim=0)[1].item()
            if self.userHistory[0][2]==1:
                TM_2+=1
            loss=loss_fnX(output, movedict.get(botDict.get((TM_2+OT_2)%3)))
            #print(output)
        loss.backward()
        optimizerX.step()
        optimizerX.zero_grad()
        #print(output)
        return output

    #This function is used in botVbot mode so that moves can be generated independently of the loss calculation.
    def secondOutput(self,modelX):
        output=modelX(torch.cat((self.memory.getContents(),(self.outcomeHistory if not self.memoryType else torch.cat((self.outcomeHistory,self.userHistory),1))),1))
        return output

    #Companion to the previous function
    def secondUpdate(self,output,optimizerX,loss_fnX,userMove, botPrevMove):
        loss=0
        if self.lossMetric==1:#throwPrediction: output is interpreted as rock, paper, or scissors
            loss=loss_fnX(output,movedict.get(upDict.get(userMove[0])))
        elif self.lossMetric==2:#directionMirroring: output is interpreted as the change between the bot's previous move and current move. Get's converted into the Rock Paper Scissors move space back in the main loop
            targetMove = nn.functional.softmax(movedict.get(userMove)[-1], dim=0).data
            TM_2 = torch.max(targetMove, dim=0)[1].item()
            originMove = nn.functional.softmax(self.userHistory[-1], dim=0).data
            OM_2 = torch.max(originMove, dim=0)[1].item()
            loss=loss_fnX(output,movedict.get(botDict.get(((TM_2-OM_2)+3)%3)))
        elif self.lossMetric==3: #directionPrediction: output interpreted as change
            targetMove = nn.functional.softmax(movedict.get(upDict.get(userMove))[-1], dim=0).data #the correct move the bot should have made
            TM_2 = torch.max(targetMove, dim=0)[1].item()
            originMove = nn.functional.softmax(movedict.get(botPrevMove)[-1], dim=0).data #the previous move the bot did make
            OM_2 = torch.max(originMove, dim=0)[1].item()
            loss=loss_fnX(output,movedict.get(botDict.get(((TM_2-OM_2)+3)%3)))
        elif self.lossMetric==4 or self.lossMetric==5:#PM Outcome and BM Outcome. Output is... weird. Very abstract
            daValue=None
            if(self.lossMetric==4):#PM Outcome
                daValue=self.userHistory
            else:#BM Outcome
                daValue=movedict.get(botPrevMove)
            targetMove=movedict.get(upDict.get(userMove[0])) #Don't undo this. The correct bot output is the move that BEATS the player's move.
            originMove = nn.functional.softmax(daValue[-1], dim=0).data
            OM_2 = torch.max(originMove, dim=0)[1].item()
            if self.outcomeHistory[0][1]==1: #This is where changes for when the player won the previous round are made
                targetMove=torch.tensor([[targetMove[0][0].item(),targetMove[0][2].item(),targetMove[0][1].item()]]) #Flip the mappings for paper & rock
                if OM_2<2: #This is meant to change the shift direction. 
                    OM_2=not OM_2
            targetMove = nn.functional.softmax(targetMove[-1], dim=0).data
            TM_3 = torch.max(targetMove, dim=0)[1].item()
            loss=loss_fnX(output, movedict.get(botDict.get((TM_3+OM_2)%3)))
        else:#PM Outcome 2#
            targetMove=movedict.get(upDict.get(userMove[0]))
            outcomeTarget = nn.functional.softmax(self.outcomeHistory[-1], dim=0).data
            OT_2 = torch.max(A2, dim=0)[1].item()
            if self.userHistory[0][1]==1:
                targetMove=torch.tensor([[targetMove[0][0].item(),targetMove[0][2].item(),targetMove[0][1].item()]])
            targetMove = nn.functional.softmax(targetMove[-1], dim=0).data
            TM_2 = torch.max(targetMove, dim=0)[1].item()
            if self.userHistory[0][2]==1:
                TM_2+=1
            loss=loss_fnX(output, movedict.get(botDict.get((TM_2+OT_2)%3)))
        loss.backward()
        optimizerX.step()
        optimizerX.zero_grad()

    ##The heart of the AI. This is the main loop that accepts user input and passes data between the net and the user, primary & secondary bots
    def mainLoop(self):
        global botVbotAmbassador, primaryAmbassador, secondaryAmbassador, dataCollectionMoveRecord,dataCollectionSplits
        #Initializeation
        score=0
        self.initialize()
        dataDiscardMoveRecorderThing=True #Var used for tracking data discard interval stuff specifically for move recording. If False, then it will assume data discard is over/does not apply
        if self.collectionDiscardInterval==0:
            dataDiscardMoveRecorderThing=False
        elif (self.botVbotCountDown<=0): #if data discard interval is too big, data isn't supposed to be lost. Since I can't reset what's already been printed/written, a special pre-emptive var is required for protecting move recording data
            if self.trainingCountDown<self.collectionDiscardInterval:
                dataDiscardMoveRecorderThing=False
        else:
            if self.botVbotCountDown<self.collectionDiscardInterval:
                dataDiscardMoveRecorderThing=False
        automatedTrainerActive=False #As long as the bot is being trained by the automated trainer, this variable is True
        if((not self.isTrainer) and self.dataCollection and fileName[0]!='['): #Opening save file for data recording
            outputFile=open(dataCollectionPath+fileName+".txt","a")
        if(self.trainingCountDown>0): #Get first move from automated trainer
            automatedTrainerActive=True
            userMove=self.trainerClass.getNextMove(None,None,None) 
        elif(self.botVbotCountDown>0):#User input isn't requrested in the first rounds of botVbot mode.
            userMove=" " 
        else: #Get the user's first move
            print("Type R for rock, P for paper, or S for scissors. Type C to set a checkpoint. Type EXIT to end.")
            userMove=input("Enter your move: ").lower()
            while userMove=="" or (userMove!="exit" and userMove[0]!='s' and userMove[0]!='r' and userMove[0]!='p'):
                    userMove=input("Invalid input, enter new value: ").lower()
        ###Main main loop
        while userMove!="exit": #Loop until user enters "exit" keyword
            userMove=userMove[0]
            
            if(self.botRefreshCyclePeriod>0):#If the current bot refresh period is ending, move the newly trained bot to the foreground, and prepare a new background bot. Otherwise, train the background bot.
                if(self.numRounds==self.botRefreshCyclePeriod):
                    self.dualTrainingStart=True
                if(self.numRounds>self.botRefreshCyclePeriod and self.numRounds%self.botRefreshCyclePeriod==0):
                    self.model=self.modelInTraining
                    self.optimizer=self.optimizerInTrainig
                    self.loss_fn=self.loss_fnTraining

                    self.modelInTraining=NeuralNetwork(self.memoryLength, self.layer1hiddenStateSize,self.layer2hiddenStateSize, self.memoryType).to(device)
                    self.modelInTraining.hiddenState0=self.model.hiddenState0
                    self.modelInTraining.hiddenState1=self.model.hiddenState1
                    self.modelInTraining.train()
                    self.loss_fnTraining=correctLinearLoss()
                    self.optimizerInTrainig = torch.optim.Adagrad(self.modelInTraining.parameters(), lr=0.07)     
                if(self.dualTrainingStart and not(self.numRounds<self.memoryLength)):
                    if self.botVbotCountDown>0 and (automatedTrainerActive==False or self.trainingCountDown<=0):
                        outputTraining=self.secondOutput(self.modelInTraining)
                    else:
                        self.modelUpdate(self.modelInTraining,self.optimizerInTrainig,self.loss_fnTraining,userMove,botMove)

            #Bot move generation and model update
            if self.numRounds<self.memoryLength:#While memory is being populated, the model isn't used. This way, the bot isn't given inaccurate features
                randy=random.random()
                if randy<=(1.0/3.0):
                    botMove='r'
                elif randy<=(2.0/3.0):
                    botMove='s'
                else:
                    botMove='p'
            else:
                if self.botVbotCountDown>0 and (automatedTrainerActive==False or self.trainingCountDown<=0):
                    output=self.secondOutput(self.model) #Generating bot move for botVbot mode
                else:
                    output=self.modelUpdate(self.model,self.optimizer,self.loss_fn,userMove,botMove) #generating bot move and calculating loss + model update for non-botVbot mode
                #Process output and convert it to "move space" (movedict mapping).
                #In other words, if the loss metric altered the userMove when calculating loss, the following code performs the reverse operations to get the AI's output into the same domain as the userMove
                prob = nn.functional.softmax(output[-1], dim=0).data
                if (self.lossMetric==2 or self.lossMetric==3): #Converting from change space to move space.
                    botMove = botDict.get((torch.max(prob, dim=0)[1].item()+reverseBotDict.get(botMove))%3)
                    if self.lossMetric==2:# Direction mirroring can be highly accurate... but if it makes a mistake, it tends not to correct itself. This is where the correction is made
                        if self.outcome==1:#if bot lost last round
                            botMove=upDict.get(upDict.get(botMove))
                        elif self.outcome==0:#if bot tied last round
                            botMove=upDict.get(botMove)
                elif(self.lossMetric==4 or self.lossMetric==5): #Good luck figuring out this mapping. The high level idea is that player losing and tieing have the same mapping, winning swaps the mapping for paper and rock,
                    daValue=None                                #Upgrading a move shifts the mapping one to the right for tie/player lose, and one to the left for player win.
                    if(self.lossMetric==4):
                        daValue=self.userHistory
                    else:
                        daValue=movedict.get(botMove)
                    targetMoveKey = nn.functional.softmax(daValue[-1], dim=0).data #It's sort of like a hash or encryption. The correct mapping requires using the history and outcomes as keys
                    TMK_2 = torch.max(targetMoveKey, dim=0)[1].item()
                    if self.outcomeHistory[0][1]==1 and TMK_2<2:
                        TMK_2=not TMK_2#Swap paper and rock
                    if TMK_2>0:
                        TMK_2-=1 #Reverse mod direction. 2 normally downgrades, but if you make it equal 1 it now upgrades. That's the idea here.
                        TMK_2=not TMK_2
                        TMK_2+=1
                    prob_2 = torch.max(prob, dim=0)[1].item()
                    prob=movedict.get(botDict.get((prob_2+TMK_2)%3)) #Officially undoes the shift
                    if self.outcomeHistory[0][1]==1:
                        prob=torch.tensor([[prob[0][0].item(),prob[0][2].item(),prob[0][1].item()]]) #Undoes the swap
                    botMove = botDict.get((torch.max(prob[0], dim=0)[1].item()))
                elif(self.lossMetric==6): #I don't know what to say about this mapping.
                    prob_2=torch.max(prob, dim=0)[1].item()
                    if self.userHistory[0][2]==1:
                        prob_2+=2
                    outcomeTargetKey = nn.functional.softmax(self.outcomeHistory[-1], dim=0).data#A2
                    OTK_2 = torch.max(outcomeTargetKey, dim=0)[1].item()
                    if A2_2>0:
                        OTK_2-=1
                        OTK_2=not OTK_2
                        OTK_2+=1
                    prob=movedict.get(botDict.get((prob_2+OTK_2)%3))
                    if self.userHistory[0][1]==1:
                        prob=torch.tensor([[prob[0][0].item(),prob[0][2].item(),prob[0][1].item()]])
                    botMove = botDict.get((torch.max(prob[0], dim=0)[1].item()))
                else:
                    botMove = botDict.get((torch.max(prob, dim=0)[1].item()))
            
            if (automatedTrainerActive==False or self.trainingCountDown<=0):
                #The botVbot threads swap moves, and then calculate loss and update parameters
                if (self.botVbotCountDown>0):
                    if self.isTrainer==True:
                        botVbotAmbassador=botMove
                        secondaryAmbassador.release()
                        primaryAmbassador.acquire()
                        userMove=botVbotAmbassador
                    else:
                        secondaryAmbassador.acquire()
                        userMove=botVbotAmbassador
                        botVbotAmbassador=botMove
                        primaryAmbassador.release()
                    if not(self.numRounds<self.memoryLength):
                        if(self.dualTrainingStart):
                            self.secondUpdate(outputTraining,self.optimizerInTrainig,self.loss_fnTraining,userMove,botMove)
                        self.secondUpdate(output,self.optimizer,self.loss_fn,userMove,botMove)
                else:
                    if (not self.isTrainer) and not self.dataCollection: #Prints bot's move. When using a trainer to prepare a bot to play against a human, you'll want to print the bot's last move. Any other time a trainer is used, never print bot move.
                        print("Bot move: "+botMove)
            #Score keeping and outcome update
            if userMove==botMove:
                self.numTies+=1
                self.numTiesC+=1
                score-=1
                self.outcome=0
                self.outcomeHistory=torch.tensor([[1., 0., 0.]])
            elif upDict.get(userMove)==botMove:
                self.numBotWins+=1
                self.numBotWinsC+=1
                score-=1
                self.outcome=2
                self.outcomeHistory=torch.tensor([[0., 0., 1.]])
            else:
                self.numPlayerWins+=1
                self.numPlayerWinsC+=1
                score+=2
                self.outcome=1
                self.outcomeHistory=torch.tensor([[0., 1., 0.]])
            self.numRounds+=1
            self.numRoundsC+=1
            if(self.dataCollection and dataCollectionMoveRecord>0 and not dataDiscardMoveRecorderThing): ##Recording this round's moves to the output file
                if((self.botVbotDuration<0) or (not self.isTrainer and (automatedTrainerActive==False or self.trainingCountDown<=0))):
                    if(dataCollectionMoveRecord==3): #No return values here
                        self.recordMoves("(%s,%s)"%(botMove,userMove))
                    elif fileName[0]!='[':
                        outputFile.write(self.recordMoves("(%s,%s)"%(botMove,userMove)))
                    else:
                        holdEm=self.recordMoves("(%s,%s)"%(botMove,userMove))
                        if dataCollectionMoveRecord==2:
                            holdEm=holdEm[:-1]#Remove newline char, as it just messes with the print
                        print(holdEm)
            
            if(not self.memoryType):#Calculate change
                A1 = nn.functional.softmax(movedict.get(userMove)[-1], dim=0).data
                A1_2 = torch.max(A1, dim=0)[1].item()
                A2 = nn.functional.softmax(self.userHistory[-1], dim=0).data
                A2_2 = torch.max(A2, dim=0)[1].item()
                self.changeHistory=movedict.get(botDict.get(((A1_2-A2_2)+3)%3))
                #[1,0,0]=tie, [0,1,0]=downgraded, [0,0,1]=upgraded
            #Update histories
            self.userHistory=movedict.get(userMove)
            self.memory.update((self.changeHistory if self.memoryType else self.userHistory))

            #Wrap up the round and get the next user move. Among other things.
            if automatedTrainerActive==False or self.trainingCountDown<=0:
                if(self.botVbotCountDown>=1):
                    self.botVbotCountDown-=1
                    if(not self.isTrainer and dataCollectionSplits>0 and self.dataCollection and self.numRounds%dataCollectionSplits==0):
                        self.dataSplitsStorage.append([self.numRounds,self.numPlayerWins,self.numBotWins,self.numTies])
                    if self.dataCollection==True and (self.botVbotCountDown>0 and self.botVbotCountDown-(self.botVbotDuration-self.collectionDiscardInterval)==0):
                       #Data discard interval score reset for botVbot mode
                        self.numBotWins=0
                        self.numPlayerWins=0
                        self.numTies=0
                        self.numRounds=0
                        self.dataSplitsStorage=[] ##Highly inefficit way to keep splits from tracking discarded data for botVbotMode
                        dataDiscardMoveRecorderThing=False
                if(self.botVbotCountDown<=0):##Training AND botVbot mode have ended.
                    if(self.isTrainer):
                        userMove="exit" #Completely turn off the secondary bot's thread at the end of botVbot mode.
                    else: #Primary bot prints out/saves it's scoring data.
                        if(self.dataCollection and fileName[0]!='['): #Save data to output file
                            if(dataCollectionMoveRecord>0):
                                if(dataCollectionMoveRecord==1 and not(dataCollectionSplits>0 and self.numRounds%dataCollectionSplits==0)):
                                    outputFile.write('\n')
                                #This is where the table of move matchups is written to the output file
                                matchupBracket=lambda insertable: ["(%s,r)"%(insertable),"(%s,p)"%(insertable),"(%s,s)"%(insertable)]
                                listOinsertables=['r','p','s']
                                for inserta in listOinsertables:
                                    for m in matchupBracket(inserta):
                                        outputFile.write("|%s: %5.d(%3.5f) "%(m,self.matchUpDict.get(m)[0],float(self.matchUpDict.get(m)[0])/float(self.numRounds)))
                                    outputFile.write('\n')##Add new line at the end of each row of the table
                            if(len(self.delimiter)==0):
                                if self.botVbotDuration>0 and dataCollectionSplits>0 and self.numRounds%dataCollectionSplits==0: #The point of this is to keep the bot from printing to final round twice
                                    self.dataSplitsStorage.pop()
                                for datum in self.dataSplitsStorage:
                                    outputFile.write("Round:  "+self.scorePrinter(datum[0],datum[1],datum[2],datum[3],0)+"\n\n")#Extra space for parsing
                                outputFile.write("Rounds: "+self.scorePrinter(self.numRounds,self.numPlayerWins,self.numBotWins,self.numTies,0)+"\n")
                                if(dataCollectionSplits>0 or (dataCollectionMoveRecord>0 and dataCollectionMoveRecord<3)):
                                    outputFile.write("-----------------------------")
                                outputFile.write("\n")
                            else:
                                if self.botVbotDuration>0 and dataCollectionSplits>0 and self.numRounds%dataCollectionSplits==0: #The point of this is to keep the bot from printing to final round twice
                                    self.dataSplitsStorage.pop()
                                for datum in self.dataSplitsStorage:
                                    outputFile.write(self.scorePrinter(datum[0],datum[1],datum[2],datum[3],1)+"\n\n")
                                outputFile.write(self.scorePrinter(self.numRounds,self.numPlayerWins,self.numBotWins,self.numTies,1)+"\n")
                                if(dataCollectionSplits>0 or (dataCollectionMoveRecord>0 and dataCollectionMoveRecord<3)):
                                    outputFile.write("-----------------------------")
                                outputFile.write("\n")
                            outputFile.close()
                        else: #Print scoring data to console
                            if(self.dataCollection):
                                if(dataCollectionMoveRecord>0):
                                    if(dataCollectionMoveRecord==1 and not(dataCollectionSplits>0 and self.numRounds%dataCollectionSplits==0)):
                                        print("")#print a new line
                                    #This is where the table of move matchups is printed to console
                                    matchupBracket=lambda insertable: ["(%s,r)"%(insertable),"(%s,p)"%(insertable),"(%s,s)"%(insertable)]
                                    listOinsertables=['r','p','s']
                                    for inserta in listOinsertables:
                                        printable=""#Since print automatically adds a new line char, a different approach is needed
                                        for m in matchupBracket(inserta):
                                            printable+="|%s: %5.d(%3.5f) "%(m,self.matchUpDict.get(m)[0],float(self.matchUpDict.get(m)[0])/float(self.numRounds))
                                        print(printable)
                                if self.botVbotDuration>0 and dataCollectionSplits>0 and self.numRounds%dataCollectionSplits==0: #The point of this is to keep the bot from printing to final round twice
                                    self.dataSplitsStorage.pop()
                                for datum in self.dataSplitsStorage:
                                    print("Round:  "+self.scorePrinter(datum[0],datum[1],datum[2],datum[3],0)+"\n")#Extra space for parsing
                            print("Rounds: "+self.scorePrinter(self.numRounds,self.numPlayerWins,self.numBotWins,self.numTies,0))
                            if(self.scoreCheckPoint==True):
                                print("Checkpoint: "+self.scorePrinter(self.numRoundsC,self.numPlayerWinsC,self.numBotWinsC,self.numTiesC,0))
                            print("Score: %d\n" % (score))
                        if(self.dataCollection==True): #End the game if in data collection mode
                            userMove="exit"
                        else: #Else, wait for user input
                            userMove=input("Enter your move: ").lower()
                            while userMove=="" or (userMove!="exit" and userMove[0]!='s' and userMove[0]!='r' and userMove[0]!='p'):
                                if(userMove[0]=='c'):
                                    self.scoreCheckPoint=True
                                    self.numRoundsC=0
                                    self.numBotWinsC=0
                                    self.numPlayerWinsC=0
                                    self.numTiesC=0
                                    print("Checkpoint set")
                                    userMove=input("Enter your move: ").lower()
                                else:
                                    userMove=input("Invalid input, enter new value: ").lower()       
            else:#At the end of training for botVbot mode, or at the end of the data discard interval, reset the score keeping variales (excluding the literal score)
                userMove=self.trainerClass.getNextMove(userMove, botMove, self.outcome)
                self.trainingCountDown-=1
                if(dataCollectionSplits>0 and self.dataCollection and self.botVbotDuration<=0 and self.numRounds%dataCollectionSplits==0):
                    self.dataSplitsStorage.append([self.numRounds,self.numPlayerWins,self.numBotWins,self.numTies])
                if ((self.dataCollection==True and (self.botVbotCountDown<=0 and self.trainingCountDown-(self.automationTrainingDuration-self.collectionDiscardInterval)+1==0))
                    or (self.botVbotDuration>0 and self.trainingCountDown==0 and self.botVbotCountDown==self.botVbotDuration)):
                    self.numBotWins=0
                    self.numPlayerWins=0
                    self.numTies=0
                    self.numRounds=0
                    self.dataSplitsStorage=[] #Highly inefficit way to keep splits from tracking discarded data
                    if(not self.botVbotDuration>0): #If botVbot mode, then the data discard interval hasn't happened yet
                        dataDiscardMoveRecorderThing=False
                    else:
                        dataDiscardMoveRecorderThing=True
                    if self.botVbotCountDown==self.botVbotDuration:
                        score=0

    ##Function for printing out the outcome totals and percentages
    ##r=self.numRounds, pw=self.numPlayerWins, bw=self.numBotWins, t=self.numTies (or their checkpoint equivlents)
    ##deliminat=boolean determining which format method to return.
    ##Returns: A string to be appended onto wither "Rounds: " or "Checkpoint: "
    def scorePrinter(self,r,pw,bw,t,deliminat):
        if not deliminat:
            return("%d  Player: %d (%3.5f)  Bot: %d (%3.5f)  Tie: %d (%3.5f)" % (r,pw,float(pw)/r*100,
                                                                                bw,float(bw)/r*100,t,float(t)/r*100))
        else: #Funky way of handling deliminator-seperated output.
            daGoods=str(r)
            stats=((pw,float(pw)/r*100),(bw,float(bw)/r*100),(t,float(t)/r*100))
            for pair in stats:
                daGoods+="%s%d%s%3.5f"%(self.delimiter,pair[0],self.delimiter,pair[1])
            return daGoods

    ##Function for handling recording each rounds moves and formatting the output
    ##matchUp=a string representing each palyer's moves that round. Both moves are a single lowercase letter, with the AI's move first, surruonded by parentheses and seperated by a comma
    ##Returns: A string formatted according to the global dataCollectionMoveRecord and dataCollectionSplits
    def recordMoves(self,matchUp):
        global dataCollectionMoveRecord #0 means this feature is off.
        self.matchUpDict.get(matchUp)[0]+=1
        if dataCollectionMoveRecord==1: #Just a single line of move match ups
            if(fileName[0]=='['):
                if dataCollectionSplits>0 and self.numRounds%dataCollectionSplits==0:
                    return f"{matchUp} Round {self.numRounds}"
                else:
                    return matchUp
            if(self.numRounds==1):
                return matchUp+(f" Round {self.numRounds}\n" if dataCollectionSplits==1 else "")
            else:
                if dataCollectionSplits>0:
                    if dataCollectionSplits==1:
                        return f"{matchUp} Round {self.numRounds}\n"
                    if self.numRounds%dataCollectionSplits==0:
                        return f",{matchUp} Round {self.numRounds}\n"
                    elif (self.numRounds-1)%dataCollectionSplits==0:
                        return matchUp
                return f",{matchUp}"
        if dataCollectionMoveRecord==2: #Collumns
            if(self.numRounds==1):
                return("Bot\tTrainer\n%s\t%s\n"%(matchUp[1],matchUp[3]))
            else:
                return("%s\t%s"%(matchUp[1],matchUp[3])+(" Round "+str(self.numRounds)+"\n" if dataCollectionSplits>0 and self.numRounds%dataCollectionSplits==0 else "\n"))

#I'm not 100% sure why I have the global booleans. They all relate to if the features of the AI are on or off, which is already encoded in the sign of the
#Corrosponsing AI variable. But eh. If a variable is followed by 'Sec', that means it's the 'Secondary' AIs variable.
botRefreshCycleBoolean=False
automationBoolean=False
botRefreshCycleBooleanSec=False
automationBooleanSec=False
dataCollection=False
numberOfGames=30 #Number of games played during data collection
botVbotBoolean=False
targetBot=None #Pointer used for deciding which AI to set the variables for
dataCollectionPath=getcwd()+"\\RPSbotWD\\RPSbotDataCollection\\" #Absolute Save Directory
fileName="output" #File save location
Deliminator="''"
dataCollectionSplits=0
dataCollectionMoveRecord=0

outputQue=deque(maxlen=5)#Que used for tracking most recent data output locations

###This function is used to parse the menu inputs.
###commandString=a string that the user entered from any interface other than the actual game UI
###Returns: an array of strings, each of which represents one command to execute
def settingsParseBatch(commandString):
    commandString=commandString+"," #Extra comma added for... I dunno. Catching corner cases I guess.
    quotemark=""
    coms=[] #Commands. Parsed commands are stored here
    partialComs=[""] #Partial commands. Stores commands as they are parsed. If a closing quotation is forgotten, each element in this list is copied to coms as seperate elements. Otherwise, they're all combined into one command
    for i in commandString:
        partialComs[len(partialComs)-1]+=i
        if quotemark=="":
            if i==",":
                skeleton=""
                for ii in partialComs:
                    skeleton+=ii[:-1]
                coms.append(skeleton)
                partialComs=[""]
            if (i=="\"" or i=="\'"):
                quotemark=i
        else:
            if i==",":
                partialComs.append("")
            if i==quotemark:
                skeleton=""
                for ii in partialComs:
                    skeleton+=ii[:]
                partialComs=[skeleton]
                quotemark=""
    if(len(partialComs)>1): #Clean up remaining commands, I don't know why I needed this.
        for ii in partialComs:
            if len(ii)>0:
                coms.append(ii[:-1])
    for i in range(len(coms)): #Cleans up all the parsed commands, and adds some corner case catching whitespace (important for if the user entered a blank command)
        coms[i]=coms[i].strip()+" "
    return coms

##Checks in the input contains a number.
##argString=the argument (NO COMMAND LETTER) that needs to be parsed.
##Returns: the string representation of the first continuous integer found in the input, or "" if no ints are found.
def settingsParseInt(argString):
    daGoods=""
    end=False
    for i in argString:
        if end==True:
            if not i.isdigit():
                exit
            else:
                daGoods=daGoods+i
        if i.isdigit() and end==False:
            daGoods=daGoods+i
            end=True
    return daGoods

##Handles the formatting for printing most of the AI's parameters. It also handles printing the Secondary bot's parameters in an organized way.
##parameter=the uncompiled variable that the desired value is associated with.
##spacingInt=an string representing an int, used for formatting the parameters (how many characters long should the parameter be printed as)
##Returns: The formatted string to be printed out.
def smartFormatter(parameter, spacingInt='0'):
    if parameter=="botRefreshCycleBoolean" or parameter=="automationBoolean":
        return eval("'%-"+spacingInt+"."+spacingInt+"s%s'%(str("+parameter+"),('' if not botVbotBoolean else f'  Secondary: {str("+parameter+"Sec)}'))")
    #EVAL NOTE: spacingInt comes from a hard coded value, so it shouldn't be much of a risk. parameter is usually variable names related to the bot, EXCEPT for when it's
    #the trainerClass name. This value comes from the user written script for running the AI, and thus is semi-arbitrary
    elif parameter=="trainerClass.___Name___()":
        return eval("'%-"+spacingInt+"."+spacingInt+"s%s'%(primaryBot."+parameter+",('' if not botVbotBoolean else f'  Secondary: {secondaryBot."+parameter+"}'))")
    else:
        return eval("'%-"+spacingInt+".d%s'%(abs(primaryBot."+parameter+"),('' if not botVbotBoolean else f'  Secondary: {abs(secondaryBot."+parameter+")}'))")


dataCollectionSplits=0
dataCollectionMoveRecord=0

##The data collection sub menu
##val=the string of arguments passed from the top menu
##Returns: Nothing at all. Return statements are just used to end the function
##Globals used: target/primary/secondaryBot, dataCollection, numberOfGames, dataCollectionPath, fileName, Deliminator,automationBoolean,botVbotBoolean,outputQue,
##dataCollectionSplits,dataCollectionMoveRecord
def setDataCollects(val):
    global dataCollection, numberOfGames
    global dataCollectionPath, fileName, Deliminator
    global dataCollectionSplits, dataCollectionMoveRecord
    global automationBoolean
    MoveRecString={0:"OFF",1:"Single line",2:"Columns",3:"Running Total"}
    #Some checks for if top menu has passed some arguments or not. If yes, prepares them to be ran throuh the command parser
    if(len(val)>1 and val[0]==val[-1] and (val[0]=="\"" or val[0]=="\'")):
        command=val[1:-1]+',q'
    else:
        command=None
    if(command==None):
        print("\n==============================\nDATA COLLECTION SETTINGS\nSet the program to run the Automatic Trainer/Bot Versus Bot mode for a set number of games (rather than rounds), while saving the results of each game to a file.\n"+
              "To enable or disable this feature, enter E. To set the number of games to run for, enter G. To set the number of rounds to ignore the outcomes of (the Data Discard Interval), enter I. "+
              "To set the default directory files will be saved to (the Absolute Save Directory), enter A (entering A with no arguments prints the current Absolute Save Directory). "+
              "To set the name of the output file, enter F. Supplying no arguments will send output to the console, where it will NOT be saved. Passing \\ as the argument will print the last five save locations. "+
              "To set the Delimiter, enter D, and surround your argument in quotation marks. Having no delimiter will cause ouput to be formatted as usual. Having a delimiter will cause output to be formatted "+
              "as raw numbers seperated by your delimiter. To set the bot to record data in splits, enter S. Set this parameter to zero to turn it off. "+
              f"To turn on and set the formmating for move recording, enter M. Once done, enter Q to return to the top menu.\n"+
              f"\nData Collection (E)nabled: {str(dataCollection)}\nNumber of (G)ames: {numberOfGames}\nDiscard (I)nterval: {'OFF' if not primaryBot.collectionDiscardInterval else primaryBot.collectionDiscardInterval}\n"+
              f"Output (F)ile Destination: {fileName}\n(D)elimiter: {Deliminator}\n(S)plits: {(dataCollectionSplits if dataCollectionSplits>0 else 'OFF')}\n(M)ove Record: {MoveRecString.get(dataCollectionMoveRecord)}"+("\nSwap Target (B)ot" if botVbotBoolean else ""))
        command=input("\nEnter your command: ")
    parsedCommands=settingsParseBatch(command)
    while True:
        for i in parsedCommands:
            if i[0].lower()=='e': #Enable data collection
                dataCollection=not dataCollection
                primaryBot.dataCollection=dataCollection
                secondaryBot.dataCollection=dataCollection
                print("Data Collection is now "+ ("ON" if dataCollection else "OFF"))
                if ((not automationBoolean and not botVbotBoolean) and (dataCollection)):
                    automationBoolean=True
                    primaryBot.automationTrainingDuration=primaryBot.automationTrainingDuration*-1
                    print("Automatic Training has been turned ON. Data Collection requires either an Automated Trainer or Bot Versus Bot mode to run.")
            elif i[0].lower()=='i': #Set data discard interval
                ii=settingsParseInt(i)
                if ii.isdigit():
                    if int(ii)>=0:
                        primaryBot.collectionDiscardInterval=int(ii)
                        secondaryBot.collectionDiscardInterval=int(ii)
                        print(f"Data Discard Interval = {'OFF' if ii=='0' else ii}")
                        if (int(ii)>=abs(primaryBot.automationTrainingDuration) if not botVbotBoolean else int(ii)>=primaryBot.botVbotDuration):
                            print("NOTE: If Data Discard Interval is equal to or greater than the number of rounds played in a game, then it WON'T AFFECT THE DATA")
                            
                    else:
                        print("Error: The Data Discard Interval can not be a negative number")
                else:
                    print("Error: input is not in integer format")
            elif i[0].lower()=='g': #Set the number of games
                ii=settingsParseInt(i)
                if ii.isdigit():
                    if int(ii)>0:
                        numberOfGames=int(ii)
                        print(f"Number of Games = {ii}")
                    else:
                        print("Error: The number of games can not be a non-positive number")
                else:
                    print("Error: input is not in integer format")
            elif i[0].lower()=='a': #Set the Absolute Save Directory
                ii=i[1:].strip()
                if(len(ii)==0):
                    print(f"Current Absolute Save Directory: {dataCollectionPath}")
                elif (exists(ii) and not isfile(ii)):
                    dataCollectionPath=ii+"\\"
                    print(f"Absolute Save Directory set to {ii}")
                    ff=open(getcwd()+"\\RPSbotWD\\RPSbotMeta\\saveDir.txt",'w')
                    ff.write(dataCollectionPath)
                    ff.close()
                else:
                    print(f"Error: {ii} is not a valid folder. Please create this folder manually, or use a different directory. Note that this input should NOT be a text file.")
            elif i[0].lower()=='f': #Set output file name
                ii=i[1:].strip()
                validity=0
                if(len(ii)>0):
                    if(ii[0]=='\\'):
                        print("Five most recent save locations, old to new:\n")
                        for iii in outputQue:
                            print(iii)
                        validity=2
                    else:
                        for iii in ii:#Check file name for illegal characters. Files can be saved in locations relative to the absolute save directory
                            test=(iii.isalpha() or iii.isdigit() or iii=='-' or iii=='.' or iii=='_' or iii=='\\')
                            if not test:
                                validity=1
                                exit
                else:
                    ii="[output to console]"
                if validity==0:
                    fileName=ii
                    print(f"File Name set to {ii}")
                elif validity==1:
                    print("Error: the file name you provided is invalid. Note that only alphanumeric characters, along with '-' '.' and '_' are allowed.")
            elif i[0].lower()=='d': #Set the delimiter
                ii=i[1:].strip()
                startt=ii.find('\"')
                end=ii[startt+1:].find('\"')#NOTE: end is found using a substring of ii, but will be employed for parsing the whole string. Hence the up coming offset.
                if startt!=-1 and end!=-1:
                    Deliminator=ii[startt:end+startt+2]#Set the global delimiter as a substring of the command. Start from startt, go to end.
                    primaryBot.delimiter=Deliminator[1:-1]#While the delimiter shown in the settings keeps it's quotation marks, the one saved to the bots omits the quotation marks for easier parsing
                    secondaryBot.delimiter=primaryBot.delimiter
                    print(f"Delimiter set to: {Deliminator}")
                else: #Same code, different type of quotation marks
                    startt=ii.find('\'')
                    end=ii[startt+1:].find('\'')
                    if startt!=-1 and end!=-1:
                        Deliminator=ii[startt:end+startt+2]
                        primaryBot.delimiter=Deliminator[1:-1]
                        secondaryBot.delimiter=primaryBot.delimiter
                        print(f"Delimiter set to: {Deliminator}")
                    else:
                        print("Error: Invalid input. Make sure to surround the delimiter with \" \".")
            elif i[0].lower()=='s':#Enable and set splits
                ii=settingsParseInt(i)
                if ii.isdigit():
                    if int(ii)>=0:
                        dataCollectionSplits=int(ii)
                        print(f"Splits = {(dataCollectionSplits if dataCollectionSplits>0 else 'OFF')}")
                    else:
                        print("Error: You can not take splits at intervals less than 0")
                else:
                    print("Error: input is not in integer format")
            elif i[0].lower()=='m':#Enable and set move recording
                ii=settingsParseInt(i)
                if ii.isdigit():
                    if int(ii)>=0 and int(ii)<=3:
                        dataCollectionMoveRecord=int(ii)
                        print(f"Move Record Formatting = {MoveRecString.get(dataCollectionMoveRecord)}")
                    else:
                        print("Error: There are only 4 options for this variable. These options are labeled 1,2,3 or to turn move recording off, 0.")
                else:
                    print("Error: input is not in integer format. Move recording has only 4 options. These options are labeled 1,2,3 or to turn move recording off, 0.")
            elif i[0].lower()=='b' and botVbotBoolean: #Obligatory bot swap
                swapTargetBot()
            elif i[0].lower()=='q': #End function and discard remaining commands for this submenu
                return 0 #end function
            else: #Invalid command
                print("Invalid command")
        print("\n------------------------------")
        if botVbotBoolean:
            print(f"Current Target bot: {'Primary bot' if targetBot==primaryBot else 'Secondary bot'}")
        print(f"\nData Collection (E)nabled: {str(dataCollection)}\nNumber of (G)ames: {numberOfGames}\nDiscard (I)nterval: {'OFF' if not primaryBot.collectionDiscardInterval else primaryBot.collectionDiscardInterval}\n"+
              f"Output (F)ile Destination: {fileName}\n(D)elimiter: {Deliminator}\n(S)plits: {(dataCollectionSplits if dataCollectionSplits>0 else 'OFF')}"+
              f"\n(M)ove Record: {MoveRecString.get(dataCollectionMoveRecord)}"+("\nSwap Target (B)ot" if botVbotBoolean else "")+"\n(Q)uit")
        command=input("\nEnter your command: ")
        parsedCommands=settingsParseBatch(command)

##The automated trainer and botVbot sub menu
##val=the string of arguments passed from the top menu
##Returns: Nothing at all. Return statements are just used to end the function
##Globals used: target/primary/secondaryBot, automationBoolean, botVbotBoolean, dataCollection,trainerDict
def setAutomation(val):
    global targetBot
    global automationBoolean, automationBooleanSec, botVbotBoolean
    global dataCollection
    #Some checks for if top menu has passed some arguments or not. If yes, prepares them to be ran throuh the command parser
    if(len(val)>1 and val[0]==val[-1] and (val[0]=="\"" or val[0]=="\'")):
        command=val[1:-1]+',q'
    else:
        command=None
    if(command==None):
        print("\n==============================\nAUTOMATIC TRAINER SETTINGS\nEnable the AI to play against either a preprogramed algorithm or another AI, either for data collection or to train it to play against a human. "+
              "Note that the Bot Versus Bot mode can be used for training the AI for a game against a human, but the second AI is NOT an Automated Trainer. The Secondary bot has almost all the same configurable "+
              "parameters as the Primary bot, including an independent Automated Trainer.\n"+
              "To enable or disable algorithmic training (Automated Training), enter E. To set the number of rounds to train for (the Training Duration), enter D. To set the algorithm used by the Automated Trainer, enter A. "+
              "To get a list of Automated Trainer algorithms, enter A with no aregumenrs.\n"+
              "To enable or disable Bot Versus Bot mode, enter V. To set the number of rounds the two bots will play against each other (excluding each bot\'s Training Duration), enter R. Once done, enter Q to return to the Top menu.\n"+
              f"\nAuto Training (E)nabled: {smartFormatter('automationBoolean','17')}\nTraining (D)uration: {smartFormatter('automationTrainingDuration','21')}\nAuto Trainer (A)lgorithm: {smartFormatter('trainerClass.___Name___()','16')}\n\n"+
              f"Bot (V)ersus Bot mode active: {str(botVbotBoolean)}\nNumber of (R)ounds: {abs(primaryBot.botVbotDuration)}"+("\nSwap Target (B)ot" if botVbotBoolean else ""))
        command=input("\nEnter your command: ")
    parsedCommands=settingsParseBatch(command)
    while True:
        for i in parsedCommands:
            if i[0].lower()=='e': #Enable automated trainer
                differ=""
                if targetBot!=primaryBot:
                    differ="Sec"
                    automationBooleanSec=not automationBooleanSec
                else:
                    automationBoolean=not automationBoolean
                targetBot.automationTrainingDuration=abs(targetBot.automationTrainingDuration) if eval("automationBoolean"+differ) else -1*abs(targetBot.automationTrainingDuration)
                print("Auto Training is now "+ ("ON" if eval("automationBoolean"+differ) else "OFF"))
                if ((not eval("automationBoolean"+differ) and not botVbotBoolean) and (dataCollection)):
                    dataCollection=False
                    primaryBot.dataCollection=False
                    secondaryBot.dataCollection=False
                    print("Data Collection has been turned OFF. Data Collection requires either an Automated Trainer or Bot Versus Bot mode to run.")
            elif i[0].lower()=='d': #Set training duraton
                ii=settingsParseInt(i)
                if ii.isdigit():
                    if int(ii)>0:
                        targetBot.automationTrainingDuration=int(ii) if automationBoolean else -1*int(ii)
                        print(f"Training Period = {ii}")
                        if ((not botVbotBoolean) and primaryBot.collectionDiscardInterval>=int(ii)):
                            print("NOTE: If Data Discard Interval is equal to or greater than the number of rounds played in a game, then it WON'T AFFECT THE DATA")
                    else:
                        print()
                else:
                    print("Error: input is not in integer format")
            elif i[0].lower()=='a': #Sets the automated trainer algorithm NOTE: This and loading are the only times that the trainer's __init__() is called. Otherwise, the called function is the user created initializer
                ii=i[1:].strip()
                if(len(ii)==0):
                    print("List of Trainer Algorithms:\n")
                    algoy=0 #Each algorithm is associated with a number. This is just an incrementer for them
                    out=trainerDict.get(algoy)
                    while out!=None:
                        print(f"{algoy}: {out[(0 if not algoy else 9):]}")#Algorithms are stored with __main__ appended on, except for algorithm 0. As such, remove these characters for every algo except the first one
                        algoy+=1
                        out=trainerDict.get(algoy)
                elif getTrainer(ii)==1: #Prints the following string on a success, and lets the error be printed by getTrainer()
                    print(f"Automated Trainer Algorithm = {targetBot.trainerClass.___Name___()}")
            elif i[0].lower()=='v': #enables bot V bot mode
                botVbotBoolean=not botVbotBoolean
                primaryBot.botVbotDuration=abs(primaryBot.botVbotDuration) if botVbotBoolean else -1*abs(primaryBot.botVbotDuration)
                secondaryBot.botVbotDuration=primaryBot.botVbotDuration
                if not botVbotBoolean:
                    targetBot=primaryBot
                print("Bot Versus Bot mode is now "+("OFF" if not botVbotBoolean else ("ON\nNOTE: The secondary \"trainer\" AI\'s parameters will be listed alongside the primary bot's parameters "+
                                                     "as Secondary. To set the Secpndary AI\'s parameters, enter B, and proceed as normal. You can swap which bot\'s parameters you are setting by entering B.\n")))
                if ((not automationBoolean and not botVbotBoolean) and (dataCollection)):
                    dataCollection=False
                    primaryBot.dataCollection=False
                    secondaryBot.dataCollection=False
                    print("Data Collection has been turned OFF. Data Collection requires either an Automated Trainer or Bot Versus Bot mode to run.")
            elif i[0].lower()=='r': #Sets the number of rounds to run the bot versus bot trainer for
                ii=settingsParseInt(i)
                if ii.isdigit():
                    if int(ii)>0:
                        primaryBot.botVbotDuration=int(ii) if botVbotBoolean else -1*int(ii)
                        secondaryBot.botVbotDuration=primaryBot.botVbotDuration
                        print(f"Bot Versus Bot Rounds = {ii}")
                        if(botVbotBoolean and primaryBot.collectionDiscardInterval>=int(ii)):
                            print("NOTE: If Data Discard Interval is equal to or greater than the number of rounds played in a game, then it WON'T AFFECT THE DATA")
                    else:
                        print("Error: The number of rounds can not be a non-positive number")
                else:
                    print("Error: input is not in integer format")
            elif i[0].lower()=='b' and botVbotBoolean:#obligatory bot swap
                swapTargetBot()
            elif i[0].lower()=='q':#End function and discard remaining commands for this submenu
                return 0 #end function
            else:#Invalid command
                print("Invalid command")
        print("\n------------------------------")
        if botVbotBoolean:
            print(f"Current Target bot: {'Primary bot' if targetBot==primaryBot else 'Secondary bot'}")
        print(f"\nAuto Training (E)nabled: {smartFormatter('automationBoolean','17')}\nTraining (D)uration: {smartFormatter('automationTrainingDuration','21')}"+
              f"\nAuto Trainer (A)lgorithm: {smartFormatter('trainerClass.___Name___()','16')}\n\n"+
              f"Bot (V)ersus Bot mode active: {str(botVbotBoolean)}\nNumber of (R)ounds: {abs(primaryBot.botVbotDuration)}"+
              ("\nSwap Target (B)ot" if botVbotBoolean else "")+"\n(Q)uit")
        command=input("\nEnter your command: ")
        parsedCommands=settingsParseBatch(command)

##A helper function for parsing trainer names and setting the trainers.
##i=a string either representing an int associated with a trainer algorithm, or the exact name of a trainer algorithm
##Returns: 1 on success, otherwise 0
##Globals used: targetBot,trainerDict,trainerDictR

###!!!!IMPORTANT VULNERABILITY NOTE!!!
###Part of the security of using eval on trainer class names is that python syntax prevents class names from containing malicious code.
###However, after the user written code is compiled the python file can be altered to bypass this security measure, allowing malicious code to be added to the file and executed
###by an eval statement. The window for this attack is small due to data only being loaded as text once. This attack also requires access to the user written script, at which point it
###becomes a user executing malicious code on their own device, and no one can stop that. Risk low, but possible.
def getTrainer(i):
    global targetBot
    i=i.strip()
    ii=i
    if ii.isdigit():
        trainName=trainerDict.get(int(ii))
        if (trainName!=None):
            targetBot.trainerClass=eval(trainName+"()",{"__main__":__main__,"defaultTrainer":defaultTrainer},{})
            return(1)
        else:
            print("Error: The number is outside the valid range. For the default algorithm, enter 0. Every other algorithm starts at 1 and increases according to apperance in "
                  +"source file")
            return(0)
    else:
        ii=trainerDictR.get(i)
        if(ii!=None):
            trainName=trainerDict.get(ii)
            targetBot.trainerClass=eval(trainName+"()",{"__main__":__main__,"defaultTrainer":defaultTrainer},{})
            return 1
        else:
            print(f"Error: Their is no Trainer Algorithm with the name {i}")
            return 0

##The bot refresh sub menu
##val=the string of arguments passed from the top menu
##Returns: Nothing at all. Return statements are just used to end the function
##Globals used: target/primary/secondaryBot, botRefreshCycleBoolean, botVbotBoolean
def setBotRefresh(val):
    global targetBot
    global botRefreshCycleBoolean, botRefreshCycleBooleanSec
    #Some checks for if top menu has passed some arguments or not. If yes, prepares them to be ran throuh the command parser
    if(len(val)>1 and val[0]==val[-1] and (val[0]=="\"" or val[0]=="\'")):
        command=val[1:-1]+',q'
    else:
        command=None
    if(command==None):    
        print(f"\n==============================\nBOT REFRESH SETTINGS\nEnable\\disable Bot Refreshing, where the trained model is periodically rebooted mid-game.\n"+
                      f"To enable or disable this feature, enter E. To set the number of rounds played before each refresh (the Refresh Period), enter P. Once done, enter Q to return to the Top menu.\n"+
                      f"\nRefreshing (E)nabled: {smartFormatter('botRefreshCycleBoolean','10')}\nRefresh (P)eriod: {smartFormatter('botRefreshCyclePeriod','14')}"+("\nSwap Target (B)ot" if botVbotBoolean else ""))
        command=input("\nEnter your command: ")
    parsedCommands=settingsParseBatch(command)
    while True:
        for i in parsedCommands:
            if i[0].lower()=='e': #Enable feature
                differ=""
                if targetBot!=primaryBot:
                    differ="Sec"
                    botRefreshCycleBooleanSec=not botRefreshCycleBooleanSec
                else:
                    botRefreshCycleBoolean=not botRefreshCycleBoolean
                targetBot.botRefreshCyclePeriod=abs(targetBot.botRefreshCyclePeriod) if eval("botRefreshCycleBoolean"+differ) else -1*abs(targetBot.botRefreshCyclePeriod)
                print("Bot Refreshing is now "+ ("ON" if eval("botRefreshCycleBoolean"+differ) else "OFF"))
            elif i[0].lower()=='p': #Set refresh period
                ii=settingsParseInt(i)
                if ii.isdigit():
                    if int(ii)>0:
                        targetBot.botRefreshCyclePeriod=int(ii) if botRefreshCycleBoolean else -1*int(ii)
                        print(f"Refresh Period = {ii}")
                    else:
                        print("Error: The Refresh Period can not be a non-positive number")
                else:
                    print("Error: input is not in integer format")
            elif i[0].lower()=='b' and botVbotBoolean:#Obligatory bot swap
                swapTargetBot()
            elif i[0].lower()=='q':#End function and discard remaining commands for this submenu
                return 0 #end function
            else:#invalid command
                print("Invalid command")
        print("\n------------------------------")
        if botVbotBoolean:
            print(f"Current Target bot: {'Primary bot' if targetBot==primaryBot else 'Secondary bot'}")
        print(f"\nRefreshing (E)nabled: {smartFormatter('botRefreshCycleBoolean','10')}\nRefresh (P)eriod: {smartFormatter('botRefreshCyclePeriod','14')}"+("\nSwap Target (B)ot" if botVbotBoolean else "")+"\n(Q)uit")
        command=input("\nEnter your command: ")
        parsedCommands=settingsParseBatch(command)

##The sub menu for setting misc parameters
##val=the string of arguments passed from the top menu
##Returns: Nothing at all. Return statements are just used to end the function
##Globals used: target/primary/secondaryBot, metrics, botVbotBoolean
def setNetStructure(val):
    global targetBot
    global metrics
    #Some checks for if top menu has passed some arguments or not. If yes, prepares them to be ran throuh the command parser
    if(len(val)>1 and val[0]==val[-1] and (val[0]=="\"" or val[0]=="\'")):
        command=val[1:-1]+',q'
    else:
        command=None    
    LM=metrics.get(primaryBot.lossMetric)#Converts number to printable loss metric name
    LM2=metrics.get(secondaryBot.lossMetric)
    condensedMemTypePrint=lambda tarBot: "%-27.27s"%(('Change' if tarBot.memoryType else 'Move'))
    if(command==None):
        print(f"\n==============================\nNUERAl NET STRUCTURE\nYou can set various elements of the nueral network's structure from here, such as how many values each layer outputs for it's hidden state.\n"+
              "To set the number of values in the first layer's Hidden State, enter F. The set the number of values in the second layer's, enter S. To set the number of previous moves saved to "+
              "the bot's memory, enter M. To toggle Memory Type between classic 'Move' and the experimental 'Change' types, enter T. To pick which values the bot uses to calculate loss (the Loss Metric), enter L. Once done, enter Q to return to the Top menu.\n"+
              f"\n(F)irst Hidden State Size: {smartFormatter('layer1hiddenStateSize','15')}\n(S)econd Hidden State Size: {smartFormatter('layer2hiddenStateSize','14')}\n(M)emory Length: {smartFormatter('memoryLength','25')}\n"+
              f"Memory (T)ype: {condensedMemTypePrint(primaryBot)+('  Secondary: '+condensedMemTypePrint(secondaryBot) if botVbotBoolean else '')}\nCurrent (L)oss Metric: {'%-19.19s'%(LM)}"+("" if not botVbotBoolean else f"  Secondary: {LM2}")+("\nSwap Target (B)ot" if botVbotBoolean else ""))
        command=input("\nEnter your command: ")
    parsedCommands=settingsParseBatch(command)
    while True:
        for i in parsedCommands:
            if i[0].lower()=='m': #Set memory length
                ii=settingsParseInt(i)
                if ii.isdigit():
                    if int(ii)>0:
                        targetBot.memoryLength=int(ii)
                        print("Memory Length = "+ii)
                    else:
                        print("Error: Memory Length can not be a non-positive number")
                else:
                    print("Error: input is not in integer format")
            elif i[0].lower()=='t':
                targetBot.memoryType=not targetBot.memoryType
                print(f"Memory Type = {('CHANGE' if targetBot.memoryType else 'MOVE')}")
            elif i[0].lower()=='l': #Set loss metric. Only accepts ints
                ii=settingsParseInt(i)
                if ii.isdigit():
                    if int(ii)>0 and int(ii)<7: #Increase this range if any new loss metrics are added
                        targetBot.lossMetric=int(ii)
                        if targetBot==primaryBot:
                            LM=metrics.get(targetBot.lossMetric)
                        else:
                            LM2=metrics.get(targetBot.lossMetric)
                        print("Metric = "+(LM if targetBot==primaryBot else LM2))
                    else:
                        print("Error: The Loss Metrics are labeled 1 through 6, and all other values are invalid")
                else:
                    print("Error: input is not in integer format")
            elif i[0].lower()=='f' or i[0].lower()=='s': #Set the size of either hidden state
                ii=settingsParseInt(i)
                if ii.isdigit():
                    if int(ii)>0:
                        if i[0].lower()=='f':
                            targetBot.layer1hiddenStateSize=int(ii)
                            print("First Hidden State Size = "+ii)
                        else:
                            targetBot.layer2hiddenStateSize=int(ii)
                            print("Second Hidden State Size = "+ii)
                    else:
                        print("Error: Hidden State Sizes cannot be a non-positive number")
                else:
                    print("Error: input is not in integer format")
            elif i[0].lower()=='b' and botVbotBoolean:#obligatory bot swap
                swapTargetBot()
            elif i[0].lower()=='q':#End function and discard remaining commands for this submenu
                return 0 #end function
            else:#invalid command
                print("Invalid command")
        print("\n------------------------------")
        if botVbotBoolean:
            print(f"Current Target bot: {'Primary bot' if targetBot==primaryBot else 'Secondary bot'}")
        print(f"\n(F)irst Hidden State Size: {smartFormatter('layer1hiddenStateSize','15')}\n(S)econd Hidden State Size: {smartFormatter('layer2hiddenStateSize','14')}\n"+
              f"(M)emory Length: {smartFormatter('memoryLength','25')}\nMemory (T)ype: {condensedMemTypePrint(primaryBot)+('  Secondary: '+condensedMemTypePrint(secondaryBot) if botVbotBoolean else '')}\n"+
              f"Current (L)oss Metric: {'%-19.19s'%(LM)}"+("" if not botVbotBoolean else f"  Trainer: {LM2}")+("\nSwap Target (B)ot" if botVbotBoolean else "")+"\n(Q)uit")
        command=input("\nEnter your command: ")
        parsedCommands=settingsParseBatch(command)

##Swaps which bot the target bot points to.
##local=a string. this is a command, and all commands are passed with a list of sub arguments, but this function doesn't use them.
##Globals are self evident
def swapTargetBot(local=""):
    global primaryBot, secondaryBot, targetBot
    if targetBot==primaryBot:
        targetBot=secondaryBot
        print("Setting parameters for SECONDARY BOT\n")
    else:
        targetBot=primaryBot
        print("Setting parameters for PRIMARY BOT\n")

##Special sub menu for viewing parameter save files. This function is called by saveParams() and loadParams(), NOT by the top menu
##readWriteMode: a string that dictates whether the user can save data,or write data. Baiscally a boolean with the possible values of "SAVE" or "LOAD"
##Returns: either a file name to save/load to, or None in the event of aborting the load/save
##Globals used: metrics
def fileExplorer(readWriteMode):
    #parameters can't be passed to this function
    print("==============================\nSAVE/LOAD MENU\nThis menu allows you to see all your save parameter save files, and to view their contents.\n"+
          "To view the contents of a file, enter V followed by the file name. To choose a file to save\load, enter C followed by the file name. "+
          ("You can also save to a new file by entering C followed by the new file name. " if readWriteMode=="SAVE" else "")+
          "To return to the top menu without saving/loading, enter Q. To reprint the list of files, enter R.\n"+
          f"NOTE: you can only save while in SAVE mode, and load while in LOAD mode. You are currently in {readWriteMode} mode. To switch, return to the top menu and enter the opposite operation.\n")
    print("loading files...")
    daList=listdir(getcwd()+"\\RPSbotWD\\RPSbotSaves\\")
    print("loading done\n")#loading saves
    for ff in range(len(daList)):
        if(daList[ff].endswith(".txt")):
            daList[ff]=daList[ff][:-4]#Removing .txt from end
            print(daList[ff])
        else:
            daList[ff]=None #if a file is not .txt, remove it. Probably will cause errors, but only if the user is storing files where they don't belong.
    print("\n(V)iew, (C)hoose, (R)eprint, (Q)uit")
    command=input("\nEnter your command: ")
    parsedCommands=settingsParseBatch(command)
    while True:
        for i in parsedCommands:
            if i[0].lower()=='v': #Prints the contents of a file
                ii=i[1:].strip()
                try:
                    MoveRecString={0:"OFF",1:"Single line",2:"Columns",3:"Running Total"}
                    index=daList.index(ii)
                    daData=open(getcwd()+"\\RPSbotWD\\RPSbotSaves\\"+daList[index]+".txt","r")
                    print(f"Automated Trainer Algorithm: {daData.readline()[:-1]}")
                    k=int(daData.readline()[:-1])
                    print(f"Auto Trainer Enabled: {('False' if k<=0 else 'True')}")
                    print(f"Automation Duration: {abs(k)}")
                    k=int(daData.readline()[:-1])
                    print(f"Bot Versus Bot Mode Enabled: {('False' if k<=0 else 'True')}")
                    print(f"Number of Rounds: {abs(k)}")
                    print(f"Data Collection enabled: {daData.readline()[:-1]}")
                    print(f"Discard Interval: {daData.readline()[:-1]}\nNumber of Games: {daData.readline()[:-1]}\nDelimiter: {daData.readline()[:-1]}")
                    print(f"Splits: {daData.readline()[:-1]}")
                    print(f"Move Recorder: {MoveRecString.get(int(daData.readline()[:-1]))}")
                    k=int(daData.readline()[:-1])
                    print(f"Bot Refresh Enabled: {('False' if k<=0 else 'True')}")
                    print(f"Refresh Period: {abs(k)}")
                    print(f"Layer 1 Hidden State Size: {daData.readline()[:-1]}\nLayer 2 Hidden State Size: {daData.readline()[:-1]}\n"+
                          f"Memory Length: {daData.readline()[:-1]}")
                    k=(daData.readline()[:-1])
                    print(f"Memory Type: {('Change' if k[0]=='1' else 'Move')}")
                    print(f"Loss Metric: {metrics.get(int(daData.readline()))}\n")
                except ValueError as error:
                    print(f"Error: {ii} does not exist")
            elif i[0].lower()=='c': #Chooses a file to save/load. Note that some functionality from the save/load functions has been copied here. This is a useability feature. Aborting saves/loads should not exit this menu
                ii=i[1:].strip()
                if len(ii)>0:
                    found=0
                    for item in daList:
                        if item==ii:
                            found=1
                            exit
                    if found==1:
                        if(readWriteMode=="SAVE" and ii=="default"):
                            warningResponse=(input("NOTE: You are attempting to overwrite the default setting. If you do, then the settings you save here will be automatically loaded every time the program is started. "+
                                        "Are you sure you want to overwrite it? Enter Y for yes, N for no.\nEnter choice: ")+" ").lower()
                        else:   
                            warningResponse=input((f"{('Overwrite ' if readWriteMode=='SAVE' else 'Load ')}{ii}? Enter Y for yes, N for no.\nEnter choice: ")+" ").lower()
                        while not(warningResponse[0]=='y' or warningResponse[0]=='n'):
                            warningResponse=(input("Sorry, your choice is unclear. Enter Y for yes, N for no.")+" ").lower()
                        if warningResponse[0]=='n':
                            print(f"{('Save' if readWriteMode=='SAVE' else 'Load')} cancelled\n")
                        else:
                            return ii
                    else:
                        if readWriteMode=="SAVE":
                            validity=0
                            for iii in ii:
                                test=(iii.isalpha() or iii.isdigit() or iii=='-' or iii=='.' or iii=='_')
                                if not test:
                                    validity=1
                                    exit
                            if(validity==1):
                                print("Error: the file name you provided is invalid. Note that only alphanumeric characters, along with '-' '.' and '_' are allowed.")
                            else:
                                warningResponse=(input(f"{ii} does not exist. Create a new file and save to it? Enter y for yes, n for no.\nEnter choice: ")+" ").lower()
                                while not(warningResponse[0]=='y' or warningResponse[0]=='n'):
                                    warningResponse=(input("Sorry, your choice is unclear. Enter Y for yes, N for no.")+" ").lower()
                                if warningResponse[0]=='n':
                                    print("Save cancelled\n")
                                else:
                                    return ii
                        else:
                            print(f"Error: {ii} does not exist, so it can't be loaded")
                else:
                    print(f"Error: invalid file name")
            elif i[0].lower()=='r': #Reprint the list of files
                for ff in daList:
                    print(ff)
            elif i[0].lower()=='q': #Return to top menu
                return None
            else:#Invalid command
                print("Invalid command")
        print("\n------------------------------")#No bot switching included for some reason, and I refuse to change that.
        if botVbotBoolean:
            print(f"Current Target bot: {'Primary bot' if targetBot==primaryBot else 'Secondary bot'}")
        print("\n(V)iew, (C)hoose, (R)eprint, (Q)uit")
        command=input("\nEnter your command: ")
        parsedCommands=settingsParseBatch(command)
                    
##Saves the parameters of the target bot to a file.
##local=file name to save to. No file extension, and NO RELATIVE PATHES
##Returns: Nothing. Returns used to escape.
##Globals used: initer. (a boolean for skipping prints on initial data load/save). All parameter ones. This is a save function, so it reads a lot of data.
def saveParams(local):
    global targetBot
    global numberOfGames
    global initer
    local=local.strip()
    validity=0
    for iii in local:#Tests for illegal characters in file name. Note that relative pathes are not allowed, because why bother? Relative pathes would also break fileExplorer()
        test=(iii.isalpha() or iii.isdigit() or iii=='-' or iii=='.' or iii=='_')
        if not test:
            validity=1
            exit
    if(validity==1):
        print("Error: the file name you provided is invalid. Note that only alphanumeric characters, along with '-' '.' and '_' are allowed. No relative paths are allowed.\nAborting save...")
        return None
    goMenue=None
    if(len(local)==0): #If no file name was given, run fileExplorer() in SAVE mode.
        goMenue=fileExplorer("SAVE")
        if goMenue==None: #Gracefully exit fileExplorer, hiding the fact that saveParams was ever executed
            return 0
        else: #Else, take the selected file and perform a save on it.
            local=goMenue
    if(not initer and goMenue==None): #fileExplorer() unused, do standard check ins with user
        if isfile(getcwd()+"\\RPSbotWD\\RPSbotSaves\\"+local+".txt"):
            if(local=="default"):
               i=(input("NOTE: You are attempting to overwrite the default setting. If you do, then the settings you save here will be automatically loaded every time the program is started. "+
                        "Are you sure you want to overwrite it? Enter Y for yes, N for no.\nEnter choice: ")+" ").lower()
            else:
                i=(input(f"NOTE: {local}.txt already exists. Are you sure you want to overwrite it? Enter Y for yes, N for no.\nEnter choice: ")+" ").lower()
            while not(i[0]=='y' or i[0]=='n'):
                i=(input("Sorry, your choice is unclear. Enter Y for yes, N for no.")+" ").lower()
            if i[0]=='n':
                print("Save cancelled")
                return 0
    f=open(getcwd()+"\\RPSbotWD\\RPSbotSaves\\"+local.rstrip()+".txt","w")
    f.write(targetBot.trainerClass.___Name___()+'\n')
    f.write(str(targetBot.automationTrainingDuration)+'\n')
    f.write(str(targetBot.botVbotDuration)+'\n')
    f.write(str(targetBot.dataCollection)+'\n')
    f.write(str(targetBot.collectionDiscardInterval)+'\n')
    f.write(str(numberOfGames)+'\n')
    f.write(Deliminator+'\n')
    f.write(str(dataCollectionSplits)+'\n')
    f.write(str(dataCollectionMoveRecord)+'\n')
    f.write(str(targetBot.botRefreshCyclePeriod)+'\n')
    f.write(str(targetBot.layer1hiddenStateSize)+'\n')
    f.write(str(targetBot.layer2hiddenStateSize)+'\n')
    f.write(str(targetBot.memoryLength)+'\n')
    f.write(str(int(targetBot.memoryType))+'\n')
    f.write(str(targetBot.lossMetric))
    f.close()
    if(not initer):
        print(f"Successfully saved {('primary bot' if targetBot==primaryBot else 'secodary bot')}\'s parameters to {local}.txt")

def loadVarification(loaddedValue, booleanEqu, errorMessage):
    if booleanEqu(loaddedValue):
        return loaddedValue
    else:
        print(errorMessage)
        raise ValueError()

##Loads the parameters from the save file to the target bot.
##local=file name to load from. No file extension, and NO RELATIVE PATHES
##Returns: Nothing. Returns used to escape.
##Globals used: initer. (a boolean for skipping prints on initial data load/save). All parameter ones. This is a load function, so it reads a lot of data.
##NOTE: This function does NOT validate parameters beyond basic type checking. This is an error that should definitly be fixed as some point...
def loadParams(local):
    global targetBot
    global botRefreshCycleBoolean, automationBoolean
    global botRefreshCycleBooleanSec, automationBooleanSec
    global dataCollection, numberOfGames, botVbotBoolean
    global Deliminator, dataCollectionSplits, dataCollectionMoveRecord
    local=local.strip()
    goMenu=None
    if(len(local)==0): #If no file name was given, run fileExplorer() in LOAD mode.
        goMenue=fileExplorer("LOAD")
        if goMenue==None: #Gracefully exit fileExplorer, hiding the fact that loadParams was ever executed
            return 0
        else: #Else, take the selected file and perform a load on it.
            local=goMenue
    setType=0 #Variables related to botVbot mode or data collection are shared between bots, and considered "globals". 
    setGlobals=False
    if(isfile(getcwd()+"\\RPSbotWD\\RPSbotSaves\\"+local+".txt")):
        f=open(getcwd()+"\\RPSbotWD\\RPSbotSaves\\"+local+".txt",'r+')
        if botVbotBoolean:
            i=(input("Load full setting? (y/n)\nIf you enter Y, then all settings, including ones related to Bot Versus Bot mode, will be loaded and set for the primary bot. If you "+
                     "enter N, then only settings unique to each bot will be loaded, and they'll be set for the current Target bot.\n"+
                     "Enter choice: ")+" ").lower()
            while not(i[0]=='y' or i[0]=='n'):
                i=(input("Sorry, your choice is unclear. Enter Y for yes, N for no.")+" ").lower()
            if i[0]=='y':
                if(targetBot==secondaryBot):
                    swapTargetBot()
                setType=1
        if((not botVbotBoolean) or setType==1):
            setGlobals=True
        try:    
            if getTrainer((f.readline()[:-1]))==0:#loads in the trainer
                print("Error: invalid Trainer Algorithm")
                raise ValueError()
            targetBot.automationTrainingDuration=loadVarification(int(f.readline()[:-1]), lambda x: x!=0,
                                                                  "Error: Invalid Training Duration. The Training Duration can not be zero")
            if(targetBot.automationTrainingDuration<=0):
                if(targetBot==primaryBot):
                    automationBoolean=False
                else:
                    automationBooleanSec=False
            else:
                if(targetBot==primaryBot):
                    automationBoolean=True
                else:
                    automationBooleanSec=True
            if(setGlobals): #The loads for globals
                primaryBot.botVbotDuration=loadVarification(int(f.readline()[:-1]), lambda x: x!=0,
                                                            "Error: Invalid Bot Versus Bot mode number of rounds. It can not be zero")
                secondaryBot.botVbotDuration=primaryBot.botVbotDuration
                if(primaryBot.botVbotDuration>0):
                    botVbotBoolean=True
                else:
                    botVbotBoolean=False
                dataCollection=f.readline()[0]=="T"
                if(not automationBoolean and not botVbotBoolean and dataCollection):
                    dataCollection=False
                    print("WARNING: Data Collection was saved as \"True\" in an impossible state. Setting Data Collection to \"False\"")
                primaryBot.dataCollection=dataCollection
                secondaryBot.dataCollection=dataCollection
                primaryBot.collectionDiscardInterval=loadVarification(int(f.readline()[:-1]), lambda x: x>=0,
                                                                      "Error: Invalid Data Discard Interval. Discard Interval must be greater than or equal to 0")
                numberOfGames=loadVarification(int(f.readline()[:-1]), lambda x: x>0,
                                               "Error: Invalid number of games. This value must be greater than 0")
                Deliminator=loadVarification(f.readline()[:-1],lambda x: len(x)>1 and (x[0]=='\'' and x[1:].find('\'')==len(x)-2 ) or (x[0]=="\"" and x[1:].find("\"")==len(x)-2),
                                             "Error: Delimiter was not saved with the correct formatting")
                primaryBot.delimiter=Deliminator[1:-1]
                secondaryBot.delimiter=primaryBot.delimiter
                dataCollectionSplits=loadVarification(int(f.readline()[:-1]), lambda x: x>=0,
                                                                      "Error: Invalid Splits length. This value must be greater than or equal to 0")
                dataCollectionMoveRecord=loadVarification(int(f.readline()[:-1]), lambda x: x>=0 and x<4,
                                                                      "Error: Invalid Move Records mode. The valid values are 0, 1, 2, or 3.")
            else: #If not loading globals, consume the lines containing their data from the stream
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()
            targetBot.botRefreshCyclePeriod=loadVarification(int(f.readline()[:-1]), lambda x: x!=0,
                                                            "Error: Invalid Bot Refresh period. The period cannot be zero")
            if(targetBot.botRefreshCyclePeriod<=0):
                if(targetBot==primaryBot):
                    botRefreshCycleBoolean=False
                else:
                    botRefreshCycleBooleanSec=False
            else:
                if(targetBot==primaryBot):
                    botRefreshCycleBoolean=True
                else:
                    botRefreshCycleBooleanSec=True
            targetBot.layer1hiddenStateSize=loadVarification(int(f.readline()[:-1]), lambda x: x>0,
                                               "Error: Invalid Layer size. This value must be greater than 0")
            targetBot.layer2hiddenStateSize=loadVarification(int(f.readline()[:-1]), lambda x: x>0,
                                               "Error: Invalid Layer size. This value must be greater than 0")
            targetBot.memoryLength=loadVarification(int(f.readline()[:-1]), lambda x: x>0,
                                               "Error: Invalid Memory Length. This value must be greater than 0")
            targetBot.memoryType=loadVarification(int(f.readline()[:-1]), lambda x: x==0 or x==1,
                                               "Error: Memory Type has an invalid value. It should be either 0 or 1")
            targetBot.lossMetric=loadVarification(int(f.readline()), lambda x: x>0 and x<7,
                                               "Error: Invalid Loss Metric value. This value must be equal to or between 0 and 6")
            
            f.close()
            if(not initer):
                if(setGlobals):
                    print(f"Successfully loaded full settings from {local}.txt")
                else:
                    print(f"Successfully loaded parameters from {local}.txt and applied them to the {('primary bot' if targetBot==primaryBot else 'secodary bot')}")
        except ValueError:
            if not initer:
                print(f"Error: {local} has invalid parameter values or is corrupted. This may be because the save is from an earlier, incompatable version of the program. Aborting load...")
            else:
                raise Exception()
    else:
        if(not initer):
            print(f"Error: {local}.txt does not exist. Aborting load...")

##The game handler.
##val=required parameter input. Doesn't do anything
##Returns: Nothing. return statement is just an escape.
##Globals used: primaryBot, secondaryBot, dataCollection, numberOfGames,botVbotBoolean, metrics, outputQue
def startGame(val):
    global metrics
    intervalWarn=False #Check for if there is a potental problem with data discard interval
    splitsAndMovePrints=False #Check if data collection is printing moves and splits to console 
    #If data discard interval is greater than the number of rounds set to be played, let user know and give them the chance to change it.
    if (dataCollection and (primaryBot.collectionDiscardInterval>=primaryBot.botVbotDuration if botVbotBoolean else primaryBot.collectionDiscardInterval>=abs(primaryBot.automationTrainingDuration))):
        intervalWarn=True
    if(dataCollection and fileName[0]=='[' and (dataCollectionSplits>0 or dataCollectionMoveRecord>0)):
        splitsAndMovePrints=True


    if(intervalWarn or splitsAndMovePrints):
        intervalWarnText=(("\nWARNING: Your Data Discard Interval is greater than the number of rounds data will be collected for. "+
                 "If you run the bot, data WILL BE RECORDED.") if intervalWarn else "")
        splitMovesText=(("\nWARNING: You are outputting data collection to console while having splits or move recording on, which will print out a lot more info. This could slow down the "
                         +"program significantly, and might result in data being lost in history.") if splitsAndMovePrints else "")
        warning=input(("------------------------"+intervalWarnText+splitMovesText+"\nRun anyway? Enter Y for yes, N for no.\nEnter choice: ")+" ").lower()
        while not(warning[0]=='y' or warning[0]=='n'):
            warning=(input("Sorry, your choice is unclear. Enter Y for yes, N for no.")+" ").lower()
        if warning[0]=='n':
            print("Aborting...")
            return None
    print("\n++++++++++++++++++++++++\nSTARTING AI...\n")
    if dataCollection and fileName[0]!='[': #Preping the data collection output file and updating the list of recent data outputs
        f=open(dataCollectionPath+fileName+".txt","a")
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S")+"\n")
        #lambda is only used in this if block, and only accepts (primaryBot,'P') and (secondaryBot,'S')
        #botPointer is the bot to pull parameters from, firstChar is the first character that will be printed for the AI vars
        stringprint=lambda botPointer, firstChar : ("%s Training Duration: %s, Refresh: %s, First Hidden: %d, Second Hidden: %d, Memory: %d, Memory Type: %s, Loss Metric: %s(%d)\n"
                                                        %(firstChar, ('0' if botPointer.automationTrainingDuration<=0 else f'{botPointer.automationTrainingDuration}, Algorithm: {botPointer.trainerClass.___Name___()}'),
                                                        ('0' if botPointer.botRefreshCyclePeriod<=0 else f'{botPointer.botRefreshCyclePeriod}'),botPointer.layer1hiddenStateSize, botPointer.layer2hiddenStateSize,
                                                        botPointer.memoryLength,('Change' if botPointer.memoryType else 'Move'),metrics.get(botPointer.lossMetric),botPointer.lossMetric))
        f.write(stringprint(primaryBot,'P'))
        if botVbotBoolean:
            f.write(stringprint(secondaryBot,'S'))
            f.write("Bot V Bot Rounds: "+str(primaryBot.botVbotDuration)+"\n")
        f.write(f"Discard Interval: {primaryBot.collectionDiscardInterval}\nNumber of Games: {numberOfGames}\n\n")
        f.close()
        try: #If current output is in the que, remove it and add it to end. Else, pop the oldest output and add the current one.
            outputQue.remove(fileName)
        except ValueError as error:
            r=42
        outputQue.append(fileName)
        f=open(getcwd()+"\\RPSbotWD\\RPSbotMeta\\outputHist.txt",'w') #Save the output history every time. Excessive, but eh.
        for i in outputQue:
            f.write(i+"\n")
        f.close()
    #The game loops
    if(botVbotBoolean): #Bot V bot mode makes use of threads, one for each bot.
        for i in range(1 if not dataCollection else numberOfGames):
            if dataCollection:
                print(f"Running game {i+1}")
            t1=threading.Thread(target=primaryBot.mainLoop)
            t2=threading.Thread(target=secondaryBot.mainLoop)
            t2.start()
            t1.start()
            t1.join()
            t2.join()
    else: #Normal mode just runs the bot in the main thread
        for i in range(1 if not dataCollection else numberOfGames):
            if dataCollection:
                print(f"Running game {i+1}")
            primaryBot.mainLoop()
    print("ENDING AI")

#trainerDict contains each found trainer class as a string, and indexed according to discovery order. R indexes the number using the name.
trainerDict={0:"defaultTrainer"}
trainerDictR={"defaultTrainer":0}

#A dictionary for the loss metrics. Used just for printing.
metrics={1:"Predict Throw",
             2:"Direction Mirroring",
             3:"Predict Direction",
             4:"PM Outcome",
             5:"BM Outcome",
             6:"PM Outcome 2"}
#The program runs a few functions that normally print some kind of output. This boolean prevents that printing.
initer=True

##The main menu loop.
##Globals listed in function.
def main():
    global targetBot
    global trainerDict, trainerDictR, metrics
    global primaryBot,secondaryBot
    global initer,dataCollectionPath
    #Initialize the two bots
    secondaryBot=daBot(isT=True)
    primaryBot=daBot()
    targetBot=primaryBot
    primaryBot.trainerClass=eval("defaultTrainer()")
    #Opens the file user is running the AI from, and reads through it searching for trainer classes
    trainerSource=open(__main__.__file__, "r")
    line=trainerSource.readline()
    numTrain=1
    while line!="":
        if(len(line)>19 and line[0:6]=="class " and line[-14:-1]=="(RPStrainer):"):
            trainTrueName=line[6:-14]
            line=trainerSource.readline()
            if(line[-22:-1]=="def ___Name___(self):"):
               line=trainerSource.readline()
               trainName=line[line.index("\"")+1:len(line)-2]
               if trainName==trainTrueName:
                   trainerDict[numTrain]="__main__."+trainName
                   trainerDictR[trainName]=numTrain
                   numTrain+=1
        line=trainerSource.readline()
    #Initializing directories, and loading savable variables
    try:
        mkdir("RPSbotWD")
    except OSError as error:
        r=2
    try:
        mkdir("RPSbotWD\\RPSbotSaves")
    except OSError as error:
        r=2
    try:
        mkdir("RPSbotWD\\RPSbotDataCollection")
    except OSError as error:
        r=2
    try:
        mkdir("RPSbotWD\\RPSbotMeta")
    except OSError as error:
        r=2
    try:
        open(getcwd()+"\\RPSbotWD\\RPSbotMeta\\saveDir.txt",'x')
        ff=open(getcwd()+"\\RPSbotWD\\RPSbotMeta\\saveDir.txt",'w')
        ff.write(dataCollectionPath+"\n")
        ff.close()
    except OSError as error:
        ff=open(getcwd()+"\\RPSbotWD\\RPSbotMeta\\saveDir.txt",'r')
        dataCollectionPath=ff.readline()[:-1]
        ff.close()
    try:
        open(getcwd()+"\\RPSbotWD\\RPSbotMeta\\outputHist.txt",'x')
    except OSError as error:
        ff=open(getcwd()+"\\RPSbotWD\\RPSbotMeta\\outputHist.txt",'r')
        filee=ff.readline()
        while filee!='':
            outputQue.append(filee[:-1])
            filee=ff.readline()
        ff.close()
    try:
        open(getcwd()+"\\RPSbotWD\\RPSbotSaves\\default.txt",'x')
        saveParams("default")
    except OSError as error:
        r=2
    try:
        loadParams("default")
    except:
        print("Error: invalid or corrupted default save. Applying fix. Some settings will be returned to factory default.\n\n==============================")
        open(getcwd()+"\\RPSbotWD\\RPSbotSaves\\default.txt",'w').close()
        saveParams("default")
        loadParams("default")
    initer=False

    #function dictionary. Lambda is basially just an if statement, and does not actually take any input
    refer={'d':setDataCollects,
           'a':setAutomation,
           'r':setBotRefresh,
           'p':setNetStructure,
           's':saveParams,
           'l':loadParams,
           'b':lambda boo : 0 if not botVbotBoolean else swapTargetBot(),
           'g':startGame}
    MoveRecString={0:"OFF",1:"Single line",2:"Columns",3:"Running Total"}
    LM=metrics.get(primaryBot.lossMetric)
    LM2=metrics.get(secondaryBot.lossMetric)
    condensedMemTypePrint=lambda tarBot: "%-27.27s"%(('Change' if tarBot.memoryType else 'Move'))
    #Cont vars are used as collapsed parameter lists. If a feature is turned off, then the parameters dictating that features behavior won't show up on the top menu.
    autoCont=f"\nTraining Duration: {smartFormatter('automationTrainingDuration','21')}\nAuto Trainer Algorithm: {smartFormatter('trainerClass.___Name___()','16')}" if (automationBoolean or (automationBooleanSec and botVbotBoolean)) else " "
    dataCont=(f"\nNumber of Games: {numberOfGames}\nData Discard Interval: {'OFF' if not primaryBot.collectionDiscardInterval else primaryBot.collectionDiscardInterval}\nOutput File Destination: {fileName}\nDelimiter: {Deliminator}"+
    f"\nSplits: {(dataCollectionSplits if dataCollectionSplits>0 else 'OFF')}\nMove Record: {MoveRecString.get(dataCollectionMoveRecord)}") if dataCollection else " "
    refreCont=f"\nRefresh Period: {smartFormatter('botRefreshCyclePeriod','24')}" if (botRefreshCycleBoolean or (botRefreshCycleBooleanSec and botVbotBoolean)) else " "
    versusCont=f"\nNumber of Rounds: {abs(primaryBot.botVbotDuration)}" if botVbotBoolean else " "
    
    print("TOP MENU\nThis is the Rock Paper Scissors AI launch menu. Enter various options to load or set parameters, or start the AI.\n"+
            "To run multiple commands in a row, type out each command seperated by a comma, then hit enter. To set variables without leaving the Top menu, type the command for the desired "+
            "menu, and then follow it with the command (inbetween quotaion marks) that you want to execute from that sub menu.\n"+
            "To set the Automatic Trainer parameters, enter A. To set the Data Collection parameters, enter D. To set the Bot Refresh parameters, enter R. To set the parameters that define the bot's behavior, enter P. "+
            "To begin the AI, enter G. To end the program, enter E.\n"+
            "To save the current setting, enter S followed by the name of the destination file. To load previously saved settings, enter L followed by the file to load from. To see what "+
            "files already exist, enter S or L followed by no input.\n\n"+
            f"Auto Training Enabled: {smartFormatter('automationBoolean','17')}{autoCont}\n\nBot Versus Bot mode active: {str(botVbotBoolean)}{versusCont}\n\n"+
            f"Data Collection Enabled: {str(dataCollection)}{dataCont}\n\nRefreshing Enabled: {smartFormatter('botRefreshCycleBoolean','20')}\n\n"+
            f"First Hidden State Size: {smartFormatter('layer1hiddenStateSize','15')}\nSecond Hidden State Size: {smartFormatter('layer2hiddenStateSize','14')}\n"+
            f"Memory Length: {smartFormatter('memoryLength','25')}\nMemory Type: {condensedMemTypePrint(primaryBot)+('  Secondary: '+condensedMemTypePrint(secondaryBot) if botVbotBoolean else '')}\n"+
            f"Current Loss Metric: {LM}"+("" if not botVbotBoolean else f"  Secondary: {LM2}")+"\n\n"+
            "(A)utomatic Trainer, (D)ata Collection, Bot (R)efresh, Bot (P)arameters, (G)o!, (S)ave, (L)oad, "+("Swap Target (B)ot, " if botVbotBoolean else "")+"(E)nd")
    inn=input("\nEnter your command: ")+" " #Extra space is corner case again. Probably.
    innSplit=settingsParseBatch(inn)
    i=0
    while innSplit[i][0].lower()!='e': #The main loop
        fun=refer.get(innSplit[i][0].lower())
        if(type(fun))!=type(refer.get(' ')):
            fun(innSplit[i][1:].strip())
        else:
            print("Invalid command")
        i+=1
        if(i>=len(innSplit)):
            print("\n==============================\nTOP MENU")
            if botVbotBoolean:
                print(f"\nCurrent Target bot: {'Primary bot' if targetBot==primaryBot else 'Secondary bot'}")
            i=0
            LM=metrics.get(primaryBot.lossMetric)
            LM2=metrics.get(secondaryBot.lossMetric)
            autoCont=f"\nTraining Duration: {smartFormatter('automationTrainingDuration','21')}\nAuto Trainer Algorithm: {smartFormatter('trainerClass.___Name___()','16')}" if (automationBoolean or (automationBooleanSec and botVbotBoolean)) else " "
            dataCont=(f"\nNumber of Games: {numberOfGames}\nData Discard Interval: {'OFF' if not primaryBot.collectionDiscardInterval else primaryBot.collectionDiscardInterval}\nOutput File Destination: {fileName}\nDelimiter: {Deliminator}"+
            f"\nSplits: {(dataCollectionSplits if dataCollectionSplits>0 else 'OFF')}\nMove Record: {MoveRecString.get(dataCollectionMoveRecord)}") if dataCollection else " "
            refreCont=f"\nRefresh Period: {smartFormatter('botRefreshCyclePeriod','24')}" if (botRefreshCycleBoolean or (botRefreshCycleBooleanSec and botVbotBoolean)) else " "
            versusCont=f"\nNumber of Rounds: {abs(primaryBot.botVbotDuration)}" if botVbotBoolean else " "
            print(f"\nAuto Training Enabled: {smartFormatter('automationBoolean','17')}{autoCont}\n\nBot Versus Bot mode active: {str(botVbotBoolean)}{versusCont}\n\n"+
            f"Data Collection Enabled: {str(dataCollection)}{dataCont}\n\nRefreshing Enabled: {smartFormatter('botRefreshCycleBoolean','20')}{refreCont}\n\n"+
            f"First Hidden State Size: {smartFormatter('layer1hiddenStateSize','15')}\nSecond Hidden State Size: {smartFormatter('layer2hiddenStateSize','14')}\n"+
            f"Memory Length: {smartFormatter('memoryLength','25')}\nMemory Type: {condensedMemTypePrint(primaryBot)+('  Secondary: '+condensedMemTypePrint(secondaryBot) if botVbotBoolean else '')}\n"+
            f"Current Loss Metric: {'%-19.19s'%(LM)}"+("" if not botVbotBoolean else f"  Secondary: {LM2}")+"\n\n"+
            "(A)utomatic Trainer, (D)ata Collection, Bot (R)efresh, Bot (P)arameters, (G)o!, (S)ave, (L)oad, "+("Swap Target (B)ot, " if botVbotBoolean else "")+"(E)nd")
            inn=input("\nEnter your command: ")+" "
            innSplit=settingsParseBatch(inn)


