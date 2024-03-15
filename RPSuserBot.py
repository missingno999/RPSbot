import rockPaperScissorsBotV1
from rockPaperScissorsBotV1 import RPStrainer, main
import random

#This trainer simply plays loop where change is always the same. Very easy to beat
class SimpleLoop(RPStrainer):
    def ___Name___(self):
        return "SimpleLoop"
    def computeFirstMove(self):
        return self.computeMove(0,0,0)
    def computeMove(self, playerPrevMove, BotPrevMove,outcome):
        if(playerPrevMove==0):#r
            return 1
        if(playerPrevMove==1):#p
            return 2
        if(playerPrevMove==2):#s
            return 0

#A more responsive version of the basic loop. Uses outcome to decide if it should alter it's pattern.
#Still fairly easy
class SimpleResponsiveLoop(RPStrainer):
    def ___Name___(self):
        return "SimpleResponsiveLoop"
    def computeMove(self, playerPrevMove, BotPrevMove,outcome):
        #tie=0, playerwin=1, botwin=2
        if(outcome==0):
            return playerPrevMove
        if(outcome==1):
            if(playerPrevMove==0):
                return 1
            if(playerPrevMove==1):
                return 2
            if(playerPrevMove==2):
                return 0
        if(playerPrevMove==0):
            return 2
        if(playerPrevMove==2):
            return 1
        if(playerPrevMove==1):
            return 0

#Another loop bot that reverses it's pattern on a loss. However, the symetrical nature of the pattern makes it harder for the bot to make accurate predictions.
#To see this in action, try using this trainer on a bot with a memory length of 1, versus a bot with a memory length of two or three.
class OverlapResponsiveLoop(RPStrainer):
    def ___Name___(self):
        return "OverlapResponsiveLoop"
    def preGameAssignment(self):
        self.trainerPosi=0
        self.trainerPattern=[0,1,2,2,1,0]
    def computeFirstMove(self):
        return self.computeMove(0,0,0)
    def computeMove(self, playerPrevMove, BotPrevMove,outcome):
        if(outcome==2):
            self.trainerPosi-=1
            if self.trainerPosi<0:
                self.trainerPosi=5
        else:
            self.trainerPosi+=1
            if self.trainerPosi==6:
                self.trainerPosi=0
        return self.trainerPattern[self.trainerPosi]

#A simple trainer with a that loops through a list, and every time the list ends, the bot throws one random move then restarts the list.
class PredictableRandomLoop(RPStrainer):
    def ___Name___(self):
        return "PredictableRandomLoop"
    def preGameAssignment(self):
        self.trainerPosi=0
        self.trainerPattern=[0,1,2,2,1,0]
    def computeMove(self, playerPrevMove, BotPrevMove,outcome):
        if self.trainerPosi==3:
            return super().randThrow()
        else:
            self.trainerPosi+=1
            return self.trainerPattern[self.trainerPosi-1]

#A function to demonstrate a trainer with configurable parameters at initialization.
#Is initialize with an arbitrary pattern that it loops through without any kind of responsivness
class customLoop(RPStrainer):
    def ___Name___(self):
        return "customLoop"
    def patternParse(self,inne):
        for i in inne:
            if(i=='r'):
                self.pattern.append(0)
            if(i=='p'):
                self.pattern.append(1)
            if(i=='s'):
                self.pattern.append(2)
    def __init__(self):
        inne=super().getUserInput("Enter a string of R,P,S: ").lower()
        self.pattern=[]
        self.patternParse(inne)
        while(len(self.pattern)==0):
            inne=super().getUserInput("Invalid pattern, please enter new one: ").lower()
            self.patternParse(inne)
    def preGameAssignment(self):
        self.trainerPosi=-1
    def computeFirstMove(self):
        return self.computeMove(0,0,0)
    def computeMove(self, playerPrevMove, BotPrevMove, outcome):
        self.trainerPosi+=1
        if self.trainerPosi>=len(self.pattern):
            self.trainerPosi=0
        return self.pattern[self.trainerPosi]
            
#This trainer chooses it's move purly based off of the AI's change over the last two rounds
class botChangeResponse(RPStrainer):
    def ___Name___(self):
        return "botChangeResponse"
    def preGameAssignment(self):
        self.botHistory=-1
        self.pastSecondMove=False
    def computeMove(self, playerPrevMove, BotPrevMove, outcome):
        if not self.pastSecondMove:
            self.pastSecondMove=True
            self.botHistory=BotPrevMove
            return super().randThrow()
        else:
            change=super().getChange(self.botHistory,BotPrevMove)
            self.botHistory=BotPrevMove
            return change

#A trainer that acts as a crude model of human play behavior.
#Roughly based off the data from this paper:
##Dyson, B., Wilbiks, J., Sandhu, R. et al. Negative outcomes evoke cyclic irrational decisions in Rock, Paper, Scissors.
##Sci Rep 6, 20479 (2016). https://doi.org/10.1038/srep20479
class simulatedCycleBehavior(RPStrainer):
    def ___Name___(self):
        return "simulatedCycleBehavior"
    def preGameAssignment(self):
        self.curBehavior=-1 #-1 means neutral 
    def computeMove(self, playerPrevMove, BotPrevMove, outcome):
        hold=0
        upgrade=0
        downgrade=0
        if(outcome==1):
            hold=0.36
            upgrade=0.31
            downgrade=0.33
        elif self.curBehavior==-1:
            if(outcome==0):
                hold=0.26
                upgrade=0.43
                downgrade=0.31
            else:
                hold=0.22
                upgrade=0.36
                downgrade=0.42
        else:
            if self.curBehavior==0:
                hold=0.29
                upgrade=0.35
                downgrade=0.36
            if self.curBehavior==1:
                hold=0.30
                upgrade=0.40
                downgrade=0.30
            if self.curBehavior==2:
                hold=0.30
                upgrade=0.30
                downgrade=0.40            
        jk=random.random()
        if jk<=hold:
            self.curBehavior=0
            return playerPrevMove
        if jk<=(hold+upgrade):
            self.curBehavior=1
            return (playerPrevMove+1)%3
        else:
            self.curBehavior=2
            return (playerPrevMove+2)%3

#The exact same idea as the previous one, except with exaggerated probabilities (making it more predictable)
class simulatedCycleBehaviorExag(RPStrainer):
    def ___Name___(self):
        return "simulatedCycleBehaviorExag"
    def preGameAssignment(self):
        self.curBehavior=-1 #-1 means neutral 
    def computeMove(self, playerPrevMove, BotPrevMove, outcome):
        hold=0
        upgrade=0
        downgrade=0
        if(outcome==1):
            hold=0.33
            upgrade=0.33
            downgrade=0.33
        elif self.curBehavior==-1:
            if(outcome==0):
                hold=0.14
                upgrade=0.55
                downgrade=0.31
            else:
                hold=0.09
                upgrade=0.36
                downgrade=0.55
        else:
            if self.curBehavior==0:
                hold=0.19
                upgrade=0.40
                downgrade=0.41
            if self.curBehavior==1:
                hold=0.17
                upgrade=0.53
                downgrade=0.20
            if self.curBehavior==2:
                hold=0.18
                upgrade=0.20
                downgrade=0.52            
        jk=random.random()
        if jk<=hold:
            self.curBehavior=0
            return playerPrevMove
        if jk<=(hold+upgrade):
            self.curBehavior=1
            return (playerPrevMove+1)%3
        else:
            self.curBehavior=2
            return (playerPrevMove+2)%3

#>bot tracks opponent's moves
#>Plays random moves, adjusted for frequency
#>IF the bot notices a pattern in the opponent's moves, it attempts to exploit it
#>PATTER: 


main()
