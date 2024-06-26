The main program is rockPaperScissorsBotV1.py. In order to run the program, you should create a new python file, import rockPaperScissorsBotV1, and call main()
An example file called RPSuserBot.py should have been included. The AI can be ran from this file, and it includes a few example RPStrainers.

This program requires the pytorch library. Please make sure it's installed before running.

This program should have come with a txt file labeld "Manual". It is strongly recommended that you read this some before using the AI, and do not delete the file.


!!WARNING!!
This program makes use of the eval() python function, which is considered a insecure function. Estimates suggest that the threat level is low for this particular program, but running the program on public machines is not recommended.


---CHANGE LOG---
3/10/2024
V 1.2

Added two new parameters: Splits and Move Recorder

Added some try-catch statements to ensure that loading data does not crash the program if the file
is outdated or corrupted

Condened score statistics printing down to one function

Small UI adjustments

Minor bug fixes through out

Very mild code reformatting

Added value validation to save data loading

Updated the formatting of the manual



3/15/2024
V 1.2.1

Fixed Move Record so that it works when outputting to a file

Changed how the Automated Trainer's class name is displayed. Now the name won't be cut off if BotVbot mode is off, and class names are always preceeded by their assigned number

Added two new example RPStrainer classes to RPSuserBot.py. These classes are crude models meant to simulate human play. Data used for the bots was taken from this paper:
Dyson, B., Wilbiks, J., Sandhu, R. et al. Negative outcomes evoke cyclic irrational decisions in Rock, Paper,
Scissors. Sci Rep 6, 20479 (2016). https://doi.org/10.1038/srep20479 

Removed a single space from the manual (and updated it to warn users about the trainers' class name being cut off when it's too long to display nicely)

3/29/2020
V 1.2.1B

Didn't change the code, just added a quick start guide

Also added to the manual a link to the PyTorch website for installing
