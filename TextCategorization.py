# Authors: Diego Holguin
# Purpose: To test how well a neural network is able to classify article's topics based solely on the contents of each article
# Dataset Used: http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html

import os
import glob
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense, Activation
from HTMLParser import HTMLParser
import csv

#region Neural Netowrk Variables
seedValue = 21 # Used for seeding the random weights for debugging purposes
EPOCHS_COUNT = 20 # number of rounds of training
K_FOLD = 10 # The number of k folds that are done
BATCH_SIZE = 20 # The number of iterations before back propagation occurred
TOP_WORDS_COUNT = 250 # 250
INPUT_NODES_COUNT = 500 # Number of input nodes
HIDDEN_NODES_COUNT = 100 # Number of output Nodes
NN_HIDDEN_LAYERS_COUNT = 5 # Number of hidden layers
NN_INPUT_FUNCTION = 'sigmoid' # Function used for the input nodes
NN_HIDDEN_FUNCTION = 'sigmoid' # Function used for the hidden nodes
NN_OUTPUT_FUNCTION = 'sigmoid' # Function used for the output nodes
MAX_TOPICS_ALLOWED = 200  # The max number of topics that tbe program stores

wordsAndCount = {} # container that maps every distinct word to their individual count variable
allTopics = {} # list of all of the topics within every .sgm file
topicsPerformance = [] # dictionary of each topic and the number of TP FP TN FN
topWords = [] # list of the top X most frequent words found within all of the .sgm files
inputVectors = [] # list of each articles input vector
outputVectors = [] # list of each articles output vector
#endregion

# Class that holds each instance of a topic's number of TP, FP, TN, FN, and MCC values
class TopicPerformance(object):
    def __init__(self):
        self.name = ""
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.MCC = 0

    def IncrementTP(self,):
        self.TP += 1
    def IncrementFP(self):
        self.FP += 1
    def IncrementTN(self):
        self.TN += 1
    def IncrementFN(self):
        self.FN += 1
    def CalculateMCC(self):
        numerator = (self.TP * self.TN) - (self.FP * self.FN)
        denominator = (self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN)
        if denominator == 0:
            denominator = 1
        self.MCC = numerator/(math.sqrt(denominator))
    def PrintData(self):
        print "TP ", self.TP, " TN ", self.TN, " FP ", self.FP, " FN ", self.FN
    def SetName(self, string):
        self.name = string
    def IncrementByClass(self, _tp, _tn, _fp, _fn):
        self.TP += _tp
        self.TN += _tn
        self.FP += _fp
        self.FN += _fn

# Class that handles the parsing process during the second pass through the sgm files
class SecondPassParser(HTMLParser):
    def __init__(self, data):
        # Init instance variables
        HTMLParser.__init__(self)
        self.sgmContents = data
        self.bodyText = ""
        self.bodyStartIndex = 0
        self.bodyEndIndex = 0
        self.reutersStartIndex = 0
        self.dateStartIndex = 0
        self.topicsStartIndex = 0
        self.topicsEndIndex = 0
        self.topicList = []
        self.feed(data)

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self.bodyStartIndex = self.getpos()[1] + 6
        elif tag == "reuters":
            self.reutersStartIndex = self.getpos()[1]
        elif tag == "date":
            self.dateStartIndex = self.getpos()[1]
        elif tag == "topics":
            self.topicsStartIndex = self.getpos()[1]

    def handle_endtag(self, tag):
        if tag == "body":
            self.bodyEndIndex = self.getpos()[1]
            self.ParseArticle()
            self.bodyEndIndex = 0
        elif tag == "topics":
            self.topicsEndIndex = self.getpos()[1]

    def obtainBodyText(self, data):
        return data[self.bodyStartIndex:self.bodyEndIndex]

    def ParseArticle(self):
        # Fist obtain the topc string for the sgm file
        reuterString = str(self.sgmContents[self.reutersStartIndex:self.dateStartIndex])

        # If the topic string contains the proper headers
        if ((reuterString.find("""LEWISSPLIT='TRAIN'""") or reuterString.find(
                """LEWISSPLIT='TEST'"""))) and reuterString.find("""TOPICS='YES'"""):

            wordsAsList = [0] * len(topWords)
            topicAsList = [0] * len(allTopics)

            # Determine which topics are active against all of the topics in a binary representation
            # Obtain the string with the articles topic data
            topicString = self.sgmContents[self.topicsStartIndex:self.topicsEndIndex]

            # Parse out meaningful data within the topic string
            while topicString.find("""<D>""") != -1:
                beginOfTopic = topicString.find("""<D>""") + 3;
                endOfTopic = topicString.find("""</D>""")

                # Add only if the topic is a new topic
                topicName = topicString[beginOfTopic:endOfTopic].lower()
                if topicName in allTopics:
                    topicAsList[allTopics[topicName]] = 1

                topicString = str(topicString).replace("<D>", "<*>", 1)
                topicString = str(topicString).replace("</D>", "</*>", 1)

            # Obtain the body text associated with the sgm file
            bodyText = self.obtainBodyText(self.sgmContents)

            # Tokenize data from body paragraph
            tokenizedInput = str(bodyText).split()

            # Iterate through each token, ignoring tokens with extraneous data, and determine which words are active against the top thousand words in binary representation
            for singleToken in tokenizedInput:
                # Compare against only relevant data only..
                if ((singleToken.find("&") == -1) and (singleToken.find(">") == -1) and (singleToken.find("<") == -1) and (singleToken.find(";") == -1) and (singleToken.find("the") == -1)
                and (singleToken.find("a") == -1) and (singleToken.find("and") == -1) and (singleToken.find("said") == -1) and (singleToken.find("for") == -1) and (singleToken.find("of") == -1)
                and (singleToken.find("mln") == -1) and (singleToken.find("it") == -1) and (singleToken.find("to") == -1) and (singleToken.find("in") == -1) and (singleToken.find("cts") == -1)):
                    if singleToken.lower() in topWords:
                        wordsAsList[topWords[singleToken.lower()]] = 1

            # print "Binary Representation (words): ", "".join(wordsAsList)
            inputVectors.append(wordsAsList)

            # Append the binary representation of the articles topics compared against all of the topics
            outputVectors.append(topicAsList)

# Class that handles the parsing process during the first pass through the sgm files
class FirstPassParser(HTMLParser):
  def __init__(self, data):

    # Init instance variables
    HTMLParser.__init__(self)
    self.sgmContents = data
    self.bodyText = ""
    self.bodyStartIndex = 0
    self.bodyEndIndex = 0
    self.reutersStartIndex = 0
    self.dateStartIndex = 0
    self.topicsStartIndex = 0
    self.topicsEndIndex = 0

    self.feed(data)

  def handle_starttag(self, tag, attrs):
    if tag == "body":
        self.bodyStartIndex = self.getpos()[1] + 6
    elif tag == "reuters":
        self.reutersStartIndex = self.getpos()[1]
    elif tag == "date":
        self.dateStartIndex = self.getpos()[1]
    elif tag == "topics":
        self.topicsStartIndex = self.getpos()[1]

  def handle_endtag(self, tag):
    if tag == "body":
        self.bodyEndIndex = self.getpos()[1]
        self.ParseArticle()
        self.bodyEndIndex = 0
    elif tag == "topics":
        self.topicsEndIndex = self.getpos()[1]

  def obtainBodyText(self, data):
      return data[self.bodyStartIndex:self.bodyEndIndex]

  def ParseArticle(self):
      # Fist obtain the topc string for the sgm file
      reuterString = str(self.sgmContents[self.reutersStartIndex:self.dateStartIndex])

      # If the topic string contains the proper headers
      if ((reuterString.find("""LEWISSPLIT='TRAIN'""") or reuterString.find(
              """ LESISSPLIT='TEST'"""))) and reuterString.find("""TOPICS='YES'"""):
          topicString = self.sgmContents[self.topicsStartIndex:self.topicsEndIndex]

          # Parse out all topics if any are specified
          while topicString.find("""<D>""") != -1:
              beginOfTopic = topicString.find("""<D>""") + 3;
              endOfTopic = topicString.find("""</D>""")

              # Add only if the topic is a new topic
              topicName = topicString[beginOfTopic:endOfTopic].lower()
              if topicName not in allTopics and len(allTopics) < MAX_TOPICS_ALLOWED:

                  # Create instance of class, setting it's name
                  performanceClass = TopicPerformance()

                  # Save the name of the topic
                  performanceClass.name = topicName

                  # append this unique topic to a list of all topic performances
                  topicsPerformance.append(performanceClass)

                  allTopics[topicName] = len(allTopics)


              topicString = str(topicString).replace("<D>", "<*>", 1)
              topicString = str(topicString).replace("</D>", "</*>", 1)

          # Obtain the body text associated with the sgm file
          bodyText = self.obtainBodyText(self.sgmContents)

          # tokenize, then iterate through the tokenize  text
          tokenizedInput = str(bodyText).split()
          for singleToken in tokenizedInput:

              # Only add relevant text to the dictionary
              if ((singleToken.find("&") == -1) and (singleToken.find(">") == -1) and (singleToken.find("<") == -1) and (
                  singleToken.find(";") == -1) and (singleToken.find("the") == -1) and (singleToken.find("a") == -1) and (singleToken.find("and") == -1) and (
                  singleToken.find("said") == -1) and (singleToken.find("for") == -1) and (singleToken.find("of") == -1) and (singleToken.find("mln") == -1) and (singleToken.find("it") == -1) and (
                  singleToken.find("to") == -1) and (singleToken.find("in") == -1) and (singleToken.find("dlrs") == -1) and (singleToken.find("pct") == -1)
                  and (singleToken.find("dlrs") == -1) and (singleToken.find("pct") == -1) and (singleToken.find("cts") == -1)):

                  # If the key doesnt exist, add it
                  if (str(singleToken).lower() not in wordsAndCount):
                      wordsAndCount[str(singleToken).lower()] = 1

                  # else, just update the word count associated with the string
                  else:
                      value = wordsAndCount.get(str(singleToken).lower())
                      value = value + 1
                      wordsAndCount[str(singleToken).lower()] = value

# FirstPassThroughFiles Class
# -- Purpose: To parse through all of the .sgm files within the project and does the following:
#             1) Tallies up the number of times a specific word was written for each and every article
#             2) Tallies up each unique topic used for each and every article
class FirstPassThroughFiles(object):

    def __init__(self, ):
        self.iterateThroughFiles("reuters21578/")

    def iterateThroughFiles(self, _path):
        for infile in glob.glob(os.path.join(_path, '*.sgm')):
            print "First Pass On: ", str(infile)
            self.parseFile(infile)

    def parseFile(self, _filePath):
        fileContents = open(_filePath, 'r')
        fileData = fileContents.read().replace('\n','')
        FirstPassParser(fileData)

# SecondPassThroughFiles Class
# -- Purpose: To parse through all of the .sgm files within the project and does the following:
#             2) Calculates the binary representation of each articles contents (The set of words within the body paragraph) and the set of the thousand words intersected
#             2) Calculates the binary representation of each articles topics and the set of all topics intersected
class SecondPassThroughFiles(object):

    def __init__(self, ):
        self.iterateThroughFiles("reuters21578/")

    def iterateThroughFiles(self, _path):
        for infile in glob.glob(os.path.join(_path, '*.sgm')):
            print "Pass Two On: ", str(infile)
            self.parseFile(infile)

    def parseFile(self, _filePath):
        fileContents = open(_filePath, 'r')
        fileData = fileContents.read().replace('\n',' ')
        SecondPassParser(fileData)

# Function that writes each unique topic's performance (TP, NP, FP, FN, and MCC) into a csv file
def SaveToCSV():
    # Open/create a results.csv file
    csvFile = open("results.csv", 'wb')

    # Create a writer to the csv file
    writer = csv.writer(csvFile, quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')

    # Then write what each column will contain
    writer.writerow('CATEGORY,TP,TN,FP,FN,MCC'.split(","))

    # Finally, write each topic performance
    for index in range(0,len(topicsPerformance)):

        topicperformance = topicsPerformance[index]

        # Obtain the string representation for the topic's TP, TN, FP, and FN
        nameString = (topicperformance.name)
        tpString = str(topicperformance.TP)
        tnString = str(topicperformance.TN)
        fpString = str(topicperformance.FP)
        fnString = str(topicperformance.FN)

        # Then calculate the MCC values for the topic
        topicperformance.CalculateMCC()

        # Then obtain the string representation of the calculated MCC value
        mccString = str(topicperformance.MCC)

        # Create an array holding all relevant data to be written to the csv row
        topicData = [nameString, tpString, tnString, fpString, fnString, mccString]

        # Write this relevant data to the csv file by row
        writer.writerow(topicData)

if(__name__ == '__main__'):
    # Retrieve the dictionary of all of the words and their respective word counts, as well as a container listing all of the topics parsed through
    FirstPassThroughFiles()

    # Sort words based upon the frequencies, starting from lowest and ending with the most frequent word
    wordsAndCount = sorted(wordsAndCount.iteritems(), key=lambda  (k,v): (v,k))

    # Then reference the top thousand words found within all .sgm files as a list
    topWordsArray = wordsAndCount[-TOP_WORDS_COUNT:]

    # Convert to a dictionary
    topWords = { }
    for i in range(0, len(topWordsArray)):
        topWords[topWordsArray[i][0]] = i

    # Retrieve the words and topics vectors
    articlesData = SecondPassThroughFiles()

    # Initialize seed value, to make this program deterministic
    numpy.random.seed(seedValue)

    # Shuffle the input and output vectors
    idx = range(len(inputVectors))
    numpy.random.shuffle(idx)
    sinput = [0]*len(inputVectors)
    soutput = [0]*len(outputVectors)
    for i in range(len(inputVectors)):
        sinput[i] = inputVectors[idx[i]]
        soutput[i] = outputVectors[idx[i]]

    # Save shuffled vectors
    inputVectors = sinput
    outputVectors = soutput

    # Calculate the how many elements each K-fold value will hold
    kBound = len(inputVectors) / K_FOLD

    for singleKFold in range(1,(K_FOLD + 1)):
        # Create a sequential neural network
        NN = Sequential()

        # Add the input layer
        NN.add(Dense(units=INPUT_NODES_COUNT, input_dim=len(inputVectors[0]), activation=NN_INPUT_FUNCTION))

        # Add a specified number of hidden layers
        for hiddenLayerInstance in range(0,NN_HIDDEN_LAYERS_COUNT):
            NN.add(Dense(HIDDEN_NODES_COUNT, activation=NN_HIDDEN_FUNCTION))

        # Add the output layer
        NN.add(Dense(units=len(outputVectors[0]), activation=NN_OUTPUT_FUNCTION))

        # Compile the network function
        NN.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])

        # Based upon the k fold instance, partition the data sets into appropriate sub sets
        if singleKFold == 1:
            x_test = inputVectors[:kBound]
            y_test = outputVectors[:kBound]
            x_training = inputVectors[kBound:]
            y_training = outputVectors[kBound:]
        else:
            # Get the training sets
            x_test = inputVectors[kBound * (singleKFold - 1):(singleKFold * kBound)]
            y_test = outputVectors[kBound * (singleKFold - 1):(singleKFold * kBound)]

            # Get the test set for the word vectors
            x_train_first_half = inputVectors[:(kBound * (singleKFold - 1))]
            x_train_second_half = inputVectors[(kBound * singleKFold):]
            x_training = x_train_first_half + x_train_second_half

            # Get the test set for the topic vectors
            y_train_first_half = outputVectors[:(kBound * (singleKFold - 1))]
            y_train_second_half = outputVectors[(kBound * singleKFold):]
            y_training = y_train_first_half + y_train_second_half

        # Fit the x and y training sets into the NN, training it
        NN.fit(x_training, y_training, epochs=EPOCHS_COUNT, batch_size=BATCH_SIZE)

        # Now predict the x data set
        classes = NN.predict(x_test, batch_size=1)

        # DEBUGGING
        print "After Training on K=", singleKFold

        for singleClass in range(0,len(classes)): # For each class object within classes

            # Calculate the highest value for each class
            topBound = -100
            for dataIndex in range(0, len(classes[singleClass])):
                if topBound <= classes[singleClass][dataIndex]:
                    topBound = classes[singleClass][dataIndex]

            # Then determine a binary representation for each class's output
            analogThreshold = topBound * 0.44
            for dataIndex in range(0, len(classes[singleClass])):

                # Determine the binary value of the output
                if classes[singleClass][dataIndex] <= analogThreshold:
                    classes[singleClass][dataIndex] = 0
                else:
                    classes[singleClass][dataIndex] = 1

            # Determine the number of TP, TN, FP, and FN, being associated with a specific topic
            for index in range(0, len(classes[singleClass])):
                if classes[singleClass][index] == 1 and y_test[singleClass][index] == 1:
                    topicsPerformance[index].IncrementTP()
                elif classes[singleClass][index] == 0 and y_test[singleClass][index] == 1:
                    topicsPerformance[index].IncrementFN()
                elif classes[singleClass][index] == 1 and y_test[singleClass][index] == 0:
                    topicsPerformance[index].IncrementFP()
                elif classes[singleClass][index] == 0 and y_test[singleClass][index] == 0:
                    topicsPerformance[index].IncrementTN()

    SaveToCSV()