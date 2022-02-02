'''
Written by A.B.Koku & I.Ozcil
Jan 28 2022
'''

# start with imports
import numpy as np
from skimage import io 
import random
import matplotlib.pyplot as plt
import random
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import signal

# continue defining parameters
# maze parameters
nCorr = 8
# assume that corridors and bases have the same width/height
boxSize = 50
maxStep = 2 * boxSize # maximum number of pixels a player can travel in one call
'''
colors are defined in the following dictionary
Each dicitonary entry is a list as follows:
((R,G,B), base_value, number of time it is repeated in the initial maze)
'''
colorz = {
        'black':((1,1,1), 0, 13),
        'clr100':((225, 1, 1), 100, 1),
        'clr50':((1, 255, 1), 50, 2), 
        'clr30':((1, 1, 255), 30, 2),
        'clr20':((200, 200, 1), 20, 2),
        'clr10':((255, 1, 255), 10, 2), 
        'clr9':((1, 255, 255), 9, 3),
        'clr8':((1,1,150), 8, 3),
        'clr7':((120,120,40), 7, 3),
        'clr6':((150,1,150), 6, 3),
        'clr5':((1,150,150), 5, 3),
        'clr4':((222,55,222), 4, 3),
        'clr3':((1, 99, 55), 3, 3),
        'clr2':((200, 100, 10),2, 3),
        'clr1':((100, 10, 200),1, 3)
}
pathPen = (255,0,0) # red spared for drawing paths
# initial points in the game for each player
Pinit = 301

maxTime = 0.2 # number of seconds allowed for a player to return from run()

# derived variables
imSize = nCorr * 2 * boxSize - boxSize
hw = int(boxSize / 2) # half of a box not really need but anyway

# 1-9 digits are used to label different paths on user responses
# let's generate digit images
digits = []
digit_size = (7,5)
base_image = np.full(digit_size, 255, dtype=np.uint8)
for d in range(10):
    fname = f'd{d}.png'
    img = Image.fromarray(base_image)
    ImageDraw.Draw(img).text((0,-2), str(d), 0)
    img.save(fname)
    digits.append(io.imread(fname))

# set up timers to limit execution time for run() function calls
# setting timeout values for players and game engine
timeout_for_game = 1e6
timeout_for_players = 0.2

# for alarm handling, timeout handler function is defined
def timeout_handler(sign, fr):
  '''
  this function handles the timeout cases which took longer than 0.2 seconds
  for processes took longer than timeout value, it raises exception
  '''
  # a print as timeout error
  print('timeout has occured')
  raise Exception()

# a timer to SIGALRM variable of signal module is set
signal.signal(signal.SIGALRM, timeout_handler)
# timer is set here using timeout value defined at top
signal.setitimer(signal.ITIMER_REAL, timeout_for_game)

# finally a utility function before classes
def printIF(mess, printIt):
    '''
    pretty straight forward, but very useful, but time consuming, 
    consinder not using at all if performance is your ultimage issue, 
    and you make zillions of calls to this function
    '''
    if printIt:
        print(mess)


# a class to manage bases, a bases is nothing but one of those colored squres on the maze
class aBase():
    digits = []
    def __init__(self, name, bbCorner, size, color, points):
        self.name = name
        self.bbCorner = bbCorner
        self.size = size
        self.color = color
        self.points = points
        self.guests = []
        # define the image for points
        # c1 = np.concatenate((digits[1], digits[0], 255*np.ones((7,1)), digits[0]), axis = 1)
        ds = {f'{d}':d for d in range(10)} # get a string key for a digit from 0 to 9
        pstr = str(points)
        blank = 255 * np.ones((digits[0].shape[0], 1))
        pimg = blank.copy()
        for d in pstr: 
            pimg = np.concatenate((pimg, digits[ds[d]], blank), axis=1)
        baseimg = self.color * np.ones((*pimg.shape,3), np.uint8)
        baseimg[np.where(pimg == 0)] = 255 # paint the base image with the digit with white
        self.pointsImage = baseimg
    
    def paint(self, img, ShowPoints = False):
        '''
        paints itself on the image
        '''
        # base index
        y,x = self.bbCorner[0], self.bbCorner[1]
        # from base index to bounding box coordinates
        by = y * 2 * self.size - self.size
        bx = x * 2 * self.size - self.size 
        # draw box on img
        img[by : by + self.size, 
            bx : bx + self.size,:] = self.color
        # draw points for base
        if ShowPoints:
            #img[yL:yL+h, xL:xL+w, 1] = header 
            dh, dw, dummy  = self.pointsImage.shape
            dy = by + int((self.size - dh)/2)
            dx = bx + int((self.size - dw)/2)
            img[dy:dy+dh, dx:dx+dw, :] = self.pointsImage

    def __back2black__(self):
        # someone conquered this bases, points are granted, goes back to black
        self.name = 'black'
        self.points = 0
        self.color = (0,0,0)
    
    def regsiterEntry(self, pk, speed):
        # register those who enter the base at the same time step
        # then the winner will be the one with maximum step
        self.guests.append({'ID':pk, 'Remaining': speed})

    def andTheWinnerIs(self):
        # return winner if there is one
        if len(self.guests) > 0:
            resSorted = sorted(self.guests, key = lambda d: d['Remaining'])
            winner = resSorted[-1]
            pointsWon = self.points
            self.__back2black__()
            return winner['ID'], pointsWon
        else:
            return 'John Doe', -1

    def __str__(self):
        # enable self printing
        return f'{self.name}: with color {self.color} at {self.bbCorner}'

class daMaze():
    '''
    This is the main class that manages the maze that is composed of bases
    '''
    def __init__(self, numCorridors, colorz):
        self.nCorr = numCorridors
        self.colorz = colorz
        self.bases = {}
        #bases will have points in between [1,...,9] x 3, [10, 20, 30, 50] x 2 , [100] x 1
        #cmask = np.ones((imSize,imSize,3), dtype=np.uint8) * 255 # mask is white
        # get colors    
        # generate all base indices and shuffle their order
        coordz = [(i,j) for i in range(1,self.nCorr) for j in range(1,self.nCorr)]
        random.shuffle(coordz)
        # color bases
        for Dclr in self.colorz.keys():
            clr, clrN = self.colorz[Dclr][0], self.colorz[Dclr][2] # get color and its base value using the current key
            # color as many times as this color needs to show up in the maze
            for i in range(clrN):
                cind = coordz.pop()
                self.bases[f'{cind[0]},{cind[1]}'] = aBase(name = Dclr, bbCorner = cind, size= boxSize, color = clr, points = self.colorz[Dclr][1])


    def paint(self, img, ShowPoints = False):
        # paints the maze on the given image: img
        for baseK in self.bases:
            self.bases[baseK].paint(img, ShowPoints)
    
    def registerPath(self, img, path, playerKey, speed):
        '''
        path is registered for player playerKey with speed
        '''
        basecolors= [ np.array([0,0,0]), np.array([255,255,255])]
        newcolors = [np.array([-1,-1,-1])]
        for i in range(len(path)-1):
            y1,x1 = path[i][0], path[i][1]
            y2,x2 = path[i+1][0], path[i+1][1]
            if x1 == x2: # moving along Y
                ind = [[y, x1] for y in range(y1+np.sign(y2-y1), y2, np.sign(y2-y1))]
            else: # moving along X
                ind = [[y1, x] for x in range(x1+np.sign(x2-x1), x2, np.sign(x2-x1))]
            for [yi,xi] in ind:
                speed -= 1 # decrement speed
                pixcolor = img[yi,xi,:]
                if not ((pixcolor == basecolors).all(axis=1)).any(): # if this is a new color register it
                    if not ((pixcolor == newcolors).all(axis=1)).any():
                        yind = int((yi+99)/100)
                        xind = int((xi+99)/100) 
                        bKey = f'{yind},{xind}'
                        # register this user in the base
                        self.bases[bKey].regsiterEntry(playerKey, speed)
                        newcolors.append(pixcolor)
                else: # back on no point point region
                    newcolors = [np.array([-1,-1,-1])]

    def AnnounceWinners(self):
        # those who made a move that is worth some points are listed and returned
        res = {}
        for baseK in self.bases:
            pk, pt = self.bases[baseK].andTheWinnerIs()
            if pt > -1: # this is not john doe
                res[pk] = pt
        return res

    def DrawPolyLine(self, img, cList, header = None, dpen = (255,0,0)):
        '''
        draws multiple lines using the given list of points, i.e. cList, on to the image: img
        header is used for printing the digit IDs of groups
        dpne is the pen color
        '''
        for i in range(len(cList)-1):
            P1, P2 = cList[i], cList[i+1]
            maxNP = max(( abs(P2[0]-P1[0]) , abs(P2[1]-P1[1])))
            img[ np.linspace(P1[0],P2[0], maxNP).astype('int'), np.linspace(P1[1],P2[1], maxNP).astype('int'), :] = dpen
        # once line is onde if header is not None, draw the header
        if header is not None: # draw the header
            h,w = header.shape
            yL, xL = cList[-1] # coordinates of the last point
            H, W, dummy = img.shape
            if yL + h >= H:
                yL = H - h - 2
            if xL + w >= W:
                xL = W - w - 2
            img[yL:yL+h, xL:xL+w, 1] = header 

    def GenerateInfo(self, Players):
        '''
        Strips the internal data structure that keeps track of players into a verion
        that can be sent to the run() function of players
        '''
        info = {}
        for player in Players:
            info[player] = Players[player][2:]
        return info

    def TrimPath(self, path, L, debugMode = False):
        '''
        path is the candidate path, i.e a list of points [[y1,x1], ... [yn,xn]]
        the function assumes that [y1,x1] is where the agent currently is, 
        and hence [y2,x2] is the first point to move
        L is the total distance that the agent can cover
        function returns the path trimmed so that it has a length of L
        '''
        # first veryfy path
        res = [path[0]] # initially coordinate is approved by default, others will be added if they are valid and within the span of L
        distRemaining = L # originially this is the distance to be covered by the send path
        for i in range(len(path)-1): # using pairs move on the path
            # move on only any distance to move is left
            printIF(f'remaining distance: {distRemaining}', debugMode)
            if distRemaining <= 0:
                return res
            p1, p2 = path[i], path[i+1] # get two consequitive point coordinates
            y1, x1 = path[i][0], path[i][1]
            y2, x2 = path[i+1][0], path[i+1][1]
            dy = y2 - y1 #p2[0] - p1[0]
            dx = x2 - x1 #p2[1] - p1[1]
            # one of these has to be zero on a N4 path
            if abs(dx) >0 and abs(dy)>0: # we have a problem, on eof them has to be zero on a N4 path
                # just return the valid path found so far
                return res
            pathL = max(abs(dy), abs(dx)) # length between p1-p2
            if pathL <= distRemaining: # this part of the path (p1 to p2) completely belongs to the resulting path
                res.append(p2)
                distRemaining -= pathL
                printIF(f'moving {pathL} from {[y1,x1]} to {[y2,x2]}', debugMode)
            else: # this is the tricky part, some part of the path will belong
                # partial path should expand either in X or Y direction
                # note that either dx or dy has to be zero at all times
                if abs(dx) > 0: # going in X direction
                    res.append([y1, x1+np.sign(dx)*distRemaining])
                    printIF(f'moving X {np.sign(dx)*distRemaining} from {[y1,x1]} to {[y1,x1+np.sign(dx)*distRemaining]}', debugMode)
                else: # going in Y direction
                    res.append([y1+np.sign(dy)*distRemaining, x1])
                    printIF(f'moving Y {np.sign(dy)*distRemaining} from {[y1,x1]} to {[y1+np.sign(dy)*distRemaining,x1]}', debugMode)
                return res
        return res