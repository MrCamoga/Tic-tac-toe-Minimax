import tkinter as tk
import math
import random
from itertools import product
from functools import reduce
from threading import Thread, Event
from copy import deepcopy
from operator import itemgetter
import time

# tamaño de las celdas/tablero
CELLSIZE = 67
BLOCKSIZE = CELLSIZE*3
WIDTH = CELLSIZE*9
HEIGHT = WIDTH
lines = (0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6) # lineas del tres en raya

# TABLERO

class Board():
    def __init__(self, player1=None, player2=None):
        self.board = [[0  for _ in range(9)]  for _ in range(9)]
        self.wBoard = [0  for _ in range(9)]
        self.nextMove = [-1]
        self.lastMove = []
        self.possibleMoves = [set(range(9)) for _ in range(9)]
        self.win = False
        self.turn = 1
        self.players = {1:player1,2:player2}
        self.index13 = [0]*9 # index of small boards base 3
        self.index23 = 0 # index of large board base 3
        self.index24 = 0 # index of large board base 4
        self.pow3 = [3**n for n in range(9)]
        self.pow4 = [4**n for n in range(9)]
        if player1 is not None:
            player1.turn = 1
            player1.board = self
        if player2 is not None:
            player2.turn = 2
            player2.board = self

    def start(self,gui=True,log=True):
        while self.win == 0:
            eventMove.clear()
            self.players[self.turn].play(gui) # si se juega en modo consola se llama a move2
        if log:
            if self.win != 3:
                print("Ha ganado ",self.players[self.win].name)
            else:
                print("Empate")

    # logica del juego  
    def move(self,cell):
        if not self.canMove(cell):
            return False
        self.__setCell(cell,self.turn)
        self.index13[cell[0]] += self.pow3[cell[1]]*self.turn # actualizar indices
        self.possibleMoves[cell[0]].remove(cell[1])
        render.after(0, render.drawMove,cell,self.turn)

        if self.wBoard[cell[0]] != 3: # si el tablero no está empatado se comprueba si se ha ganado
            self.wBoard[cell[0]] = winLUT[self.index13[cell[0]]]
            if self.wBoard[cell[0]] != 0:
                if self.wBoard[cell[0]] != 3:
                    render.after(0,render.winBoard,cell[0],self.turn)
                self.win = self.checkWin(self.wBoard)
        
        self.nextMove.append(-1 if (self.wBoard[cell[1]] in [1,2] or len(self.possibleMoves[cell[1]]) == 0) else cell[1])
        self.__switchTurn()
        render.after(0,render.updateNextMove,cell)
        self.lastMove.append(cell)
        return True

    def move2(self,cell): # mover ia sin render
        self.__setCell(cell,self.turn)
        self.index13[cell[0]] += self.pow3[cell[1]]*self.turn # actualizar indices
        self.possibleMoves[cell[0]].remove(cell[1])
        self.wBoard[cell[0]] = winLUT[self.index13[cell[0]]]
        if self.wBoard[cell[0]] != 0:
            self.win = self.checkWin(self.wBoard)
            
        self.nextMove.append(-1 if (self.wBoard[cell[1]] in [1,2] or len(self.possibleMoves[cell[1]]) == 0) else cell[1])
        self.__switchTurn()
        self.lastMove.append(cell)

    def undomove2(self):
        move = self.lastMove.pop()
        self.__switchTurn()
        self.nextMove.pop()
        self.__setCell(move,0)
        self.index13[move[0]] -= self.pow3[move[1]]*self.turn # actualizar indices
        self.possibleMoves[move[0]].add(move[1])
        self.wBoard[move[0]] = winLUT[self.index13[move[0]]]
        self.win = 0 

    def checkWin(self,board):
        if any(all(i == self.turn for i in itemgetter(*line)(board)) for line in lines):
            return self.turn
        
        if all(bitor(itemgetter(*line)(board))==3 for line in lines): # comprobar si todas las lineas estan empatadas y declarar empate
            return 3
        return 0

    def getCell(self,cell):
        return self.board[cell[0]][cell[1]]

    def __setCell(self,cell,value):
        self.board[cell[0]][cell[1]] = value

    def getTurn(self,neg=0): # obtener turno (neg=1 turno contrario)
        return self.turn^(3*neg)

    def __switchTurn(self):
        self.turn ^= 3

    def canMove(self,cell):
        if 0<=cell[0]<=8:
            if self.nextMove[-1] == -1 or self.nextMove[-1] == cell[0]:
                if self.getCell(cell) == 0 and self.wBoard[cell[0]] in [0,3]:
                    return True
        return False

    def clone(self):
        return deepcopy(self)

    def getPossibleMoves(self):
        if self.win != 0:
            return []
        if self.nextMove[-1] == -1:
            l = [(i,j) for i in range(9) for j in self.possibleMoves[i] if self.wBoard[i] in [0,3]]
        else:
            l = [(self.nextMove[-1],j) for j in self.possibleMoves[self.nextMove[-1]]]
        random.shuffle(l)
        return l

    def getMove(self):
        return len(self.lastMove)//2

# GUI

class Render(tk.Canvas):
    def __init__(self,window,*args,**kwargs):
        tk.Canvas.__init__(self,window,*args,**kwargs)
        
        self.pack(expand=tk.YES, fill=tk.BOTH)
        
        if mouse is not None:
            self.bind('<Button-1>', mouse.mousePressed)
            self.bind('<ButtonRelease-1>', mouse.mouseReleased)
            self.bind('<Motion>', mouse.mouseMoved)

        self.__initGUI()

    def __initGUI(self):
        self.colors = {1:'#ff5533',2:'#1080d0'} # colores de jugadores
        self.hovercolors = {1:'#ff3209',2:'#0d6eb4'}
        
        for i in range(1,9): # rejilla
            col = self.create_line(CELLSIZE*i,0,CELLSIZE*i,HEIGHT,width=1+int(i%3==0))
            row = self.create_line(0,CELLSIZE*i,WIDTH,CELLSIZE*i,width=1+int(i%3==0))

        # marcar cuadrante que se va a jugar en el siguiente movimiento
        self.__forbiddenMove = [self.create_rectangle(*self.__getBigCellCoords(k),tags="forbidden_move",width=0,fill='',stipple='gray25') for k in range(9)]
        self.__nextMove = [self.create_rectangle(*self.__getBigCellCoords(k),tags="next_move",width=0,outline=self.colors[board.getTurn(1)]) for k in range(9)]
        self.__afterID = None
        self.__lastMove = self.create_rectangle(*[-10]*4,width=0,fill="#ffffff")

        self.__selectedCell = self.create_rectangle(*[-10]*4,width=0,fill="") # casilla bajo el cursor

    def __getBigCellCoords(self,cell,scale=0): # obtener coordenadas de cuadrante
        x,y = list(reversed(divmod(cell,3)))
        return (x*BLOCKSIZE-scale,y*BLOCKSIZE-scale,(x+1)*BLOCKSIZE+scale,(y+1)*BLOCKSIZE+scale)

    def __getSmallCellCoords(self,cell,scale=0): # obtener coordenadas de casilla
        y0,x0 = divmod(cell[0],3)
        y1,x1 = divmod(cell[1],3)
        return (x0*BLOCKSIZE+x1*CELLSIZE - scale,y0*BLOCKSIZE+y1*CELLSIZE - scale,x0*BLOCKSIZE+(x1+1)*CELLSIZE + scale,y0*BLOCKSIZE+(y1+1)*CELLSIZE + scale)

    def winBoard(self,cell,turn): # colorear tres en raya con el ganador
        self.create_rectangle(*self.__getBigCellCoords(cell),width=0,fill=self.colors[turn],stipple='gray75')

    def drawMove(self,cell,turn): # colorear casilla seleccionada
        self.create_rectangle(*self.__getSmallCellCoords(cell),width=0,fill=self.colors[turn])
        self.coords(self.__lastMove,*self.__getSmallCellCoords(cell,-10))
        self.tag_raise(self.__lastMove)

    def drawHoverCell(self,cell):
        if board.getCell(cell) == 0:
            coords = self.__getSmallCellCoords(cell,-5)
            self.itemconfig(self.__selectedCell,fill=self.hovercolors[board.getTurn()])
            self.coords(self.__selectedCell,coords)
        else:
            self.itemconfig(self.__selectedCell,fill='')

    # dibujar tableros en los que se puede poner el siguiente movimiento
    def updateNextMove(self,cell): 
        if self.__afterID is not None:
            self.after_cancel(self.__afterID) # si la animacion ya se está ejecutando se cancela y se empieza de nuevo

        # Hacer el movimiento para ver dónde puede jugarse el siguiente movimiento
        newboard = board.clone()
        if newboard.getCell(cell) == 0:
            newboard.board[cell[0]][cell[1]] = board.getTurn()
            newboard.wBoard[cell[0]] = newboard.checkWin(newboard.board[cell[0]])
        
        animatecells = []
        self.tag_raise("forbidden_move")
        for k in range(9):
            canmove = newboard.wBoard[cell[1]] in [0,3] and k==cell[1] or (newboard.wBoard[k] in [0,3] and newboard.wBoard[cell[1]] not in [0,3])
            self.itemconfig(self.__nextMove[k],outline=self.colors[newboard.getTurn(1)],width=5*int(canmove))
            self.itemconfig(self.__forbiddenMove[k],fill='' if (newboard.nextMove[-1]==k or newboard.nextMove[-1]==-1) or newboard.wBoard[k] != 0 else '#404040')
            animatecells += [k]
        self.__animateNextMove(animatecells)

    # animacion de los cuadrantes
    def __animateNextMove(self,cells,time=0,total=800,rate=16):# rate = 16ms (60fps)
        func = lambda t: 3*(-0.5+math.cos(2*math.pi*t))
        f = func(time)
        self.tag_raise("next_move")

        for k in cells:
            self.coords(self.__nextMove[k],*self.__getBigCellCoords(k,f))

        self.__afterID = self.after(16,lambda: self.__animateNextMove(cells,time=time+rate/total))

class Mouse():
    def __init__(self):
        self.lastcell = None
        self.controlling = None

    def mousePressed(self, event):
        pass

    def mouseMoved(self, event):
        if self.controlling is None:
            return
        cell = self.getCell(event)
        
        if 0<=cell[0]<=8 and cell != self.lastcell:
            self.lastcell = cell
            if board.canMove(cell):
                render.updateNextMove(cell)
                render.drawHoverCell(cell)
            else:
                render.drawHoverCell([-1,-1])
                render.updateNextMove([-1,-1])

    def mouseReleased(self, event): # hacer movimiento cuando levanta el cursor
        if self.controlling is None:
            return
        cell = self.getCell(event)
        self.controlling.pressed=cell
        eventMove.set() # continua la ejecucion de Human.play()

    # obtener coordenadas de celda a partir del cursor
    def getCell(self, event):
        x = event.x // CELLSIZE
        y = event.y // CELLSIZE
        return (3*(y//3)+(x//3),3*(y%3)+(x%3))

    def enableMouse(self,player): # activar raton para jugador
        self.controlling = player
        
    def disableMouse(self): # desactivar raton 
        self.controlling = None

# JUGADORES

class Player():
    def __init__(self,name):
        self.name = name
        self.turn = None

    def play(self):
        raise NotImplementedError("Play method not implemented")

class Human(Player):
    def __init__(self,name,*kargs):
        Player.__init__(self,name)
        self.pressed = None

    def play(self,gui=True):
        mouse.enableMouse(self)
        eventMove.wait()
        if board.move(self.pressed):
            self.pressed = None
            mouse.disableMouse()

class IA(Player):
    def __init__(self,name,depth=4,heuristic=6,info=False):
        Player.__init__(self,name)
        self.heuristic = [self.__h0,self.__h1,self.__h2,self.__h3,self.__h4,self.__h5,self.__h6,self.__h7][heuristic]
        self.hvalue = heuristic
        self.depth = depth
        self.node = None
        self.info = info
        self.nodes_explored = []
        
    def play(self,gui=True):
        self.nodes_explored += [0]
        t0 = time.time()
        value,moves = self.minimax(self.board.clone(),self.depth,1 if self.turn == 1 else -1)
        t1 = (time.time()-t0)*1000
        if gui:
            self.board.move(moves[0])
        else:
            self.board.move2(moves[0])
        if self.info:
            value *= (1 if self.turn == 1 else -1)
            v = math.copysign(1000-abs(value),value)
            value += (-1 if (self.turn == 1) ^ (self.depth&1) else 1)*self.depth
            
            tnodes = "∞ knodes/s" if t1 == 0 else "%.3f knodes/s"%(self.nodes_explored[-1]/t1)
            teval = "#%d"%(v//2) if abs(v)<162 else "%.3f"%value
            tcureval = "%.3f"%self.heuristic(self.board)
            print(self.name,self.hvalue,tnodes,moves,teval,tcureval)
            
    def __evalMove(self,board,move):
        board.move2(move)
        res = [0,-1000,-1000,0][board.win] if board.win else self.heuristic(board)
        board.undomove2()
        return res

    def minimax(self,b,depth,side,alpha=-1000,beta=1000):
        self.nodes_explored[-1] += 1
        if b.win != 0:
            return [0,-1000,-1000,0][b.win] + self.depth-depth,[]
        if depth == 0:
            return side*self.heuristic(b)+self.depth,[]
        depth -= 1
        bestvalue = -1000000
        bestmoves = []
        moves = b.getPossibleMoves() if depth == 0 else sorted(b.getPossibleMoves(), key=lambda k: -side*self.__evalMove(b,k))
        for child in moves:
            b.move2(child)
            value,moves = self.minimax(b,depth,-side,-beta,-alpha)
            value *= -1
            b.undomove2()
            if value > bestvalue:
                bestvalue = value
                bestmoves = [child]+moves
            alpha = max(alpha,bestvalue) # poda alfa-beta
            if alpha >= beta:
                break
        return bestvalue,bestmoves

    # HEURISTICOS

    def __h0(self,board):
        s = [heuristicsLUT[board.index13[i]][2] for i in range(9)]
        res = heuristicsLUT[indexBoard(s)][2]
        return res
    
    def __h1(self,board):
        return sum(heuristicsLUT[board.index13[i]][0] for i in range(9) if board.wBoard[i] != 3)
    
    def __h2(self,board):
        return sum(heuristicsLUT[board.index13[i]][1] for i in range(9) if board.wBoard[i] != 3)

    def __h3(self,board):
        return board.wBoard.count(1)-board.wBoard.count(2)
    
    def __h4(self,board):
        return self.__h1(board)+self.__h3(board)

    def __h5(self,board):
        return fullheuristicsLUT[indexBoard(board.wBoard,base=4)]

    def __h6(self,board):
        return self.__h1(board)+self.__h5(board)

    def __h7(self,board):
        return random.randint(-10,10)

# PRECOMPUTAR HEURISTICOS

def h_weights(board): # asigna distintos pesos dependiendo del numero de casillas completadas en cada fila y los suma todos
    weights = {0:0,1:0.2,2:0.5,3:1}
    return sum(0 if bitor(l)==3 else weights[l.count(1)]-weights[l.count(2)] for line in lines if (l:=list(itemgetter(*line)(board))))

def h_weights_weak(board): # asigna distintos pesos dependiendo del numero de casillas completadas en cada fila sin tener en cuenta si se puede o no hacer linea y los suma todos
    weights = {0:0,1:0.2,2:0.5,3:1}
    return sum(weights[l.count(1)]-weights[l.count(2)] for line in lines if (l:=list(itemgetter(*line)(board))))

def h_count(board): # cuenta la diferencia de casillas
    return sum((b==1)-(b==2) for b in board)

def h_countlines(board): # cuenta el numero de lineas que puede hacer cada jugador
    v = sum((bitor(itemgetter(*line)(board))==1)-(bitor(itemgetter(*line)(board))==2) for line in lines)
    return (v>0) + 2*(v<0)

def indexBoard(board,base=3): # indexar tableros entre 0 y base^9-1
    return reduce(lambda a,b: base*a+b,board)

def checkWin(board):
    if any(all(i == 1 for i in itemgetter(*line)(board)) for line in lines):
        return 1
    if any(all(i == 2 for i in itemgetter(*line)(board)) for line in lines):
        return 2
    if all(bitor(itemgetter(*line)(board))==3 for line in lines): # comprobar si todas las lineas estan empatadas y declarar empate
        return 3
    return 0

# UTILIDADES

def printBoard(board):
    print("\n".join("".join(" ".join("".join(str(board[3*i+j][3*k+l]) for l in range(3)) for j in range(3))+"\n" for k in range(3)) for i in range(3)))
	
def bitor(x):
    return x[0]|x[1]|x[2]
#    return reduce(lambda a,b: a|b,x)

# COMPARACION DE IAS

def compareIA(h1,h2,depth1=6,depth2=6,n=100,output=False): # h1 vs h2
    ia1 = IA("IA %d"%h1,depth1,h1)
    ia2 = IA("IA %d"%h2,depth2,h2)
    results = [0]*n
    moves = [0]*n
    for i in range(n):
        if output and i%(n//10)==0:
            print("Round",i)
        board = Board(ia1,ia2)
        board.start(False,False)
        results[i] = board.win
        moves[i] = len(board.lastMove)
    print("Results:")
    wins = results.count(1)
    draws = results.count(3)
    loses = results.count(2)
    print("IA %d wins/draws/loses: %d/%d/%d"%(h1,wins,draws,loses))
    mean = sum(moves)/n
    print("min/avg/max/mdev moves: %d/%.3f/%d/%.3f"%(min(moves),mean,max(moves),sum(abs(m-mean) for m in moves)/n))
    return results,moves

def compareNodesByHeuristics(depth=4,n=10,output=False): # compara el numero de nodos visitados por cada heuristico a profundidad fija
    nodes = [[0]*41 for i in range(8)]
    for A in range(8):
        iaA = IA("IA %d"%A,depth,A)
        for B in range(8):
            iaB = IA("IA %d"%B,depth,B)
            print(A,"vs",B)
            for i in range(n):
                if output and i%(n//10 + 1)==0:
                    print("Round",i)
                board = Board(iaA,iaB)
                board.start(False,False)
                for j in range(len(iaA.nodes_explored)):
                    nodes[A][j] += iaA.nodes_explored[j]
                for j in range(len(iaB.nodes_explored)):
                    nodes[B][j] += iaB.nodes_explored[j]
                iaA.nodes_explored = []
                iaB.nodes_explored = []
    return nodes

def compareNodesByDepth(h1,h2,n=10,output=False): # compara el numero de nodos visitados por h1 vs h2 a profundidad variable
    nodes = {i:[0]*41 for i in range(1,8)}
    for d in nodes.keys():
        ia1 = IA("IA%d"%h1,d,h1)
        ia2 = IA("IA%d"%h2,d,h2)
        print("Profundidad", d)
        t0 = time.time()
        for i in range(n):
            board = Board(ia1,ia2)
            board.start(False,False)
            for j in range(len(ia1.nodes_explored)):
                nodes[d][j] += ia1.nodes_explored[j]
            for j in range(len(ia2.nodes_explored)):
                nodes[d][j] += ia2.nodes_explored[j]
            ia1.nodes_explored = []
            ia2.nodes_explored = []
        t1 = time.time()
        print("Time elapsed: ",t1-t0)
    return nodes

def play(difficulty=0):
    global window, render, eventMove, mouse, board
    
    ias = [("Fácil",3,0),("Normal",2,4),("Difícil",4,6),("Imposible",7,6)]
    ia = IA(*ias[difficulty],info=True)
    
    board = Board(Human("Jugador 1"),ia)
    mouse = Mouse()
    
    window = tk.Tk()
    window.title("Tres (tres en raya) en raya")
    window.geometry("603x603")
    window.resizable(False,False)
    render = Render(window, bg = 'lightgray')

    eventMove = Event() # evento para esperar a que los jugadores hagan un movimiento
    gameThread = Thread(target=board.start,name="Game Loop")
    gameThread.start()

    window.mainloop()

def eloSystem(elo=None,n=1000,K=32):
    if elo is None:
        elo = [[1500] for i in range(8)]
    for i in range(n):
        order = random.sample(range(8),8)
        for j in range(4):
            ia1 = order.pop()
            ia2 = order.pop()
            board = Board(IA("IA %d"%ia1,4,ia1),IA("IA %d"%ia2,4,ia2))
            board.start(False,False)
            res = {1:1,2:0,3:0.5}[board.win] # puntuacion victoria primer jugador, segundo 1-res
            Qa = pow(10,elo[ia1][-1]/400)
            Qb = pow(10,elo[ia2][-1]/400)
            elo[ia1] += [elo[ia1][-1]+K*(res-Qa/(Qa+Qb))]
            elo[ia2] += [elo[ia2][-1]+K*(Qa/(Qa+Qb)-res)]
        
        print(f"Round %d"%i)
    for i,x in enumerate(elo):
        plt.plot(x,label=f"h%d"%i)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Precomputando tablas")
    t = time.process_time()
    localHeuristics = [h_weights,h_count,h_countlines,h_weights_weak] # lista de heuristicos para los tres en raya pequeños
    heuristicsLUT = [[f(board) for f in localHeuristics] for board in product(range(3),repeat=9)] # tabla de consulta de heuristicos para agilizar el minimax
    fullheuristicsLUT = [h_weights(board) for board in product(range(4),repeat=9)]
    winLUT = [checkWin(board) for board in product(range(3),repeat=9)]
    fullwinLUT = [checkWin(board) for board in product(range(4),repeat=9)]

    print("%.3f s"%(time.process_time()-t))

    difficulty = int(input("Introduce la dificultad (0:fácil, 1:normal, 2:difícil, 3:imposible): "))
    play(difficulty)

    
