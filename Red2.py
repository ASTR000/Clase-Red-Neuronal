import numpy as np
import math
import matplotlib.pyplot as plt
import time                                                


def Act(x, tipo="sigmoid"):
    match tipo:
        case "tanh":
            return np.tanh(x)
        case "sigmoid":
            return 1./(1+np.exp(-x))
        case "ReLU":
            return np.max([0,x])


def dAct(x, tipo="sigmoid"):
    match tipo:
        case "tanh":
            return 1-np.tanh(x)**2
        case "sigmoid":
            return Act(x)*(1-Act(x))
        case "ReLU":
            if x > 0:
                return 1
            else:
                return 0

def Escalar(ndarray):
    maximo = np.max(ndarray)
    return ndarray/float(maximo), maximo



class Red:
    # La red requiere dos listas, los input (x), y los ouputs deseados (y).
    # ambas listas deben ser tal que cada fila es un dato distinto. Se pide asi porque es mas humanamente comprensible
    # NPC representa Nodos Por Capa, incluyendo la capa 0 de entrada y la ultima capa de salida, que corresponde a la
    # prediccion.   NPC = [3,5,5,2] corresponde a una red cuya entrada es un vector de dimension 3 y cuya salida es un vector de
    # dimension 2. Entremedio se encuentran 2 "hidden layers" con 5 nodos o de dimension 5.
    # LA ENTRADA Y SALIDA DE LOS NODOS POR CAPA DEBE COINCIDIR CON LA DIMENSION DE LOS DATOS DE INPUT Y SALIIDA DESEADA
    def __init__(self, x, y, NPC, N_batch=1):  
        self.INPUT = x
        self.SALIDA_IDEAL = y

        self.NPC = NPC
        self.Ncapas = len(NPC)

        

        self.W = [ np.random.uniform( -1,1, (self.NPC[i+1], self.NPC[i])) for i in range(self.Ncapas-1)]      
        self.B = [ np.random.uniform( -1,1, (self.NPC[i+1], 1)          ) for i in range(self.Ncapas-1) ]

        self.dW = [ np.zeros((self.NPC[i+1], self.NPC[i]) ) for i in range(self.Ncapas-1)]
        self.dB = [ np.zeros((self.NPC[i+1], 1)           ) for i in range(self.Ncapas-1) ]

        random_state = np.random.get_state()
        self.X, self.normx = Escalar(x)
        np.random.shuffle(self.X)
        np.random.set_state(random_state)
        self.Y, self.normy = Escalar(y)
        np.random.shuffle(self.Y)

        self.mini_batches = np.array_split(self.X, N_batch)
        self.mini_pred = np.array_split(self.Y, N_batch)
        self.N_batch = N_batch
    
        self.N_datos = self.X.shape[0] ## CADA FILA ES UN DATO

        self.E = 0

        #print( "============ DEBUG ==============")
        #print("Informacion de inicializacion:")
        #print("Input y output: ", self.INPUT.shape, "  ", self.TRAIN_SET.shape )
        #print(self.X)
        #print("=================")
        #print(self.Y)
        #print("=================")
        #print("Informacion de batches ","Cantidad de batches: ", self.N_batch, " size :", self.mini_batches.shape)
        #print(self.mini_batches)

    
    def Adelante(self, batch):
        # el batch es un conjunto de datos con forma (d, n),
        # -d es la dimension de cada dato y n es la cantidad de datos en el batch
        
        for i in range(self.Ncapas):
            if i == 0:
                self.Z[i] = batch
                self.A[i] = batch
            else:
                #print( np.dot( self.W[i-1],self.A[i-1]).shape, "  ", self.B[i-1].shape)
                self.Z[i] = np.dot( self.W[i-1],self.A[i-1]) + self.B[i-1]
                self.A[i] = Act(self.Z[i])
        #print(self.A[-1])

    def Loss(self, pred):
        self.E = 0
        tmp = (self.A[-1]-pred)**2
        
        tmp = np.sum(tmp, axis=1, keepdims=True)
        tmp = tmp/float(pred.shape[1])

        #tmp = np.sqrt((tmp.T).dot(tmp))
 
        self.E = tmp[0][0]

    def Atras(self, pred, step):
        delta = []
        for i in reversed(range(len(self.W))):
            if i == len(self.W)-1:
                delta = 2.*(self.A[i+1]-pred)/pred.shape[1]
                delta = delta*dAct(self.Z[i+1])

                self.dB[i] = np.sum(delta,1,keepdims=True)/float(pred.shape[1])
                self.dW[i] = delta.dot(self.A[i].T)
                
            else:
                delta = (self.W[i+1].T).dot(delta)*dAct(self.Z[i+1])

                self.dB[i] = np.sum(delta,1,keepdims=True)/float(pred.shape[1])
                self.dW[i] = delta.dot(self.A[i].T)
                

            #print("delta: ", delta.shape,"  B: ", self.B[i].shape,"  dB: ", self.dB[i].shape)
            
            self.W[i] -= self.dW[i]*step
            self.B[i] -= self.dB[i]*step
            
        
        #print("===="*10)

    def Evaluar(self):
        batch = self.X.T
        batch_size = batch.shape[1]
        self.Z = [ np.empty( (i, batch_size) ) for i in self.NPC ]
        self.A = [ np.empty( (i, batch_size) ) for i in self.NPC ]

        self.Adelante(batch)
        

        # print( "====== PREDICCION ======")
        # print(self.A[-1])
        # print("======= IDEAL =========")
        # print(self.Y.T)
        ##################################################################################

    def EvaluarConInput(self, batch):
        batch = batch.T
        batch_size = batch.shape[1]
        self.Z = [ np.zeros( (i, batch_size) ) for i in self.NPC ]
        self.A = [ np.zeros( (i, batch_size) ) for i in self.NPC ]

        self.Adelante(batch)
        #print(self.A[-1])

        return self.A[-1]

    def Entrenar(self, iteraciones, paso, tolerancia, print_cada = 100): 
      
        for iteracion in range(iteraciones):
            self.Z = []
            self.A = []
            
            for batch, pred in zip(self.mini_batches, self.mini_pred):
                batch = batch.T
                pred = pred.T
                batch_size = batch.shape[1]
                self.Z = [ np.empty( (i, batch_size) ) for i in self.NPC ]
                self.A = [ np.empty( (i, batch_size) ) for i in self.NPC ]

                self.Adelante(batch)
                self.Loss(pred)

                if abs(self.E) < abs(tolerancia):
                
                    print("TOLERANCIA ALCANZADA")
                    print("Iteraciones: ", iteracion, ".  Error promedio : ", self.E)
                    print( "con tolerancia: ", abs(tolerancia))
                    return
                
                self.Atras(pred, paso)

            #self.EvaluarConInput(self.X)
            #self.Loss(self.Y)
            if iteracion % print_cada == 0:
                pass
                print("Iteraciones: ", iteracion, ".  Error promedio : ", self.E)
                      

