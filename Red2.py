import numpy as np
import matplotlib.pyplot as plt                                           


class Red:
    # La red requiere dos listas, los input (x), y los ouputs deseados (y).
    # ambas listas deben ser tal que cada fila es un dato distinto. Se pide asi porque es mas humanamente comprensible
    # NPC representa NODOS POR CAPA, incluyendo la capa 0 de entrada y la ultima capa de salida, que corresponde a la
    # prediccion.   NPC = [3,5,5,2] corresponde a una red cuya entrada es un vector de dimension 3 y cuya salida es un vector de
    # dimension 2. Entremedio se encuentran 2 "hidden layers" con 5 nodos o de dimension 5.
    # LA ENTRADA Y SALIDA DE LOS NODOS POR CAPA DEBE COINCIDIR CON LA DIMENSION DE LOS DATOS DE INPUT Y SALIIDA DESEADA
    def __init__(self, NPC):  

        self.NPC = NPC
        self.Ncapas = len(NPC)   

    
        self.W = [ np.random.uniform( -1,1, (self.NPC[i+1], self.NPC[i])).astype(np.double) for i in range(self.Ncapas-1)]      
        self.B = [ np.random.uniform( -1,1, (self.NPC[i+1], 1)          ).astype(np.double)  for i in range(self.Ncapas-1) ]

        self.dW = [ np.zeros((self.NPC[i+1], self.NPC[i]) ).astype(np.double)  for i in range(self.Ncapas-1)]
        self.dB = [ np.zeros((self.NPC[i+1], 1)           ).astype(np.double)  for i in range(self.Ncapas-1) ]

        self.E = 0
    
    def Adelante(self, batch):
        # el batch es un conjunto de datos con forma (d, n),
        # d es la dimension de cada dato y n es la cantidad de datos en el batch
        self.Z[0] = batch
        self.A[0] = batch

        for i in range(1,self.Ncapas):
            self.Z[i] = np.dot( self.W[i-1],self.A[i-1]) + self.B[i-1]
            self.A[i] = Act(self.Z[i])

    
    def Loss(self, ideal):
        self.E = 0
        cantidad_datos = ideal.shape[1]
        tmp = (self.A[-1]-ideal)**2 
        tmp = np.sum(tmp, axis=1, keepdims=True)
        tmp = tmp/float(cantidad_datos)

        self.E = tmp[0][0]

    def Atras(self, ideal, step):
        delta = []
        cantidad_datos = ideal.shape[1]
        
        for i in reversed(range(len(self.W))):
            if i == len(self.W)-1:
                delta = 2.*(self.A[i+1]-ideal)/float(cantidad_datos)
                delta = delta*dAct(self.Z[i+1])

                self.dB[i] = np.sum(delta,1,keepdims=True)/float(cantidad_datos)
                self.dW[i] = delta.dot(self.A[i].T)
                
            else:
                delta = (self.W[i+1].T).dot(delta)*dAct(self.Z[i+1])

                self.dB[i] = np.sum(delta,1,keepdims=True)/float(cantidad_datos)
                self.dW[i] = delta.dot(self.A[i].T)
                

            #print("delta: ", delta.shape,"  B: ", self.B[i].shape,"  dB: ", self.dB[i].shape)
            
            self.W[i] -= self.dW[i]*step
            self.B[i] -= self.dB[i]*step
            
        
        #print("===="*10)


    def EvaluarBatch(self, batch):
        batch = batch.T
        batch_size = batch.shape[1]
        self.Z = [ np.zeros( (i, batch_size) ) for i in self.NPC ]
        self.A = [ np.zeros( (i, batch_size) ) for i in self.NPC ]

        self.Adelante(batch)

        return self.A[-1]

    def Entrenar(self, entrada, salida, iteraciones, paso, tolerancia=1e-8, N_batch=1, print_cada = 1): 
        
        self.INPUT = entrada
        self.SALIDA_IDEAL = salida

        random_state = np.random.get_state()
        self.X, self.normx = Escalar(entrada)
        np.random.shuffle(self.X)
        np.random.set_state(random_state)
        self.Y, self.normy = Escalar(salida)
        np.random.shuffle(self.Y)

        self.mini_batches = np.array_split(self.X, N_batch)
        self.mini_pred = np.array_split(self.Y, N_batch)
        self.N_batch = N_batch
    
        self.N_datos = self.X.shape[0] ## CADA FILA ES UN DATO

        for iteracion in range(iteraciones):
            self.Z = []
            self.A = []
            
            for batch, pred in zip(self.mini_batches, self.mini_pred):
                batch = batch.T
                pred = pred.T
                batch_size = batch.shape[1]
                self.Z = [ np.empty( (i, batch_size) ).astype(np.double)  for i in self.NPC ]
                self.A = [ np.empty( (i, batch_size) ).astype(np.double)  for i in self.NPC ]

                self.Adelante(batch)
                self.Loss(pred)

                if abs(self.E) < abs(tolerancia):
                
                    print("TOLERANCIA ALCANZADA")
                    print("Iteraciones: ", iteracion, ".  Error promedio : ", self.E)
                    print( "con tolerancia: ", tolerancia)
                    return
                
                self.Atras(pred, paso)

            ## Esta es la forma de visualizar el progreso dentro de la terminal
            if iteracion % print_cada == 0:
                pass
                print("Iteraciones: ", iteracion, ".  Error promedio : ", self.E)
                      
#####################################################################################
################################## Funciones miscelaneas ############################
#####################################################################################

tipo_default = "sigmoid"
def Act(x, tipo=tipo_default):
    match tipo:
        case "tanh":
            return np.tanh(x)
        case "sigmoid":
            return 1./(1+np.exp(-x))


def dAct(x, tipo=tipo_default):
    match tipo:
        case "tanh":
            return 1-np.tanh(x)**2
        case "sigmoid":
            return Act(x)*(1-Act(x))


def Escalar(ndarray):
    maximo = np.max(ndarray)
    return ndarray/float(maximo), maximo
