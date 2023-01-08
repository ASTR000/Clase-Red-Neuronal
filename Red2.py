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

        self.W = [ np.random.uniform( -1,1, (self.NPC[i+1], self.NPC[i]) ) for i in range(self.Ncapas-1)]      
        self.B = [ np.random.uniform( -1,1, (self.NPC[i+1], 1)           ) for i in range(self.Ncapas-1) ]

        self.dW = [ np.zeros((self.NPC[i+1], self.NPC[i]) )  for i in range(self.Ncapas-1)]
        self.dB = [ np.zeros((self.NPC[i+1], 1)           ) for i in range(self.Ncapas-1) ]

        self.E = 1e10
    
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
        n_w = len(self.W)


        for i in reversed( range( len(self.W) ) ):
            if i == len(self.W)-1:
                delta = 2.*(self.A[i+1]-ideal)/float(cantidad_datos)
                delta = delta*dAct(self.Z[i+1])

                self.dB[i] = np.sum(delta,1,keepdims=True)/float(cantidad_datos)
                self.dW[i] = delta.dot(self.A[i].T)
                
            else:
                delta = (self.W[i+1].T).dot(delta)*dAct(self.Z[i+1])

                self.dB[i] = np.sum(delta,1,keepdims=True)/float(cantidad_datos)
                self.dW[i] = delta.dot(self.A[i].T)
            
            self.W[i] -= self.dW[i]*step
            self.B[i] -= self.dB[i]*step
            
    def EvaluarBatch(self, batch):
        batch = batch.T
        batch_size = batch.shape[1]
        self.Z = [ np.zeros( (i, batch_size) ) for i in self.NPC ]
        self.A = [ np.zeros( (i, batch_size) ) for i in self.NPC ]

        self.Adelante(batch)


        return self.A[-1]

    def Entrenar(self, entrada, salida, iteraciones, paso, tolerancia=1e-8, N_batch=1, print_cada = 1): 
        ## Esta es la funcion principal para entrenar la red. La entrada y salida son arrays de datos
        ## donde cada fila es un dato nuevo. (al operar estos valores se trasponen).

        ## Guardando los datos originales.
        self.INPUT = entrada
        self.SALIDA = salida

        ## Mezclamos los datos para que la red generalize mejor al
        ## utilizar datos y reescalamos para que el máximo valore que tome
        ## alguno de los nodos de entrada sea mayor que 1 (o menor a -1)
        random_state = np.random.get_state() ## asi me aseguro que se desordenen correlacionadamente
        self.X, self.normx = Escalar(entrada)
        np.random.shuffle(self.X)
        np.random.set_state(random_state)
        self.Y, self.normy = Escalar(salida)
        np.random.shuffle(self.Y)

        ## Creando los batches desde los datos mezclados
        self.mini_batches = np.array_split(self.X, N_batch)
        self.mini_ideal = np.array_split(self.Y, N_batch)
        self.N_batch = N_batch
        self.N_datos = self.X.shape[0] ## En este punto cada fila es un dato

        for iteracion in range(iteraciones):
            self.Z = []
            self.A = []
            
            # Cada iteracion pasa por todos los batches. Con cada uno se aplica
            # la propagación hacia atras. (Descenso de Gradiente Estocastico)
            for batch, ideal in zip(self.mini_batches, self.mini_ideal):
                batch = batch.T     # ahora cada columna
                ideal = ideal.T     # es un dato
                batch_size = batch.shape[1]
                self.Z = [ np.empty( (i, batch_size) ).astype(np.double)  for i in self.NPC ]
                self.A = [ np.empty( (i, batch_size) ).astype(np.double)  for i in self.NPC ]

                self.Adelante(batch)
                
                self.Loss(ideal)

                self.Atras(ideal, paso)

                ## Mensaje en caso de que la tolerancia sea alcanzada.
                if abs(self.E) < abs(tolerancia):
                    print("TOLERANCIA ALCANZADA")
                    print("Iteraciones: ", iteracion, ".  Error promedio : ", self.E)
                    print( "con tolerancia: ", tolerancia)
                    return
                
            ## Esta es la forma de visualizar el progreso dentro de la terminal
            if iteracion % print_cada == 0:
                pass
                print("Iteraciones: ", iteracion, ".  Error promedio : ", self.E)
                      
#####################################################################################
################################## Funciones miscelaneas ############################
#####################################################################################


## Funcion de activación. Actualmente no funciona ReLU :(
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


# Funcion que escala según el valor mas grande que se encuentre
# en alguna de los componentes de los datos. De esta forma todos los nodos
# de entrada, para cualquier dato, solo entrarán valores entre -1 y 1
def Escalar(ndarray):
    maximo = np.max(ndarray)
    return ndarray/maximo, maximo
