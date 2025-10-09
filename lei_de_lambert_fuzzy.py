import numpy as np
import matplotlib.pyplot as plt


###############################################################################
##########################   Tratamento de Dados   ############################
###############################################################################

def _trata_triplicata(dados):
    """
    Calcula média e desvio padrão para medições feitas em triplicatas.

    Args:
        dados (list): lista contendo todos os dados medidos, sendo cada tripla de dados correspondente a uma condição de medição. Ex: cada tripla representa os dados coletados para uma concentração.

    Returns:
        media (list): lista contendo a média de cada triplicata.
        desv_pad (list): lista contendo o desvio padrão de cada triplicata.
    """
    
    media = []
    desv_pad = []

    for i in range(0, len(dados), 3):
        med = (dados[i] + dados[i+1] + dados[i+2] ) / 3

        trip = [dados[i], dados[i+1], dados[i+2]]
        dpad = np.std(trip)

        media.append(med)
        desv_pad.append(dpad)

    return media, desv_pad


def _tira_branco(media, desv_pad):
    """
    Tira o branco de um conjunto de dados com base na média e no desvio padrão das triplicatas.
    
    Args:
        media (list): lista contendo a média de cada triplicata.
        desv_padrao (list): lista contendo o desvio padrão de cada triplicata.

    Returns:
        med_final (list): lista contendo a média de cada triplicata após retirar o branco.
        desv_pad_final (list): lista contendo o desvio padrão de cada triplicata após retirar o branco.
        
    """
    
    branco = media[0]
    med_final = []
    desv_pad_final = desv_pad[1:]
    
    for m in media[1:]:
        med_branco_local = m - branco

        med_final.append(med_branco_local)

    return med_final, desv_pad_final


def trata_dados(dados):
    """
    Calcula média e desvio padrão das triplicatas, e subtrai o valor do branco.
    
    Args:
        dados (list): lista contendo todos os dados medidos, sendo cada tripla de dados correspondente a uma condição de medição. Ex: cada tripla representa os dados coletados para uma concentração.
        
    Returns:
        med_final (list): lista contendo a média de cada triplicata após retirar o branco.
        desv_pad_final (list): lista contendo o desvio padrão de cada triplicata após retirar o branco.
        
    """
        
    media, desv_pad = _trata_triplicata(dados)

    med_final, desv_pad_final = _tira_branco(media, desv_pad)
    
    return med_final, desv_pad_final


#######################   Tratamento de Dados Fuzzy   #########################

def _trata_triplicata_fuzzy(dados):
    """
    Calcula média e desvio padrão para medições feitas em triplicatas, tranformando dados reais em um número triangular fuzzy.

    Args:
        dados (list): lista contendo todos os dados medidos, sendo cada tripla de dados correspondente a uma condição de medição. Ex: cada tripla representa os dados coletados para uma concentração.

    Returns:
        dados_fuzzy (list of tuple): lista contendo todos os dados medidos, no formato de números triângulares fuzzy.
    """
    
    media, desv_pad = _trata_triplicata(dados)
    
    dados_fuzzy = []

    for m, e in zip(media, desv_pad):
        dados_fuzzy.append([(m-e,m, m+e)])
    
    return dados_fuzzy
    
    
def _tira_branco_fuzzy(dados_fuzzy):
    """
    Tira o branco de um conjunto de dados fuzzy.
    
    Args:
        dados_fuzzy (list of tuple): lista contendo todos os dados medidos, no formato de números triângulares fuzzy.

    Returns:
        dfuzzy_sem_branco (list of tuple): lista contendo todos os dados medidos após retirar o branco, no formato de números triângulares fuzzy.
        
    """
    
    branco = dados_fuzzy[0][0]
    a, b, c = branco
    #oposto_branco = (-c, -b, -a)
    oposto_branco = oposto_fuzzy(branco)
    
    dfuzzy_sem_branco = []
    
    for tupla in dados_fuzzy[1:]:
        for i in tupla:
            sub_branco = soma_iterativa_0(i, oposto_branco)
            dfuzzy_sem_branco.append([sub_branco])
        
    return dfuzzy_sem_branco


def trata_dados_fuzzy(dados):
    """
    Calcula média e desvio padrão das triplicatas fuzzy, e subtrai o valor do branco.

    Args:
        dados (list): lista contendo todos os dados medidos, sendo cada tripla de dados correspondente a uma condição de medição. Ex: cada tripla representa os dados coletados para uma concentração.

    Returns:
        dados_fuzzy (list of tuple): lista contendo todos os dados medidos após retirar o branco, no formato de números triângulares fuzzy.
    """
    
    parcial_fuzzy = _trata_triplicata_fuzzy(dados)

    dados_fuzzy = _tira_branco_fuzzy(parcial_fuzzy)
    
    return dados_fuzzy


###############################################################################
###############   Matriz coluna de resultado de uma função   ##################
###############################################################################

def _aplica_funcao(x_data, f_expr):
    """
    Avalia a função f(x), definida por uma string, nos pontos de x_data e retorna os resultados como vetor coluna.

    Args:
        x_data (array-like): vetor coluna (n x 1) contendo os valores de entrada x₀ nos quais a função será avaliada.
        f_expr (str): Expressão representando a função f(x) interpretada via eval, por exemplo: "x**2 + 3*x - 1" ou "math.cos".
        
    Returns:
        results_f (np.ndarray): vetor coluna (n x 1) contendo os valores f(x₀) para cada x₀ em x_data.
    """
    
    # Criar função f(x) a partir da expressão f_expr
    f = lambda x: eval(f_expr)
    
    # Lista para armazenar os resultados de f(x0), x0 em x_data
    results_f = []
    
    # Itera sobre os valores em x_data e calcula f(x)
    for x in x_data:
        results_f.append(f(x[0]))
        
    # Transforma results_f em uma matriz coluna com cada valor f(x0), x0 em x_data
    results_f = np.array(results_f)
    
    results_f = results_f.reshape(-1,1)
    
    return results_f


###############################################################################
########################   Decomposição Lower-Upper   #########################
###############################################################################

def ordena_dados_crescente(x_data, y_data, x_real=None):
    """
    Ordena três conjuntos de dados relacionados entre si pelo índice, com base na orde crescente do primeiro deles. 
   
    Args:
        x_data (array-like): matriz coluna de dados reais de entrada.
        y_data (array-like): matriz coluna de dados fuzzy de saída, na forma de tuplas (a, b, c).
        x_real (array-like, optional): valores reais originais de x, se diferentes dos usados em x_data (padrão: None).
      
    Returns:
        x_ord (np.ndarray): matriz coluna de dados reais de entrada ordenados em ordem crescente.
        y_ord (np.ndarray of tuple): matriz coluna de dados fuzzy de saída, na forma de tuplas (a, b, c) ordenados em ordem crescente de x.
        x_real_ord (np.ndarray): valores reais originais de x ordenados em ordem crescente de x (padrão: None, x_real_ord = x_ord).       
        
    """
    
    indices_ordenados = np.argsort(x_data[:, 0])
    
    x_ord = x_data[indices_ordenados]
    y_ord = y_data[indices_ordenados]
    
    if x_real is None:
        x_real_ord = x_ord.copy()
    
    else:
        x_real_ord = x_real[indices_ordenados]
        
    return x_ord, y_ord, x_real_ord


def _ordena_array(array, resposta):
    """
    Realiza o pivoteamento parcial de um sistema linear, reordenando as linhas da matriz de coeficientes para maximizar os valores absolutos na diagonal principal. 
   
    Args:
      array (array-like): matriz quadrada (n x n) contendo os coeficientes do sistema linear.
      resposta (array-like): vetor coluna (n x 1) contendo os termos independentes do sistema linear.
      
    Returns:
        a (array-like): matriz de coeficientes pivoteada.
        b (array-like): vetor de termos independentes pivoteado.
        
    """
    
    a = array.copy()
    b = resposta.copy()
    n = len(a)
    for j in range(n): #para cada coluna
        for i in range (j+1, n): #para cada linha
            if abs(a[j][j]) == 0 and abs(a[i][j]) != 0:
                a[j], a[i] = a[i], a[j]
                b[j], b[i] = b[i], b[j]
            if abs(a[j][j]) < abs(a[i][j]): 
                a[j], a[i] = a[i], a[j]
                b[j], b[i] = b[i], b[j]
    return a,b


def _decomposicaoLU(A):
    """
    Realiza a decomposição de uma matriz de coeficientes de um sistema linear em uma matriz triangular inferior (L) e uma matriz triangular superior (U), tal que A = L @ U.
   
    Args:
      A (array-like): matriz quadrada (n x n) contendo os coeficientes de um sistema linear.
      
    Returns:
        (L,U): tupla contendo as matrizes obtidas na decomposição LU:
            L (np.ndarray): matriz triangular inferior, contendo os multiplicadores utilizados no escalonamento da matriz A.
            U (np.ndarray): matriz triangular superior resultante do escalonamento da matriz A.
            
    Raises:
        ValueError: Se a matriz A contiver pivôs nulos na diagonal, impedindo a decomposição LU.
        
    """
    
    for i in range(len(A)):
        for j in range(len(A[i])):
            A[i][j] = float(A[i][j])
    U = np.copy(A)
    nlU, ncU = np.shape(U) # formato da matriz
    L = np.eye(nlU) # define L como uma matriz identidade
    for j in range (nlU): # para cada coluna -> índice da coluna
        pivo = U[j][j] # elementos da diagonal principal
        if pivo == 0:
            raise ValueError("A matriz A não é adequada para decomposição LU (pivô nulo identificado)")
        for i in range (j+1, nlU): # da coluna/linha seguinte a do pivo até a ultima linha
            multiplicador  = U[i][j] / pivo
            L[i][j] = multiplicador # redefinindo a matriz L
            U[i][:] = U[i][:] - multiplicador*U[j][:] # matruiz U : L2 (linha 2) = L2 - a21/a11 (multiplicador) * L1 (linha 1)
                # 0: -> a partir da primeira coluna
    return (L, U)


def _sistema_triangular_inferior(L, b):
    """
    Resolve um sistema linear do tipo Ly = b, onde L é uma matriz triangular inferior, utilizando substituição direta.

    Args:
        L (np.ndarray): matriz quadrada (n x n) triangular inferior contendo os coeficientes do sistema linear.
        b (array-like): vetor coluna (n x 1) contendo os termos independentes do sistema linear.

    Returns:
        y (np.ndarray): vetor solução (n x 1) do sistema linear.
        
    """
    
    nl, nc = np.shape(L)
    y = np.zeros((nl, 1))  # matriz x
    y[0][0] = b[0][0] / L[0][0]  # primeiro termo da matriz triangular inferior
    for i in range(1, nl):
        y[i][0] = (b[i][0] - np.dot(L[i, :i], y[:i, 0])) / L[i, i]  # y = (b - L*y) / L[i,i]
    return y


def _sistema_triangular_superior(U, y):
    """
    Resolve um sistema linear do tipo Ux = y, onde U é uma matriz triangular superior, utilizando substituição retroativa.

    Args:
        U (np.ndarray): matriz quadrada (n x n) triangular superior contendo os coeficientes do sistema linear.
        y (np.ndarray): vetor coluna (n x 1) contendo os termos independentes do sistema linear.

    Returns:
        x (np.ndarray): vetor solução (n x 1) do sistema linear.
        
    """
    
    nl, nc = np.shape(U)
    x = np.zeros((nl,1)) #matriz solução x
    for i in range(nl-1, -1, -1):
        if i == nl - 1:
            x[i][0] = y[i][0] / U[i][i]  # caso base para a última linha
        else:
            x[i][0] = (y[i][0] - np.dot(U[i, i+1:nl], x[i+1:nl, 0])) / U[i, i]
    return x


def metodoLU(A, b):
    """
    Resolve um sistema linear da forma Ax = b utilizando decomposição LU. A matriz A é decomposta em uma matriz triangular inferior (L) e uma superior (U), tal que A = L @ U. Em seguida, o sistema é resolvido em duas etapas: 1. Ly = b  (substituição direta); 2. Ux = y (substituição retroativa). 
   
    Args:
      A (array-like): matriz quadrada (n x n) contendo os coeficientes do sistema linear.
      b (array-like): vetor coluna (n x 1) contendo os termos independentes do sistema linear.
      
    Returns:
      x (np.ndarray): vetor solução (n x 1) do sistema linear.
        
    """
    
    L,U = _decomposicaoLU(A)
    y = _sistema_triangular_inferior(L, b)
    x = _sistema_triangular_superior(U,y)
    
    return x


def metodoLU_ord(arr, b_arr):
    """
    Resolve um sistema linear da forma Ax = b, pivoteando o sistema e utilizando decomposição LU. A matriz A é decomposta em uma matriz triangular inferior (L) e uma superior (U), tal que A = L @ U. Em seguida, o sistema é resolvido em duas etapas: 1. Ly = b  (substituição direta); 2. Ux = y (substituição retroativa). 
   
    Args:
        arr (array-like): matriz quadrada (n x n) contendo os coeficientes do sistema linear.
        b_arr (array-like): vetor coluna (n x 1) contendo os termos independentes do sistema linear.
      
    Returns:
        x (np.ndarray): vetor solução (n x 1) do sistema linear.
        
    """
        
    A, b = _ordena_array(arr, b_arr)
    L,U = _decomposicaoLU(A)
    y = _sistema_triangular_inferior(L, b)
    x = _sistema_triangular_superior(U,y)
    
    return x


###############################################################################
###########################   Ajuste Tradicional  #############################
###############################################################################

def quadrados_minimos(x_data, y_data, g_lista):
    """
    Resolve um problema de regressão pelo método dos quadrados mínimos com função de base g.
    
    Args:
        x_data (array-like): matriz coluna de dados reais de entrada.
        y_data (array-like): matriz coluna de dados reais de entrada.
        g_lista (list of sting): lista com expressões, em forma de string, das funções de base a serem utilizadas na regressão. Ex: para um ajuste linear, g_list = ["1", "x"].
        
    Returns:
        coefs (np.ndarray): vetor contendo os coeficientes da função de ajuste, com índices correspondentes às funções em g_list. 
    """
    
    n = len(x_data) # n é definido como a quantidade de dados de entrada
    g = len(g_lista) # g é definido com o número de funões de base de entrada
    
    x_extended = np.linspace(x_data[0], x_data[n-1], num=n*10) # x_extended é definido como uma lista com 10 vezes o número de dados n, no intervalo de x_data, a fim de melhorar a posterior plotagem da curva aproximada
    
    g_arr = [] # g_arr é definido como uma lista vazia
    
    for i in range(g): # para cada função de base gi(x) em g_lista (1<=i<=n)
        g_arr.append(_aplica_funcao(x_data,g_lista[i])) # adiciona-se, em g_arr, uma matriz coluna com os valores de g(x0), x0 em x_data
    
    A = np.zeros((g,g)) # A é definido como uma matriz quadrada na ordem g (número de funções de base)
    b = np.zeros((g,1)) # b é definido com uma matriz coluna na ordem g (número de funções de base)

    for i in range(g): # para cada função de base gi(x) em g_lista (1<=i<=n)
        b[i][0] = np.dot(g_arr[i].T, y_data) # cada linha i da matriz b é redefinida como o produto escalar da matriz g_arr[i], correspondente aos resultados de gi(x0) sendo x0 em x_data, pela matriz com dados de y
        for j in range(g): # para cada função de base gj(x) em g_lista (1<=j<=n)
            A[i][j] = np.dot((g_arr[i].T), g_arr[j])  # cada elemento Aij da matriz A é redefinida como o produto escalar da matriz g_arr[i], correspondente aos resultados de gi(x0) sendo x0 em x_data, pela matriz g_arr[j], correspondente aos resultados de gj(x0) sendo x0 em x_data
            
    coefs = metodoLU(A,b) # resolve-se o sistema linear Ax = b pelo , método LU
    
    return coefs


def fita_lambert(concentracoes, absorbancias):
    """
    Resolve um problema de regressão linear pelo método dos quadrados mínimos para a lei de Lambert-Beer.
    
    Args:
        concentracoes (list): lista com valores de concentração (eixo x) da curva a ser fitada.
        absorbancias (list): lista com valores de absorbância (eixo y) da curva a ser fitada.
        
    Returns:
        coefs (list): vetor contendo os coeficientes da função de ajuste, com índices correspondentes às funções em g_list. 
    """
    conc = np.array(concentracoes).reshape(-1,1)
    absorb = np.array(absorbancias).reshape(-1,1)
    
    # Regressão linear
    coefs_matrix = quadrados_minimos(conc, absorb, ["1","x"])
    coefs = coefs_matrix.flatten().tolist()
    
    return coefs


def prev_lambert (coefs, abs_prev):
    """
    Prevê concentrações para absorbancias desejadas seguindo a lei de Lambert-Beer.
    
    Args:
        coefs (list): vetor contendo os coeficientes da função linear de ajuste. 
        abs_prev (list, optional): absorbâncias para as quais se deseja prever a concetração.
        
    Returns:
        conc_prev (list): concentrações previstas para cada absorbância.
    """
    beta_0, beta_1 = coefs
    
    conc_prev =  (np.array(abs_prev) - beta_0)/beta_1
    conc_prev = conc_prev.tolist()
    
    return conc_prev


def plota_lambert(concentracoes, absorbancias, desv_pad, coefs, abs_prev = None, titulo = "Curva de Calibração", xlabel = "Concentração", ylabel = "Absorbância"):
    """
    Plota a solução de um problema de regressão linear pelo método dos quadrados mínimos para a lei de Lambert-Beer, com barra de erro e intervalo de confiança.
    
    Args:
        concentracoes (list): lista com valores de concentração (eixo x) da curva a ser fitada.
        absorbancias (list): lista com valores de absorbância (eixo y) da curva a ser fitada.
        desv_padrao (list): lista contendo o desvio padrão de cada triplicata.
        coefs (list): vetor contendo os coeficientes da função linear de ajuste. 
        abs_prev (list, optional): lista contendo as  absorbâncias para as quais se deseja prever a concetração.
        titulo (string, optional): título do gráfico e da figura salva.
        xlabel (string, optional): legenda do eixo x, preferencialmente contendo a unidade de medida.
        ylabel (string, optional): legenda do eixo y, preferencialmente contendo a unidade de medida.
    """
    
    n=len(concentracoes)
    
    beta_0, beta_1 = coefs
    
    #Linha de regressão
    if abs_prev is None:
        x_vals = np.array([min(concentracoes), max(concentracoes)])
    else:
        conc_prev = prev_lambert(coefs, abs_prev)
        conc = np.concatenate((concentracoes, conc_prev))
        x_vals = np.array([min(conc), max(conc)])
    y_vals = beta_0 + beta_1 * x_vals
    
    # Banda de confiança
    residuos = np.array(absorbancias) - (beta_0 + beta_1 * np.array(concentracoes))
    residuo_std = np.std(residuos)
    y_upper = y_vals + 1.96 * residuo_std / n**(1/2)
    y_lower = y_vals - 1.96 * residuo_std / n**(1/2)

    # Gráfico
    plt.figure(dpi = 200)

    # Pontos com barra de erro
    plt.errorbar(concentracoes, absorbancias, yerr=desv_pad, fmt='.', ms = 5, color='#666666',
                 ecolor='#666666', capsize=4, label='Dados', zorder=1)

    # Linha de regressão
    #plt.plot(x_vals, y_vals, color='#111111', label=f'Regressão Linear\n$R^2 = {r2:.4f}$', linewidth=0.5)
    plt.plot(x_vals, y_vals, color='#111111', label=f'Regressão Linear', linewidth=0.5,zorder=2)

    # Banda de confiança
    plt.fill_between(x_vals, y_lower, y_upper, color='deeppink', alpha=0.3, label='IC 95%', zorder=3)
    
    if abs_prev is not None:
        plt.scatter(conc_prev, abs_prev, s = 5, color="deeppink", label='Previsão', zorder=4)

    # Rótulos e título
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0), borderaxespad=0.5)
    plt.grid(True)
    plt.savefig(titulo,bbox_inches='tight')
    plt.show()
    

def lei_lambert(concentracoes, absorbancias, desv_pad, abs_prev = None, titulo = "Curva de Calibração", xlabel = "Concentração", ylabel = "Absorbância"):
    """
    Resolve um problema de regressão linear pelo método dos quadrados mínimos para a lei de Lambert-Beer, e plota a solução obtida com barra de erro e intervalo de confiança.
    
    Args:
        concentracoes (list): lista com valores de concentração (eixo x) da curva a ser fitada.
        absorbancias (list): lista com valores de absorbância (eixo y) da curva a ser fitada.
        desv_padrao (list): lista contendo o desvio padrão de cada triplicata.
        coefs (list): vetor contendo os coeficientes da função de ajuste, com índices correspondentes às funções em g_list. 
        abs_prev (list, optional): lista contendo as  absorbâncias para as quais se deseja prever a concetração. 
        
    Returns:
        coefs (list): vetor contendo os coeficientes da função linear de ajuste. 
        conc_prev (list): concentrações previstas para cada absorbância.
        titulo (string, optional): título do gráfico e da figura salva.
        xlabel (string, optional): legenda do eixo x, preferencialmente contendo a unidade de medida.
        ylabel (string, optional): legenda do eixo y, preferencialmente contendo a unidade de medida.
    """
    
    coefs = fita_lambert(concentracoes, absorbancias)
    plota_lambert(concentracoes, absorbancias, desv_pad, coefs, abs_prev, titulo, xlabel, ylabel)

    if abs_prev is None:
        return coefs 
    else:
        conc_prev = prev_lambert(coefs, abs_prev)
        return coefs, conc_prev
    

###############################################################################
#############################   Funções Fuzzy   ###############################
###############################################################################

def alpha_nivel (phif, range_x, alpha):
    """ Calcula o subconjunto real correspondente a pertinência alpha num número fuzzy.
    
    Args:
        phif (list of string): função de pertinencia do número fuzzy.
        range_x (array-like): valores para os quais phi = (0,1,0) na forma (xlim_esq, x_pico, xlim_dir).
        alpha (float): alpha entre 0 e 1 em questão.
    
    Return:
        alpha_nivel (list): subconjunto real correspondente ao alpha indicado para o número fuzzy.
    
    """
    
    n1, n2 = phif
    
    x_lim1, x_pico, x_lim2 = range_x
    
    x1 = np.linspace(x_lim1, x_pico, 1001)
    x2 = np.linspace(x_pico, x_lim2, 1001)
    
    alpha_lim1 = []
    alpha_lim2 = []
    
    for i in range(1001):
        if not alpha_lim1: 
            phi1_x = _aplica_funcao([[x1[i]]], n1)[0][0]
            if phi1_x == alpha:
                alpha_lim1.append(x1[i])
        if not alpha_lim2: 
            phi2_x = _aplica_funcao([[x2[i]]], n2)[0][0]
            if phi2_x == alpha:
                alpha_lim2.append(x2[i])
        else:
            break
    
    alpha_nivel = alpha_lim1 + alpha_lim2
            
    return alpha_nivel


def plota_alpha_nivel (phif, range_x, alpha, alpha_nivel):
    """ Plota o alpha nível de um número fuzzy.
    
    Args:
        phif (list of string): função de pertinencia do número fuzzy.
        range_x (array-like): valores para os quais phi = (0,1,0) na forma (xlim_esq, x_pico, xlim_dir).
        alpha (float): alpha entre 0 e 1 em questão.
        alpha_nivel (array_like): subconjunto real correspondente ao alpha indicado para o número fuzzy.
    """
        
    phi1, phi2 = phif
    
    x_lim1, x_pico, x_lim2 = range_x
    
    x1 = np.linspace(x_lim1, x_pico, 1001)
    x2 = np.linspace(x_pico, x_lim2, 1001)
    
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    
    n1 = _aplica_funcao(x1, phi1)
    n2 = _aplica_funcao(x2, phi2)
    
    plt.plot(x1, n1, color = "deeppink", label = "Número fuzzy")
    plt.plot(x2, n2, color = "deeppink")
    plt.plot([alpha_nivel[0], alpha_nivel[0], alpha_nivel[1], alpha_nivel[1]], [0, alpha, alpha, 0], color = "darkgrey", linestyle = "--", label = f"alpha = {alpha}")
    plt.plot(alpha_nivel, [0,0], color = "indigo", linewidth = 5, label = f"alpha-nível {alpha_nivel}")
    plt.title(f"Alpha-nível: {alpha}-nível = {alpha_nivel}")
    plt.xlabel("Conjunto real")
    plt.ylabel("Função de Pertinência")
    plt.grid()
    plt.legend()
    plt.show()
    
    
def phif_ntfuzzy(nt):
    """
    Define a função de pertinência phi(x) de um número triangular x = (a,b,c).

    Args:
        nt (tuple or array-like): número triangular fuzzy x na forma x = (a,b,c).
        
    Returns:
        phif (list of str): lista contendo a expressão algébrica que representa a função de pertinência de x
    """
    
    a, b, c = nt
    
    n1 = f"(x-{a})/{b-a}"
    n2 = f"({c}-x)/{c-b}"
    
    phif = [n1,n2]
    
    return phif



def f_alpha_ntfuzzy(nt):
    """
    Define um número triangular fuzzy em função de seus alpha-níveis.

    Args:
        nt (tuple or array-like): número triangular fuzzy x na forma x = (a,b,c).
        
    Returns:
        alphaf (list of str): lista contendo a expressão algébrica que representa x em função de seus alpha-níveis.
    """
        
    m,n,p = nt
    a,b,c = nt
    
    alpha1 = f"x*({b}-{a}) + {a}"
    alpha2 = f"-x*({c}-{b}) + {c}"
    
    alphaf = [alpha1, alpha2]
    
    return alphaf


def real2fuzzy(a, n_alphas):
    a_fuzzy1 = []
    for _ in range(n_alphas):
        a_fuzzy1.append(a)
        
    a_fuzzy = [a_fuzzy1, a_fuzzy1]
    
    return a_fuzzy


#####################   Operações entre números fuzzy   #######################

def soma_iterativa_0 (x1,x2):
    """
    Realiza a soma otimista (gamma zero) de dois números triangulares fuzzy, segundo Wasques et al. (2020b).

    Args:
        x1 (tuple or array-like): número triangular fuzzy x1 na forma x1 = (a,b,c).
        x2 (tuple or array-like): número triangular fuzzy x2 na forma x2 = (d,e,f).

    Returns:
        soma (tuple): número triangular fuzzy resultante da soma otimista.
        
    """
    
    #define uma lista vazia para armazenar o resultado
    x_soma = []
    
    #renomeia cada componente dos números fuzzy a serem somados
    a,b,c = x1
    d,e,f = x2
    
    #define os valores dos diâmetros dos números fuzzy
    diam_x1 = c - a
    diam_x2 = f - d
    
    #define as somas entre picos, maior e menor, e menor e maior componente dos números fuzzy
    s_picos = b+e
    s_meios = c+d
    s_extremos = a+f
    
    #realiza a soma fuzy seguindo as condições de diâmetro
    if diam_x1 >= diam_x2: # se o diâmetro do primeiro for maior que o do segundo
        x_soma.append(min(s_extremos, s_picos)) # escolhe a menor soma entre picos ou extremos para definir o menor componente da soma
        x_soma.append(s_picos) # soma os picos para definir o pico da soma
        x_soma.append(max(s_picos, s_meios)) # escolhe a maior soma entre picos ou meios para definir o maior componente da soma
        
    elif diam_x1 <= diam_x2: # se o diâmetro do primeiro for menor que o do segundo
        x_soma.append(min(s_picos, s_meios)) # escolhe a menor soma entre picos e meios para definir o menor componente da soma
        x_soma.append(s_picos) # soma os picos para definir o pico da soma
        x_soma.append(max(s_extremos, s_picos)) # escolhe a maior soma entre picos e extremos para definir o maior componente da soma
        
    #transforma a lista "soma" em uma tupla imutável, notação utilizada para números fuzzy neste trabalho
    x_soma = tuple(x_soma)
    
    return x_soma             


def oposto_fuzzy(x):
    """
    Calcula o oposto de um número fuzzy, ou seja, realiza o produto entre -1 e um número triangular fuzzy.

    Args:
        x (tuple or array-like): número triangular fuzzy x na forma x = (a,b,c).

    Returns:
        x_oposto (tuple): número triangular fuzzy oposto a x (-x = (-c, -b, -a)).
        
    """
    
    a,b,c = x
    x_oposto = (-c, -b, -a)
    return x_oposto


def divisao_intervalar(x_num, x_den):
    """
    Realiza a divisão intervalar de dois números triangulares fuzzy, sem considerar a função de pertinência.

    Args:
        x_num (tuple or array-like): número triangular fuzzy x_num na forma x_num = (an, bn, cn), numerador na divisão.
        x_den (tuple or array-like): número triangular fuzzy x_den na forma x_den = (ad, bd, cd), denominador na divisão.

    Returns:
        x_quo (tuple): número triangular fuzzy x_quo na forma x_quo = (aq, bq, cq), quociente na divisão.
        
    """
        
    an, bn, cn = x_num
    ad, bd, cd = x_den
    
    P = [an/ad, an/cd, cn/ad, cn/cd]
    
    aq = min(P)
    bq = bn/bd
    cq = max(P)
    
    x_quo = (aq,bq,cq)
    
    return x_quo
    
    
def divisao_fuzzy(x_num, x_den):
    """
    Realiza a divisão intervalar de dois números triangulares fuzzy a partir de seus alpha-niveis.

    Args:
        x_num (tuple or array-like): número triangular fuzzy x_num na forma x_num = (an, bn, cn), numerador na divisão.
        x_den (tuple or array-like): número triangular fuzzy x_den na forma x_den = (ad, bd, cd), denominador na divisão.

    Returns:
        x_quo_alpha (list of np.ndarray): lista de vetores colunas (n x 1) representando os valores de x_quo em função dos alpha-níveis, em que cada vetor corresponde a um intervalo da função de pertinência por partes.
    """
    
    a_num, b_num, c_num = x_num
    a_den, b_den, c_den = x_den
    
    alpha = np.arange(0, 1.001, 0.001).reshape((-1,1))
    
    n1 = _aplica_funcao(alpha, f_alpha_ntfuzzy(x_num)[0])
    n2 = _aplica_funcao(alpha, f_alpha_ntfuzzy(x_num)[1])
    d1 = _aplica_funcao(alpha, f_alpha_ntfuzzy(x_den)[0])
    d2 = _aplica_funcao(alpha, f_alpha_ntfuzzy(x_den)[1])

    caso1 = [(i / j).item() for i, j in zip(n1, d1)]
    caso2 = [(i / j).item() for i, j in zip(n1, d2)]
    caso3 = [(i / j).item() for i, j in zip(n2, d1)]
    caso4 = [(i / j).item() for i, j in zip(n2, d2)]

    x_q1 = []
    x_q2 = []
         
    for i in range(len(alpha)):
        q1_i = min(caso1[i], caso2[i], caso3[i], caso4[i])
        q2_i = max(caso1[i], caso2[i], caso3[i], caso4[i])
        x_q1.append(q1_i)
        x_q2.append(q2_i)
 
    x_quo_alpha = [x_q1,x_q2]
    
    return x_quo_alpha


def g_divisao_fuzzy(x_num,x_den):    
    """
    A SER IMPLEMENTADO
    
    Realiza a divisão generalizada de dois números triangulares fuzzy a partir de seus alpha-niveis, segundo Stefanini (2010).

    Args:
        x_num (tuple or array-like): número triangular fuzzy x_num na forma x_num = (an, bn, cn), numerador na divisão.
        x_den (tuple or array-like): número triangular fuzzy x_den na forma x_den = (ad, bd, cd), denominador na divisão.

    Returns:
        x_g_quo (list of np.ndarray): lista de vetores colunas (n x 1) representando os valores de x_g_quo em função dos alpha-níveis, em que cada vetor corresponde a um intervalo da função de pertinência por partes.
    """ 
    pass


def representacao_triangular(x_alpha):
    """
    Resgata valores de x(alpha) em x_alpha para os quais alpha = 0,1, expressando-os da mesma forma que um número triangular fuzzy x = (a, b, c). Note que x_alpha não é necessariamente triangular, está apenas sendo representado dessa forma.
    
    Args:
        x_alpha (list of np.ndarray): lista de vetores colunas (n x 1) representando os valores de x_alpha em função dos alpha-níveis, em que cada vetor corresponde a um intervalo da função de pertinência por partes.
        
    Returns:
        x (tuple): número fuzzy x_alpha, não necessariamente triangular, na forma de um número triangular fuzzy x = (a, b, c).
    """
    
    x1,x2 = x_alpha
    x = (x1[0],x1[-1], x2[0])
    
    return x


#########################   Sistema Linear Fuzzy   ############################

def sistema_fuzzy_2x2(U, V):
    """ 
    Resolve um sistema linear fuzzy do tipo Ux = V com duas equações e duas variáveis, onde U contém números reais, e V e x contêm números triangulares fuzzy da forma (a,b,c).
    
    Args:
        U (array-like): matriz quadrada (2x2) com os coeficientes reais do sistema linear.
        V (array-like): matriz coluna (2x1) com os termos independentes do sistema linear, representados por números triangulares fuzzy, na forma de tuplas.
        
    Returns:
        solucoes (array-like): lista contendo as diferentes matrizes solução com variáveis triangulares fuzzy encontradas.
    
    """
    
    # Transforma os elementos de U e V em float para garantir que os resultados obtidos sejam corretos
    for i in range(len(U)):
        for j in range(len(U[i])):
            U[i][j] = float(U[i][j])
            
    for i in range(len(V)):
        for j in range(len(V[i])):
            lista = list(V[i][j])
            for k in range(len(lista)):
                lista[k] = float(lista[k])  
            V[i][j] = tuple(lista)
    
    # renomeando os elementos das matrizes:
    # U = | a11    a12 |     V = | (r;s;t) |
    #     | a21    a22 |         | (u;v;w) |
    
    a11,a12 = U[0]
    a21,a22 = U[1]
    
    r,s,t= V[0][0]
    u,v,w = V[1][0]

    x=np.ones((2,3))  # define x como uma matriz 2x3 (cada coluna representa um dos valores de um número triangular fuzzy)
    
    # x  = | (a;b;c) |
    #      | (d;e;f) |
    
    # a = x[0][0]
    # b = x[0][1] 
    # c = x[0][2]
    # d = x[1][0]
    # e = x[1][1]
    # f = x[1][2]
    
    #Passo 1: achar os picos das soluções do sistema fuzzy - U[b|e] = [s|v]
    
    x[0][1],x[1][1] = metodoLU_ord(U, [[s],[v]]) # substitui os elementos da coluna central de x pelos valores obtidos na solução do sistema U[b|e] = [s|v]
    
    # faz-se 4 cópias de x - correspondentes às 4 possíveis soluções do sistema
    x1 = x.copy()
    x2 = x.copy()
    x3 = x.copy()
    x4 = x.copy()
    
    #Passo 2: 8 sistemas baseados nos sinais de U, 2 a 2 - 2 positivos e 2 negativos ou os 4 iguais (tabela 1)
    #Passo 3: 4 sistemas baseados nos sinais de U, 1 em 4 - 1 positivo e os demais negativos, ou vice-versa (Zk*x = z / tabela 2) 
    
    # define-se as condições dos sinais da matriz U
    
    situ1 = (a11>=0 and a12>=0 and a21>=0 and a22 >=0)
    situ2 = (a11<=0 and a12<=0 and a21<=0 and a22 <=0)
    situ3 = (a11>=0 and a12>=0 and a21<=0 and a22 <=0)
    situ4 = (a11<=0 and a12<=0 and a21>=0 and a22 >=0)
    
    situ5 = (a11>= 0 and a21 >=0 and a12<= 0 and a22<=0)
    situ6 = (a11>= 0 and a22 >=0 and a12<= 0 and a21<=0)
    situ7 = (a12>= 0 and a21 >=0 and a11<= 0 and a22<=0)
    situ8 = (a12>= 0 and a22 >=0 and a11<= 0 and a21<=0)
    
    situ9 = (a11>=0 and a12>=0 and a21>=0 and a22 <=0)
    situ10 = (a11>=0 and a12>=0 and a21<=0 and a22 >=0)
    
    situ11 = (a11>=0 and a12<=0 and a21>=0 and a22 >=0)
    situ12 = (a11<=0 and a12>=0 and a21>=0 and a22 >=0)
    
    situ13 = (a11>=0 and a12<=0 and a21<=0 and a22 <=0)
    situ14 = (a11<=0 and a12>=0 and a21<=0 and a22 <=0)
    
    situ15 = (a11<=0 and a12<=0 and a21>=0 and a22 <=0)
    situ16 = (a11<=0 and a12<=0 and a21<=0 and a22 >=0)
     
    # define-se as matrizes z e Z1~Z8 (Passo 3 - (Zk*x = z / tabela 2) )
    
    z= [[r],[u],[t],[w]]
    
    Z1 = [[a11,0,0,a12],
         [a21,0,a22,0],
         [0,a11,a12,0],
         [0,a21,0,a22]]
    
    Z2 = [[a11,0,0,a12],
         [0,a21,0,a22],
         [0,a11,a12,0],
         [a21,0,a22,0]]
    
    Z3 = [[0,a11,a12,0],
         [a21,0,a22,0],
         [a11,0,0,a12],
         [0,a21,0,a22]]
    
    Z4 = [[0,a11,a12,0],
         [0,a21,0,a22],
         [a11,0,0,a12],
         [a21,0,a22,0]]
    
    Z5 = [[0,a11,0,a12],
         [a21,0,0,a22],
         [a11,0,a12,0],
         [0,a21,a22,0]]
    
    Z6 = [[0,a11,0,a12],
         [0,a21,a22,0],
         [a11,0,a12,0],
         [a21,0,0,a22]]
    
    Z7 = [[a11,0,a12,0],
         [a21,0,0,a22],
         [0,a11,0,a12],
         [0,a21,a22,0]]
    
    Z8 = [[a11,0,a12,0],
         [0,a21,a22,0],
         [0,a11,0,a12],
         [a21,0,0,a22]]
    
    # testa em qual caso a matriz U se encaixa e resolve os sistemas lineares necessários - seguindo as tabelas 1 ou 2
    # substitui os valores obtidos aos seus correspondentes nas matrizes x1~x4
   
    # Passo 2: Tabela 1 - Caso 1 
    if situ1 or situ2 or situ3 or situ4:
        #print("A matriz U se encaixa no caso 1")
        # a,f
        # c,d
        
        x1[0][0], x1[1][2] = metodoLU_ord(U,[[r],[u]])
        x1[0][2], x1[1][0] = metodoLU_ord(U,[[t],[w]])
                
        x2[0][0], x2[1][2] = metodoLU_ord(U,[[r],[w]])
        x2[0][2], x2[1][0] = metodoLU_ord(U,[[t],[u]])
                
        x3[0][0], x3[1][2] = metodoLU_ord(U,[[t],[u]])
        x3[0][2], x3[1][0] = metodoLU_ord(U,[[r],[w]])
                
        x4[0][0], x4[1][2] = metodoLU_ord(U,[[t],[w]])
        x4[0][2], x4[1][0] = metodoLU_ord(U,[[r],[u]])
        
    # Passo 2: Tabela 1 - Caso 2     
    elif situ5 or situ6 or situ7 or situ8: 
        #print("A matriz U se encaixa no caso 2")
        # a,d
        # c,f
                
        x1[0][0], x1[1][0] = metodoLU_ord(U,[[r],[u]])
        x1[0][2], x1[1][2] = metodoLU_ord(U,[[t],[w]])
                
        x2[0][0], x2[1][0] = metodoLU_ord(U,[[r],[w]])
        x2[0][2], x2[1][2] = metodoLU_ord(U,[[t],[u]])
                
        x3[0][0], x3[1][0] = metodoLU_ord(U,[[t],[u]])
        x3[0][2], x3[1][2] = metodoLU_ord(U,[[r],[w]])
                
        x4[0][0], x4[1][0] = metodoLU_ord(U,[[t],[w]])
        x4[0][2], x4[1][2] = metodoLU_ord(U,[[r],[u]])
    
    # Passo 3: Tabela 2 - Caso 3 
    elif situ9 or situ10 or situ15 or situ16:
        #print("A matriz U se encaixa no caso 3")
        # a,c,d,f
        x1[0][0], x1[0][2], x1[1][0], x1[1][2] = metodoLU_ord(Z1, z)
        x2[0][0], x2[0][2], x2[1][0], x2[1][2] = metodoLU_ord(Z2, z)
        x3[0][0], x3[0][2], x3[1][0], x3[1][2] = metodoLU_ord(Z3, z)
        x4[0][0], x4[0][2], x4[1][0], x4[1][2] = metodoLU_ord(Z4, z)
        
    # Passo 3: Tabela 2 - Caso 4         
    elif situ11 or situ12 or situ13 or situ14:
        #print("A matriz U se encaixa no caso 4")
        # a,c,d,f
        x1[0][0], x1[0][2], x1[1][0], x1[1][2] = metodoLU_ord(Z5, z)
        x2[0][0], x2[0][2], x2[1][0], x2[1][2] = metodoLU_ord(Z6, z)
        x3[0][0], x3[0][2], x3[1][0], x3[1][2] = metodoLU_ord(Z7, z)
        x4[0][0], x4[0][2], x4[1][0], x4[1][2] = metodoLU_ord(Z8, z)
                
    #Passo 4: testa a<=b<=c e d<=e<=f para delvolver as soluções possíveis que seguem a estrutura de números triangulares fuzzy
    x1_solucao = (x1[0][0]<= x1[0][1]<=x1[0][2]) and (x1[1][0]<= x1[1][1] <= x1[1][2])
    x2_solucao = (x2[0][0]<= x2[0][1]<=x2[0][2]) and (x2[1][0]<= x2[1][1] <= x2[1][2])
    x3_solucao = (x3[0][0]<= x3[0][1]<=x3[0][2]) and (x3[1][0]<= x3[1][1] <= x3[1][2])
    x4_solucao = (x4[0][0]<= x4[0][1]<=x4[0][2]) and (x4[1][0]<= x4[1][1] <= x4[1][2])
    
    # transforma os elementos das matrizes x em tuplas, seguindo a estrutura de numeros fuzzy definida no código
    x1[0] = tuple(x1[0])
    x1[1] = tuple (x1[1])
    
    x2[0] = tuple(x2[0])
    x2[1] = tuple (x2[1])
    
    x3[0] = tuple(x3[0])
    x3[1]= tuple (x3[1])
    
    x4[0] = tuple(x4[0])
    x4[1]= tuple (x4[1])
    
    # lista com as soluções possíveis para o sistema
    solucao = []
    
    # testa as soluções: se for True, printa a solução e adiciona na lista solucao
    if x1_solucao:
        #print(f" A matriz {x1} é solução do sistema linear fuzzy.")
        solucao.append(x1)
        
    if x2_solucao:
        #print(f" A matriz {x2} é solução do sistema linear fuzzy.")
        solucao.append(x2)
            
    if x3_solucao:
        #print(f" A matriz {x3} é solução do sistema linear fuzzy.")
        solucao.append(x3)
            
    if x4_solucao:
        #print(f" A matriz {x4} é solução do sistema linear fuzzy.")  
        solucao.append(x4)
        
    # retorna as soluções possíveis    
    return solucao


###################  Método de Quadrados Mínimos Fuzzy   ######################

def mqm_fuzzy (x_data, y_data):
    """
    Resolve um problema de regressão linear fuzzy pelo método dos quadrados mínimos para duas variáveis.
    
    Args:
        x_data (array-like): matriz coluna de dados reais de entrada.
        y_data (array-like): matriz coluna de dados fuzzy de saída, na forma de tuplas (a, b, c).
        
    Returns:
        solucao (list): lista contendo as soluções do sistema, com os coeficientes fuzzy representados como tuplas.
        
    """
        
    n = len(x_data)
    
    x_extended = np.arange(x_data[0][0], x_data[-1][0])
    
    g1_arr = _aplica_funcao(x_data,"x")
    g2_arr = _aplica_funcao(x_data,"1")
    
    W = np.zeros((n,2))
    
    for i in range(n):
        W[i][0] = g1_arr[i][0]
        W[i][1] = g2_arr[i][0]
        
    Wt = W.T

    U = np.dot(Wt,W)
    
    V0 = np.zeros((2,n),dtype=object)
    #Matriz V n colunas, cada coluna sendo uma lista de três numeros (equivlentes a um número triângular fuzzy)
    for i in range(n):
        for y in y_data[i]:
            V0[0][i] = tuple(g1_arr[i] * y)
            V0[1][i] = tuple(g2_arr[i] * y)
    
    V = [[(0,0,0)],[(0,0,0)]]
    
    for i in range (n):
        l1 = soma_iterativa_0(V[0][0], V0[0][i])
        l2 = soma_iterativa_0(V[1][0], V0[1][i])  
        V[0][0] = l1
        V[1][0] = l2
        
    #print(f"Seu sistema linear fuzzy é Uc = V, com U = Wt.W = {U} e V = Wt.Y = {V}.")
    #print(" ")
    
    solucao = np.round(sistema_fuzzy_2x2(U, V),5)
    
    funcao = []
    
    for s in range (len(solucao)):
        
        alpha1 = tuple(solucao[s][0])
        alpha2 = tuple(solucao[s][1])
        
        solucao[s][0] = alpha1
        solucao[s][1] = alpha2
        
        funcao.append(f"{alpha1}*x +_0 {alpha2}*1")
    
        #print(f"A função {alpha1}*x +_0 {alpha2}*1 é solução do problema.")

    return solucao


#####################   Plot 2D de soluções MQM Fuzzy   #######################

def mqm_fuzzy_2Dplot(x_data, y_data, solucao, x_real=None, prevw = 0, titulo_grafico = "Ajuste por mínimos quadrados fuzzy", xlabel = "Dados reais", ylabel = "Dados Fuzzy"):
        
    """
    Gera um gráfico 2D das soluções obtidas no ajuste por mínimos quadrados fuzzy, com visualização da banda fuzzy e previsão opcional.

    Args:
        x_data (array-like): matriz coluna de dados reais de entrada.
        y_data (array-like): matriz coluna de dados fuzzy de saída, na forma de tuplas (a, b, c)
        solucao (list): lista contendo as soluções do sistema, com os coeficientes fuzzy da banda ajustada representados como tuplas.
        x_real (array-like, optional): valores reais originais de x, se diferentes dos usados em x_data (padrão: None).
        prevw (int, optional): número de pontos futuros a prever (padrão: 0).
        titulo_grafico (str, optional): título do gráfico (padrão: "Ajuste por mínimos quadrados fuzzy").
        xlabel (str, optional): rótulo do eixo x (padrão: "Dados Reais").
        ylabel (str, optional): rótulo do eixo y (padrão: "Dados Fuzzy").
        
    """    
    
    if x_real is None:
        x_real = x_data
        
    n = len(x_data) 
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    x_extended = np.arange(x_data[0][0], x_data[-1][0]+prevw+0.5, 1)
    
    cores = ["#5f2ad5", "#ff1e97", "#ffd700", "#244572"]
    
    plt.figure(dpi=200)
    legend_labels = set() 
    plt.plot([], [], color="#666666", marker = ".", ms = 5, label="Dados")
    plt.plot([], [], color="#111111", label="Pico das soluções")
    
    for s in range (len(solucao)):
        
        alpha1 = solucao[s][0]
        alpha2 = solucao[s][1]
        
        resultados = []
        elemento = solucao[s][0]
        
        for x in x_extended:
#         for x in x_data:
            prod = []
            for k in range (3):
                prod.append(np.round(elemento[k]*x,5))
                
            resultados.append(soma_iterativa_0(prod, solucao[s][1]))

        inf = []
        pico = []
        sup = []

        for r in resultados:
            inf.append(r[0])
            pico.append(r[1])
            sup.append(r[2])
            
        plt.plot(x_extended,inf,color=cores[s], linewidth=0.5)
        plt.plot(x_extended,sup,color=cores[s],linewidth=0.5, label=f"Solução {s+1}")
        plt.fill_between(x_extended, inf, sup, color=cores[s], alpha=0.2)
        plt.plot(x_extended,pico,color="#111111", linewidth=0.5) 
    
    if np.array_equal(x_real, x_data):
        for i in range(n):
            a = x_data[i][0]
            b = tuple(y_data[i][0])
            plt.plot([a,a,a], b, color="#666666")
            plt.scatter([a],b[1],color="#666666", s = 5)
    else:
        for i in range(n):
            a = x_data[i][0]
            a_real = x_real[i][0]
            b = tuple(y_data[i][0])
            plt.plot([a,a,a], b, color="#666666")
            plt.scatter([a],b[1],color="#666666", s= 5)
    
    # plotando os dados de previsão
    x_prevw = np.zeros((prevw,1))
    x_prevw_real = np.zeros((prevw,1))
    
    for p in range(prevw):
        x_prevw[p][0] = x_data[-1][0]+1+p
        x_prevw_real[p][0] = x_data_real[-1][0]+1+p
        
    for s in range (len(solucao)):
        
        alpha1 = solucao[s][0]
        alpha2 = solucao[s][1]
        
        resultsp = []
        elemento = solucao[s][0]
        
        for xp in x_prevw:
            prodp = []
            for k in range (3):
                prodp.append((elemento[k]*xp[0]))
                
            resultsp.append(soma_iterativa_0(prodp, solucao[s][1]))
        
        for p in range(prevw):
            xp = x_prevw[p][0]
            xp_real = x_prevw_real[p][0]
            yp = tuple(np.round(resultsp[p],5))
            plt.plot([xp,xp,xp], yp, color="#888888")
            plt.scatter([xp,xp,xp],yp,color=cores[s], s = 5)
            plt.scatter([xp],yp[1],color="#888888", s = 5, alpha= 0.5)
            
    if prevw != 0:
        plt.plot([], [], color="#888888", marker = ".", ms = 5, label="Previsão")

    #plt.xticks(x_non_ticks,x_ticks)
    plt.title(f"{titulo_grafico} (Gráfico 2D de Superfície)")
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0), borderaxespad=0.5)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{titulo_grafico} (Gráfico 2D de Superfície)",bbox_inches='tight')
    plt.show()
    

#####################   Plot 3D de soluções MQM Fuzzy   #######################
    
def mqm_fuzzy_3Dplot(x_data, y_data, solucao, x_real=None, prevw = 0, titulo_grafico = "Ajuste por mínimos quadrados fuzzy", xlabel = "Dados reais", ylabel = "Dados fuzzy"):
    
    """
    Gera um gráfico 3D das soluções obtidas no ajuste por mínimos quadrados fuzzy, sendo o eixo z o grau de pertinência entre 0 e 1, com visualização da banda fuzzy e previsão opcional.

    Args:
        x_data (array-like): matriz coluna de dados reais de entrada.
        y_data (array-like): matriz coluna de dados fuzzy de saída, na forma de tuplas (a, b, c)
        solucao (list): lista contendo as soluções do sistema, com os coeficientes fuzzy da banda ajustada representados como tuplas.
        x_real (array-like, optional): valores reais originais de x, se diferentes dos usados em x_data (padrão: None).
        prevw (int, optional): número de pontos futuros a prever (padrão: 0).
        titulo_grafico (str, optional): título do gráfico (padrão: "Ajuste por mínimos quadrados fuzzy").
        xlabel (str, optional): rótulo do eixo x (padrão: "Dados Reais").
        ylabel (str, optional): rótulo do eixo y (padrão: "Dados Fuzzy").
        
    """    
    
    if x_real is None:
        x_real = x_data
        
    n = len(x_data)    
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    x_extended = np.arange(x_data[0][0], x_data[-1][0]+prevw+0.5, 1)
    
    cores = ["#5f2ad5", "#ff1e97", "#ffd700", "#244572"]
    
    z0 = [0] * len(x_extended)
    z1 = [1] * len(x_extended)
     
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    
    legend_labels_3D= set()
    ax.plot([], [], color="#666666", marker = ".", ms = 5, label="Dados")
    ax.plot([], [], [],color="#111111", label="Pico das soluções")
    
    for s in range (len(solucao)):
        
        alpha1 = solucao[s][0]
        alpha2 = solucao[s][1]
        
        resultados = []
        elemento = solucao[s][0]
        
        for x in x_extended:
#         for x in x_data:
            prod = []
            for k in range (3):
                prod.append(round(elemento[k]*x,5))
                
            resultados.append(soma_iterativa_0(prod, solucao[s][1]))

        inf = []
        pico = []
        sup = []

        for r in resultados:
            inf.append(r[0])
            pico.append(r[1])
            sup.append(r[2])

        ax.plot(x_extended,inf,z0,linewidth=0.5, color=cores[s])
        ax.plot(x_extended,sup,z0,linewidth=0.5, color=cores[s],label=f"Solução {s+1}")
        ax.plot_surface(np.array([x_extended, x_extended]), np.array([inf,pico]), np.array([z0, z1]), color=cores[s], alpha=0.3)
        ax.plot_surface(np.array([x_extended, x_extended]), np.array([pico, sup]), np.array([z1, z0]), color=cores[s], alpha=0.3)
        ax.plot(x_extended,pico,z1,linewidth=0.5,color="#111111")

    if np.array_equal(x_real, x_data):
        for i in range(n):
            a = x_data[i][0]
            b = tuple(y_data[i][0])
            ax.plot([a,a,a], b, [0,1,0], color="#666666", linewidth=0.8)
            ax.scatter([a],b[1],z1,color="#666666", s = 5)
            
    else:
        for i in range(n):
            a = x_data[i][0]
            a_real = x_real[i][0]
            b = tuple(y_data[i][0])
            ax.plot([a,a,a], b, [0,1,0],color="#666666", linewidth=0.8)
            ax.scatter([a],b[1],z1,color="#666666", s = 5)
        
    # plotando os dados de previsão
    x_prevw = np.zeros((prevw,1))
    x_prevw_real = np.zeros((prevw,1))
    
    for p in range(prevw):
        x_prevw[p][0] = x_data[-1][0]+1+p
        x_prevw_real[p][0] = x_data_real[-1][0]+1+p
        
    for s in range (len(solucao)):
        
        alpha1 = solucao[s][0]
        alpha2 = solucao[s][1]
        
        resultsp = []
        elemento = solucao[s][0]
        
        for xp in x_prevw:
            prodp = []
            for k in range (3):
                prodp.append((elemento[k]*xp[0]))
                
            resultsp.append(soma_iterativa_0(prodp, solucao[s][1]))
            
        for p in range(prevw):
            xp = x_prevw[p][0]
            xp_real = x_prevw_real[p][0]
            yp = tuple(np.round(resultsp[p],5))
            plt.plot([xp,xp,xp], yp,[0,1,0], linewidth=0.8, marker=".",color=cores[s], label=f'Previsão {p+1} (Solução {s+1}): x = {round(xp)} ({round(xp_real)}), y = {yp}')
            plt.plot([xp],yp[1],[1],linewidth=0.8, marker=".",color="#888888")
    
    if prevw != 0:
        plt.plot([], [], color="#888888", marker = ".", s = 5, label="Previsão")

#     ax.set_xticks(x_non_ticks)
#     ax.set_xticklabels(x_ticks)
    ax.set_title(titulo_grafico)
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0), borderaxespad=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Pertinência')
    ax.view_init(elev=20, azim=30)
    
    plt.savefig(titulo_grafico, bbox_inches='tight')
    plt.show()
    

########################   Previsão Inversa MQM fuzzy   #######################

def prev_mqmi_fuzzy(y_data, solucoes):
    """  
    
    Soluciona o problema inverso de uma regressão linear fuzzy, prevendo valores em x.
    
    Args:
        y_data (array-like): matriz coluna de dados fuzzy de saída, na forma de tuplas (a, b, c).
        solucoes (list): lista contendo as soluções do sistema, com os coeficientes fuzzy da banda ajustada representados como tuplas.
        
    Returns:
        x_prev (list of list): lista contendo, para cada solução de mqm fuzzy, uma lista de vetores colunas (n x 1) representando os valores de x_prev em função dos alpha-níveis, em que cada vetor corresponde a um intervalo da função de pertinência por partes.
        
    """
    
    x_prev = []
    for s in range(len(solucoes)):
        alpha1 = tuple(solucoes[s][0])
        alpha2 = tuple(solucoes[s][1])
        
        op_alpha2 = oposto_fuzzy(alpha2)
        
        x_prev_local = []
        
        for i in range(len(y_data)):
            #x = (y -0 alpha2)/alpha1
            
            soma = soma_iterativa_0(y_data[i][0], op_alpha2)
            x = divisao_fuzzy(soma, alpha1)
            
            x_prev_local.append(x)
        
        x_prev.append(x_prev_local)
          
    return x_prev
    
    
#####################   Plota Previsão Inversa MQM fuzzy   ####################

def prev_mqmi_fuzzy_2Dplot(x_data, y_data, x_data_prev, y_data_prev, xlabel = r"$X_{fuzzy}$", ylabel = r"$Y_{fuzzy}$"):
    """
    Plota um conjunto de dados com x real e y fuzzy, e as previsões feitas pelo inverso do método de quadrados mínimos fuzzy com x fuzzy predito e y fuzzy conhecido. 
    
    Args:
        x_data (array-like): conjunto de números reais conhecidos associados a cada y fuzzy.
        y_data (array-like): lista de tuplas com três elementos, representando números triangulares fuzzy correspondentes aos valores reais de x.
        x_data_prev (array-like of array-like): lista das pertinencia encontradas em função dos alpha níveis, predita a partir de cada y_data_prev conhecido.
        y_data_prev (array-like of array-like): lista de tuplas com três elementos, representando números triangulares fuzzy conhecidos para os quais se prevê os x fuzzy desconhecidos.
    """
    
    cores = ["#5f2ad5", "#ff1e97", "#ffd700", "#244572"] 
    
    plt.figure(dpi=200)
    plt.plot([], [], color="#666666", marker = ".", ms = 5, label="Dados")
    
    
    for i in range(len(x_data)):
        plt.plot([x_data[i], x_data[i], x_data[i]], y_data[i][0], color = "#666666", linewidth=2)
        plt.scatter(x_data[i], y_data[i][0][1], color = "#666666", s=5)
    
    for s in range(len(x_data_prev)):
        plt.plot([], [], color=cores[s], alpha = 0.8, marker = ".", ms = 5, label=f"Previsão da S{s+1}")
        for i in range (len(x_data_prev[s])):
            y = [y_data_prev[i][0][1] for _ in range(len(x_data_prev[s][i][0]))]
            plt.plot(x_data_prev[s][i][0], y, color = cores[s], linewidth=1, alpha = 0.75)
            plt.plot(x_data_prev[s][i][1], y, color = cores[s], linewidth=1, alpha = 0.75)
            x_pico = x_data_prev[s][i][0][-1]
            y_pico = y_data_prev[i][0][1]
            plt.scatter(x_pico, y_pico, color = cores[s], alpha = 0.75, s=5)
            
            plt.plot([x_pico, x_pico, x_pico], y_data_prev[i][0], color = "#666666", alpha=0.5, linewidth=1)
            plt.scatter(x_pico, y_data_prev[i][0][1], color = "#666666", alpha=0.5, s=5)
            
    plt.title(f"Previsões de {xlabel} (Gráfico 2D de Superfície)")
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0), borderaxespad=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()
    
    
def prev_mqmi_fuzzy_3Dplot(x_data, y_data, x_data_prev, y_data_prev, xlabel = r"$X_{fuzzy}$", ylabel = r"$Y_{fuzzy}$"):
    """
    Plota um conjunto de dados com x real e y fuzzy, e as previsões feitas pelo inverso do método de quadrados mínimos fuzzy com x fuzzy predito e y fuzzy conhecido - associados a função de pertinecia dos números fuzzy.
    
    Args:
        x_data (array-like): conjunto de números reais conhecidos associados a cada y fuzzy.
        y_data (array-like): lista de tuplas com três elementos, representando números triangulares fuzzy correspondentes aos valores reais de x.
        x_data_prev (array-like of array-like): lista das pertinencia encontradas em função dos alpha níveis, predita a partir de cada y_data_prev conhecido.
        y_data_prev (array-like of array-like): lista de tuplas com três elementos, representando números triangulares fuzzy conhecidos para os quais se prevê os x fuzzy desconhecidos.
    """
    
    cores = ["#5f2ad5", "#ff1e97", "#ffd700", "#244572"] 

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    
    legend_labels_3D= set()
    ax.plot([], [], color="#666666", marker = ".", ms = 5, label="Dados")
    
    
    for i in range(len(x_data)):
        ax.plot([x_data[i][0], x_data[i][0], x_data[i][0]], y_data[i][0], [0,1,0], color = "#666666", linewidth=2)
        ax.scatter(x_data[i], y_data[i][0][1], 1, color = "#666666", s=5)
    
    for s in range(len(x_data_prev)):
        ax.plot([], [], color=cores[s], alpha = 0.8, marker = ".", ms = 5, label=f"Previsão da S{s+1}")
        for i in range (len(x_data_prev[s])):
            y = [y_data_prev[i][0][1] for _ in range(len(x_data_prev[s][i][0]))]
            z = np.linspace(0,1, len(x_data_prev[s][i][0]))
            ax.plot(x_data_prev[s][i][0], y, z, color = cores[s], linewidth=1, alpha = 0.8)
            ax.plot(x_data_prev[s][i][1], y, z, color = cores[s], linewidth=1, alpha = 0.8)
            x_pico = x_data_prev[0][i][s][-1]
            y_pico = y_data_prev[i][0][1]
            ax.scatter(x_pico, y_pico, 1, color = cores[s], alpha = 0.8, s=5)
            
            ax.plot([x_pico, x_pico, x_pico], y_data_prev[i][0], [0,1,0], color = "#666666", alpha=0.5, linewidth=1)
            ax.scatter(x_pico, y_data_prev[i][0][1], 1, color = "#666666", alpha=0.5, s=5)
            
    ax.set_title(f"Previsões de {xlabel}")
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0), borderaxespad=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Pertinência')
    ax.view_init(elev=20, azim=75)
    
#################   Lidando com o diâmetro de números fuzzy   #################

def calcula_diametros(x_data, y_data):
    diametros = np.zeros((len(x_data),1))
    
    for i in range(len(x_data)):
        diam = y_data[i][0][-1] - y_data[i][0][0]
        diametros[i][0] = diam
        
    return diametros


def recupera_diametros(x_data, solucoes):
    
    diametros = np.zeros((len(solucoes),(len(x_data))))
    
    for s in range(len(solucoes)):
        for i in range(len(x_data)):
            prod = []
            for k in range(3):
                prod.append(solucoes[s][0][k]*x_data[i])
            y = soma_iterativa_0(prod, solucoes[s][1])
            diam = y[-1] - y[0]
            diametros[s][i] = diam
    
    return diametros


def plota_diametros(x_data, y_data, solucoes):
    
    plt.figure(dpi=200)
    
    diametros_s = recupera_diametros(x_data,solucoes)
    diametros_y = calcula_diametros(x_data, y_data)
    
    cores = ["#5f2ad5", "#ff1e97", "#ffd700", "#244572"]
   
    #plt.plot(x_data, diametros_y, color ="#666666", label = f"Dados", marker = ".", ms = 5)

    for s in range(len(solucoes)):
        plt.plot(x_data, diametros_s[s], color = cores[s], label = f"Solução {s+1}", marker = ".", ms = 5)
        
    plt.title(r"Diâmetro do $Y_{fuzzy}$ para cada $x_{real}$")
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0), borderaxespad=0.5)
    plt.grid()
    plt.xlabel("x$_{real}$")
    plt.ylabel(r"Diâmetro de $Y_{fuzzy}$")
    plt.savefig(f"diametro_por_x",bbox_inches='tight')
    plt.show()
    
    
def calcula_diametros_prev(x_data, y_data):
 
    diametros = np.zeros((len(x_data),1))
    
    for i in range(len(x_data)):
        diam = y_data[i][0][-1] - y_data[i][0][0]
        diametros[i][0] = diam
        
    return diametros


def plota_diametros_prev(x_prev, y_data):
    """
    Plota os diâmetros previstos de X_fuzzy em função dos picos de Y_fuzzy.

    Parâmetros:
        x_prev (list or np.array): Lista contendo os valores previstos de X_fuzzy.
        y_data (list): Lista de listas representando os valores fuzzy de Y.
    """
    
    cores = ["#5f2ad5", "#ff1e97", "#ffd700", "#244572"]  
    
    plt.figure(dpi=200)
 
    y_picos = [y_data[i][0][1] for i in range(len(x_prev[0]))]  
    
    for s in range(len(x_prev)):
        # Calcula os diâmetros para as duas soluções fuzzy
        diam_xp = calcula_diametros_prev(y_picos, x_prev[s])

        plt.plot(y_picos, diam_xp, color=cores[s], label=f"Previsão da S{1+s}", marker=".", ms=5)

    plt.title(r"Diâmetro do $X_{fuzzy}$ previsto para cada pico de $Y_{fuzzy}$")
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0), borderaxespad=0.5)
    plt.grid()
    plt.xlabel("Pico de $Y_{fuzzy}$")
    plt.ylabel(r"Diâmetro de $X_{fuzzy}$ previsto")
    #plt.yscale("log")
    plt.tight_layout()
    plt.show()


##############################   Métricas   ###################################

def media(lista):
    n = len(lista)
    
    soma = 0
    for e in lista:
        soma+=e
    med = soma/n
    return med


def metr_maximo(a, b):
    
    n_coords = len(a)
    subs = [] 
    
    for i in range(n_coords):
        subs.append(abs(a[i]-b[i]))
        
    mmax = max(subs)
    
    return mmax  


def metr_hausdorff(x1, x2):
    
    n_alphas = len(x1[0])
    beta = []
    
    #beta = (max|a1 - b1|, |a2 - b2|)
    for alpha in range(n_alphas):
        beta.append(max(abs(x1[0][alpha] - x2[0][alpha]), abs(x1[1][alpha] - x2[1][alpha])))
    
    # sup beta
    mhaus=max(beta)
    
    return mhaus  
    
    
################   Ajuste de curva fuzzy com plot 2D e 3D   ###################
    
def ajuste_fuzzy(x_data, y_data, x_real = None, prevw = 0, titulo_grafico = "Ajuste por mínimos quadrados fuzzy", xlabel = "Dados Reais", ylabel = "Dados Fuzzy"):
    
    """
    Executa o ajuste por mínimos quadrados fuzzy, gerando gráficos 2D e 3D das soluções encontradas, com opção de previsão futura.

    Args:
        x_data (array-like): matriz coluna de dados reais de entrada.
        y_data (array-like): matriz coluna de dados fuzzy de saída, na forma de tuplas (a, b, c).
        x_real (array-like, optional): valores reais originais de x, se diferentes dos usados em x_data (padrão: None).
        prevw (int, optional): número de pontos futuros a prever (padrão: 0).
        titulo_grafico (str, optional): título dos gráficos gerados (padrão: "Ajuste por mínimos quadrados fuzzy").
        xlabel (str, optional): rótulo do eixo x nos gráficos (padrão: "Dados Reais").
        ylabel (str, optional): rótulo do eixo y nos gráficos (padrão: "Dados Fuzzy").

    Returns:
        solucao (list): lista contendo as soluções do sistema, com os coeficientes fuzzy da banda ajustada representados como tuplas.
    """

    x_ord, y_ord, x_real_ord = ordena_dados_crescente(x_data, y_data, x_real)
    solucoes = mqm_fuzzy (x_ord, y_ord)
    mqm_fuzzy_2Dplot(x_ord, y_ord, solucoes, x_real_ord, prevw, titulo_grafico, xlabel, ylabel)
    mqm_fuzzy_3Dplot(x_ord, y_ord, solucoes, x_real_ord, prevw, titulo_grafico, xlabel, ylabel)
    plota_diametros(x_ord, y_ord, solucoes)
    
    return solucoes


################   Lei de Lambert fuzzy com plot 2D e 3D   ###################
    
def lei_lambert_fuzzy(x_data, y_data, x_real = None, y_prev = None, titulo = "Curva de calibração fuzzy", xlabel = "Concentração", ylabel = "Absorbância"):
    
    """
    Executa o ajuste por mínimos quadrados fuzzy, gerando gráficos 2D e 3D das soluções encontradas, com opção de previsão futura.

    Args:
        x_data (array-like): matriz coluna de dados reais de entrada.
        y_data (array-like): matriz coluna de dados fuzzy de saída, na forma de tuplas (a, b, c).
        x_real (array-like, optional): valores reais originais de x, se diferentes dos usados em x_data.
        y_prev (array-like, optional): valores fuzzy de y para os quais se deseja prever x.
        titulo (str, optional): título dos gráficos gerados.
        xlabel (str, optional): rótulo do eixo x nos gráficos.
        ylabel (str, optional): rótulo do eixo y nos gráficos.

    Returns:
        solucao (list): lista contendo as soluções do sistema, com os coeficientes fuzzy da banda ajustada representados como tuplas.
    """

    x_ord, y_ord, x_real_ord = ordena_dados_crescente(x_data, y_data, x_real)
    solucoes = mqm_fuzzy (x_ord, y_ord)
    
    mqm_fuzzy_2Dplot(x_ord, y_ord, solucoes, x_real_ord, titulo_grafico = titulo, xlabel = xlabel, ylabel = ylabel)
    mqm_fuzzy_3Dplot(x_ord, y_ord, solucoes, x_real_ord, titulo_grafico = titulo, xlabel = xlabel, ylabel = ylabel)
    plota_diametros(x_ord, y_ord, solucoes)
    
    if y_prev is not None:
        x_prev = prev_mqmi_fuzzy(y_prev, solucoes)
        
        prev_mqmi_fuzzy_2Dplot(x_ord, y_ord, x_prev, y_prev, xlabel = xlabel, ylabel = ylabel)
        prev_mqmi_fuzzy_3Dplot(x_ord, y_ord, x_prev, y_prev, xlabel = xlabel, ylabel = ylabel)
        plota_diametros_prev(x_prev, y_prev)
    
        return solucoes, x_prev
    
    else:
        return solucoes