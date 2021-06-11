from scipy.optimize import minimize
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import ctypes
import sys


def e_func(z, param, omega_k):
    omega_m_factor = param[0] * (1 + z) ** 3
    omega_de_factor = param[1] * (1 + z) ** (3 + 3 * param[2])
    omega_k_factor = omega_k * (1 + z) ** 2

    return np.sqrt(omega_m_factor + omega_de_factor + omega_k_factor)


def calc_trapezium_formula(f, lim, n):
    # Função para o cálculo da relação de recorrência do trapézio
    h = (lim[1] - lim[0]) / n  # Calcula o valor de h
    # Retorna a soma (multiplicada por h) dos valores calculados na função f,
    # Mas os valores enviados/calculados são apenas os que ainda não foram calculados previamente.
    return h * sum(f(np.arange(lim[0] + h, lim[1], 2 * h)))


def integral_calc(f, lim, eps_desired=1E-3):
    # Função que calcula a integral pelo método dos trapézios

    # verifica se os limites são diferentes
    if lim[1] == lim[0]:
        return 0, 0

    # Transforma os limites de integração lim para float128 para uma precisão maior do resultado.
    # Caso já entrem na função com a precisão de float128, nada é alterado.
    lim = list(map(np.float128, lim))

    # Calcula o primeiro resultado para n = 1
    result = (f(lim[0]) + f(lim[1])) * (lim[1] - lim[0]) / 2

    # for i in np.power(2, np.arange(1, np.log(n)/np.log(2) + 1, 1)):  # Somente para teste de algum n específico
    count = 0
    eps = 1
    while eps >= eps_desired:  # Executa enquanto o epsilon for maior que o desejado
        count += 1  # Contador para determinar o n a ser enviado
        result_old = result  # Salva o resultado anterior
        # Atribui o novo valor de acordo com a fórmula de recorrência
        result = result / 2 + calc_trapezium_formula(f, lim, 2 ** count)
        eps = np.abs((result - result_old) / result)  # Calcula o novo epsilon

    return result, 2 ** count  # Retorna o resultado e o número de intervalos


def comoving_distance(c, h0, z, param, precision=1E-10):
    omega_k = 1 - (param[0] + param[1])
    hubble_distance = c / h0

    lim = [0, z]
    integration_result = integral_calc(lambda x: 1 / e_func(x, param, omega_k), lim, eps_desired=precision)[0]

    if omega_k == 0:
        factor_k = hubble_distance * integration_result
    elif omega_k > 0:
        sqr_om = np.sqrt(omega_k)
        factor_k = hubble_distance / sqr_om * np.sinh(sqr_om * integration_result)
    else:
        sqr_om = np.sqrt(np.abs(omega_k))
        factor_k = hubble_distance / sqr_om * np.sin(sqr_om * integration_result)

    return factor_k


def luminosity_distance(c, h0, z, param, precision=1E-10):
    return (1 + z) * comoving_distance(c, h0, z, param, precision=precision)


def dist_mod(dist_lum):
    # dist_lum is the luminosity distance in Mpc
    return 5 * np.log10(dist_lum * 10 ** 6 / 10)


def lumin_dist_mod_func(h0, redshift, m_dens, de_dens, w, show=False, precision=1E-10):
    c = 299792458  # velocidade da luz em m/s

    # h0 constante de Hubble km s-1 Mpc-1
    mpc_to_km = 3.086E+19  # conversão de Mpc para km
    h0 = h0 / mpc_to_km  # Constante de Hubbl em 1/s

    z = redshift  # definição de redshift
    param = np.array([m_dens, de_dens, w])  # parametros (densidade de matéria, densidade de energia escura, w)

    lum_dist_val = luminosity_distance(c, h0, z, param, precision)
    lum_dist_val = lum_dist_val * 10 ** -3 / mpc_to_km  # conversão de m para Mpc

    dist_mod_val = dist_mod(lum_dist_val)

    if show:
        print("-" * 39)
        print("| Parâmetros aplicados        | Valor |")
        print("-" * 39)
        print("| Redshift                    | {:^5} |\n"
              "| Densidade de Matéria        | {:^5} |\n"
              "| Densidade de Energia Escura | {:^5} |\n"
              "| Param. Equação de Estado    | {:^5} |"
              "".format(z, m_dens, de_dens, w))
        print("-" * 39 + "\n")

        print("-" * 46)
        print("| Resultado                   |    Valor     |")
        print("-" * 46)
        print("| Distância de Luminosidade   | {:.2e} Mpc |\n"
              "| Módulo da Distância         | {:^12.2f} |"
              "".format(lum_dist_val, dist_mod_val))
        print("-" * 46, "\n")

    return lum_dist_val, dist_mod_val


def calc_chi(h0, real_data, params, precision=1E-10):
    chi2 = 0
    for i in range(len(real_data[0])):
        teor_data = lumin_dist_mod_func(h0, real_data[0][i], params[0], params[1], params[2], precision=precision)[1]
        chi2 += ((real_data[1][i] - teor_data) / real_data[2][i]) ** 2

    return chi2


def read_fl(fl_name):
    return np.loadtxt(fl_name, skiprows=1).T


def map_chi(h0, data, params_array, c_lib, fl_name, precision=1E-10):
    cte = None
    row_cte = None
    cte_array = None

    for i in range(len(params_array)):
        for j in range(len(params_array[i]) - 1):
            if params_array[i][j] != params_array[i][j + 1]:
                cte = None
                break
            cte = params_array[i][0]
            row_cte = i
        if cte is not None:
            cte_array = [cte, row_cte]

    rows_to_map = [0, 1, 2]
    rows_to_map.remove(cte_array[1])

    map_array = []
    for i in params_array[rows_to_map[0]]:
        part_map = []
        for j in params_array[rows_to_map[1]]:
            if cte_array[1] == 0:
                # params = [omega_m, omega_ee, w]
                params = [cte_array[0], i, j]
            elif cte_array[1] == 1:
                params = [i, cte_array[0], j]
            else:
                params = [i, j, cte_array[0]]

            part_map.append(call_c(c_lib, fl_name, h0, params[0], params[1], params[2], precision, len(data[0])))

            # i_ind = np.where(params_array[rows_to_map[0]] == i)[0][0]
            # j_ind = np.where(params_array[rows_to_map[1]] == j)[0][0]
            # percent = 100 * i_ind/len(params_array[rows_to_map[0]])
            # percent += 100 * 1/len(params_array[rows_to_map[0]]) * (j_ind+1)/len(params_array[rows_to_map[1]])
            # print("Progresso: {:>6.2f}%\t".format(percent))

        map_array.append(np.array(part_map))
    map_array = np.array(map_array)

    head = "Mapa de chi2"
    np.savetxt("Map_chi2.csv", map_array, header=head, fmt="%f", delimiter=",")

    save_params = [["parametro", "min", "max"],
                   ["omega_m",  params_array[0][0], params_array[0][-1]],
                   ["omega_ee", params_array[1][0], params_array[1][-1]],
                   ["w",        params_array[2][0], params_array[2][-1]]]
    head = "Parâmetros do Mapa de chi2"
    np.savetxt("Param_map_chi2.csv", save_params, header=head, fmt="%s", delimiter=",")


def plot_map(data, params, min_chi=None, min_map=None, triangle=None, show=False,
             save=False, name=""):
    if params[0][0] == params[0][1]:
        im_range = [params[2][0], params[2][1], params[1][0], params[1][1]]
        xlab = "w"
        ylab = "\u03a9\u2091\u2091"
    elif params[1][0] == params[1][1]:
        im_range = [params[2][0], params[2][1], params[0][0], params[0][1]]
        xlab = "w"
        ylab = "\u03a9\u2098"
    else:
        im_range = [params[1][0], params[1][1], params[0][0], params[0][1]]
        xlab = "\u03a9\u2091\u2091"
        ylab = "\u03a9\u2098"

    plt.figure(figsize=(16, 9))

    plt.title("Mapeamento do \u03c7\u00b2", fontsize=18)
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)

    plt.imshow(data, origin="lower", extent=im_range, aspect="auto", interpolation="none", cmap="Spectral")
    plt.colorbar()

    if min_chi is not None:
        plt.scatter(min_chi[1], min_chi[0], c="black", label="Mínimo Nelder-Mead")
    if min_map is not None:
        plt.scatter(min_map[1], min_map[0], c="blue", label="Mínimo Mapeamento")
    if triangle is not None:
        triangle = np.append(triangle, [triangle[0]], axis=0).T
        plt.plot(triangle[0], triangle[1], "-", c="black", label="nelder-mead")

    plt.legend(loc="upper right", bbox_to_anchor=(1, 1))

    if save:
        plt.savefig("Mapeamento_chi2{}".format(name))
    if show:
        plt.show()
    plt.close()


def plot_movie(data, params, all_dots, save_mp4=False, show=False, name=""):
    if params[0][0] == params[0][1]:
        im_range = [params[2][0], params[2][1], params[1][0], params[1][1]]
        xlab = "w"
        ylab = "\u03a9\u2091\u2091"
    elif params[1][0] == params[1][1]:
        im_range = [params[2][0], params[2][1], params[0][0], params[0][1]]
        xlab = "w"
        ylab = "\u03a9\u2098"
    else:
        im_range = [params[1][0], params[1][1], params[0][0], params[0][1]]
        xlab = "\u03a9\u2091\u2091"
        ylab = "\u03a9\u2098"

    fig = plt.figure(figsize=(16, 9))

    plt.title("Mapeamento do \u03c7\u00b2", fontsize=18)
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)
    plt.xlim(im_range[0], im_range[1])
    plt.ylim(im_range[2], im_range[3])

    plt.imshow(data, origin="lower", extent=im_range, aspect="auto", interpolation="none", cmap="Spectral")
    plt.colorbar()

    triangle = np.append(all_dots[0], [all_dots[0][0]], axis=0).T
    mov, = plt.plot(triangle[1], triangle[0], "-", c="black", label="Evolução Nelder-Mead")

    def animate(j):
        tri = np.append(all_dots[j], [all_dots[j][0]], axis=0).T
        mov.set_data(tri[1], tri[0])

    ani = animation.FuncAnimation(fig, animate, interval=300, frames=len(all_dots)-1)
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1))

    if save_mp4:  # salva como mp4
        ani.save("evolution_params{}.mp4".format(name), writer="ffmpeg", fps=len(all_dots) / 8)
    if show:
        plt.show()
    plt.close()


def plot_mead(data, params, all_dots, save=False, show=False, name=""):
    if params[0][0] == params[0][1]:
        im_range = [params[2][0], params[2][1], params[1][0], params[1][1]]
        xlab = "w"
        ylab = "\u03a9\u2091\u2091"
    elif params[1][0] == params[1][1]:
        im_range = [params[2][0], params[2][1], params[0][0], params[0][1]]
        xlab = "w"
        ylab = "\u03a9\u2098"
    else:
        im_range = [params[1][0], params[1][1], params[0][0], params[0][1]]
        xlab = "\u03a9\u2091\u2091"
        ylab = "\u03a9\u2098"

    plt.figure(figsize=(16, 9))

    plt.title("Evolução do Algorítimo de Nelder-Mead a cada duas Iterações", fontsize=18)
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)

    plt.imshow(data, origin="lower", extent=im_range, aspect="auto", interpolation="none", cmap="Spectral")
    plt.colorbar()

    for i in range(len(all_dots)):
        if i % 2 == 0:
            triangle = np.append(all_dots[i], [all_dots[i][0]], axis=0).T
            plt.plot(triangle[1], triangle[0], "-")

    if save:
        plt.savefig("evolution_Nelder_Mead{}".format(name))
    if show:
        plt.show()
    plt.close()


def open_map(fl_data, fl_param):
    data = np.loadtxt(fl_data, skiprows=1, delimiter=",")
    params = np.loadtxt(fl_param, skiprows=2, delimiter=",", dtype="str").T[1:3].T.astype(np.float)

    return data, params


def config_c_call(c_name):
    # Função que configura os aspectos da biblioteca CTypes
    c_lib = ctypes.CDLL("./{}".format(c_name))  # abre o arquivo
    # Define os tipos de entrada da subrotina em questão
    c_lib.main_execution.argtypes = [ctypes.c_char_p,
                                     ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                     ctypes.c_double, ctypes.c_double,
                                     ctypes.c_int]
    # Define os tipos de saída da subrotina em questão
    c_lib.main_execution.restype = ctypes.c_double

    return c_lib


def call_c(c_lib, fl_name, h0, omega_m, omega_ee, w, precision, nrows):
    # Subrotina que executa a subrotina em questão no C
    chi2 = c_lib.main_execution(fl_name.encode("utf-8"),
                                h0, omega_m, omega_ee,
                                w, precision,
                                nrows)
    return chi2


def opt_nelder_mead(f, init, eps_desired=1E-5):
    init = np.array(init)
    dots = [init]

    # definicao dos parametros de nelder-mead
    alpha = 1    # reflexão     (alpha>0)
    beta = 1/2   # contração    (0<beta<1)
    gamma = 2    # expansão     (gamma>alpha)
    delta = 1/2  # encolhimento (0<delta<1)

    # array dos vetores unitarios
    e = np.zeros(len(init))

    # criação do triangulo inicial
    for i in range(len(init)):
        e[i] = e[i] + 1  # vetor unitario do eixo i
        h = 0.00025 if init[i-1] == 0 else 0.05  # determinação do step a ser dado
        dots.append(init+h*e)
        e[i] = e[i] - 1  # volta ao vetor e de zeros somente
    dots = np.array(dots)

    all_dots = np.array([dots])
    count = 0
    eps = 1
    sum_old = 0
    while eps >= eps_desired:
        x_lower = x_higher = x_higher2 = 0
        # acha o menor valor, o maior e o segundo maior
        for i in range(len(dots)):
            if f(dots[i]) < f(dots[x_lower]):
                x_lower = i
            if f(dots[i]) > f(dots[x_higher]):
                x_higher = i
        for i in range(len(dots)):
            if f(dots[x_higher2]) < f(dots[i]) < f(dots[x_higher]):
                x_higher2 = i

        # calcula o centroide do melhor lado
        c = (sum(dots) - dots[x_higher]) / (len(dots)-1)

        # reflexão
        def reflect():
            x_r = c + alpha * (c - dots[x_higher])
            if f(dots[x_lower]) <= f(x_r) < f(dots[x_higher2]):
                dots[x_higher] = x_r
                # goto stop
            elif f(x_r) < f(dots[x_lower]):
                expand(x_r)
            else:
                contract(x_r)
            return

        # expansão
        def expand(x_r):
            x_e = c + gamma * (x_r - c)
            if f(x_e) < f(x_r):
                dots[x_higher] = x_e
                # goto stop
            else:
                dots[x_higher] = x_r
                # goto stop
            return

        # contração
        def contract(x_r):
            if f(dots[x_higher2]) <= f(x_r) < f(dots[x_higher]):
                # contração externa
                x_0 = c + beta * (x_r - c)
                if f(x_0) <= f(x_r):
                    dots[x_higher] = x_0
                    # goto stop
                else:
                    shrink()
            elif f(dots[x_higher]) <= f(x_r):
                # contração interna
                x_i = c + beta * (dots[x_higher] - c)
                if f(x_i) < f(dots[x_higher]):
                    dots[x_higher] = x_i
                    # goto stop
                else:
                    shrink()

        # encolhimento
        def shrink():
            nonlocal dots

            bad_dots = np.delete(dots, x_lower, 0)
            new_dots = []

            for j in bad_dots:
                new_dots.append(dots[x_lower] + delta * (j - dots[x_lower]))
            new_dots = np.array(new_dots)

            dots = np.append(new_dots, [dots[x_lower]], 0)

        reflect()
        # guarda todos os pontos
        all_dots = np.append(all_dots, [dots], 0)

        # critério de parada por falha
        count += 1
        if count > 10000:
            break

        # critério de parada por convergencia dos valores
        sum_now = 0
        for i in dots:
            sum_now += np.abs(f(i)) / len(dots)
        eps = np.abs((sum_now - sum_old) / sum_now)
        sum_old = sum_now

    all_dots = np.array(all_dots)

    all_centroids = []
    for i in all_dots:
        all_centroids.append(sum(i) / len(dots))

    c = all_centroids[-1]

    return c, all_dots, all_centroids


def find_mins(h0, fl_name, c_name, params_array, param0, param1, initial_guess,
              remap=False, integ_precsicion=1E-5, nelder_precision=1E-5,
              plots=False, name=""):
    # Argumentos:
    # h0 -> constante de hubble;
    # fl_name -> nome do arquivo de dados
    # c_name -> nome da biblioteca em c;
    # params_array -> matriz dos parâmetros
    # param0 e param1 -> parametros variáveis na ordem (x,y)
    # initial_guess -> suspeita inicial de onde o minimo está
    #                  sempre respeitando a ordem (omM, omEE, w)
    #                  apenas tirando o valor constante
    # remap -> faz o mapeamento dos chi2, apenas necessário se não houver nenhum anterior
    # integ_precision -> precisão do valor no cálculo da integral
    # nelder_precision -> precisão do valor no algorítimo Nelder-Mead
    # plots -> plotar ou não os gráficos

    data = read_fl(fl_name)
    c_lib = config_c_call(c_name)

    def call_c_red(xy):
        if params_array[0][0] != params_array[0][-1]:
            omM = xy[0]
            if params_array[1][0] != params_array[1][-1]:
                omEE = xy[1]
                W = params_array[2][0]
            else:
                omEE = params_array[1][0]
                W = xy[1]
        else:
            omM = params_array[0][0]
            omEE = xy[0]
            W = xy[1]

        return call_c(c_lib, fl_name, h0, omM, omEE, W, integ_precsicion, len(data[0]))

    if remap:
        print("Mapeando o chi2;")
        map_chi(h0, data, params_array, c_lib, fl_name, precision=integ_precsicion)

    try:
        mapped, params = open_map("Map_chi2.csv", "Param_map_chi2.csv")
    except OSError:
        sys.exit("Arquivo Map_chi2.csv não encontrado, coloque a opção remap=True")

    print("Achando o mínimo do mapeamento;")
    min_map = np.where(mapped == np.min(mapped))
    min_map = [param1[min_map[0][0]], param0[min_map[1][0]]]

    print("Achando o mínimo do algorítimo de Nelder-Mead;")
    min_nel, evolution_dots, envolution_min = opt_nelder_mead(call_c_red,
                                                              initial_guess,
                                                              eps_desired=nelder_precision)

    print("Achando o mínimo do algorítimo de Nelder-Mead pelo SciPy;")
    min_sci = minimize(call_c_red, np.array(initial_guess), method='nelder-mead').x

    if plots:
        print("Plotando Resultados.")
        plot_map(mapped, params, min_chi=min_nel, min_map=min_map, save=True, name=name)
        plot_mead(mapped, params, evolution_dots, save=True, name=name)
        plot_movie(mapped, params, evolution_dots, save_mp4=True, name=name)

        print("Comparação dos resultados:\n"
              "Mapeamento: x={:.4f}, y={:.4f}\n"
              "Nelder-Mead: x={:.4f}, y={:.4f}\n"
              "Scipy Nelder-Mead: x={:.4f}, y={:.4f}\n"
              "".format(min_map[1], min_map[0], min_nel[1], min_nel[0], min_sci[1], min_sci[0]))

    results = {"Min_Map": min_map,
               "Min_Nelder": min_nel,
               "Min_SciPy": min_sci}

    return results


def main():
    fl_name = "fake_data.cat"
    c_name = "chi.so.1"
    h0 = 70

    # w constante
    omega_ee = np.linspace(0, 1, 50)
    omega_m = np.linspace(0, 1, 50)
    w = -np.ones(50)
    params_array = np.array([omega_m, omega_ee, w])

    initial_guess = [0.8, 0.2]

    mins_w = find_mins(h0, fl_name, c_name, params_array, omega_ee, omega_m,
                       initial_guess, remap=True, plots=True, name="_fake_wcte")

    # omega_m constante
    omega_ee = np.linspace(0, 1, 50)
    omega_m = 0.3 * np.ones(50)
    w = np.linspace(-2, 0, 50)
    params_array = np.array([omega_m, omega_ee, w])

    initial_guess = [0.5, -0.5]

    mins_omM = find_mins(h0, fl_name, c_name, params_array, w, omega_ee, initial_guess,
                         remap=True, plots=True, name="_fake_omMcte")

    # omega_ee constante
    omega_ee = 0.7 * np.ones(50)
    omega_m = np.linspace(0, 1, 50)
    w = np.linspace(-2, 0, 50)
    params_array = np.array([omega_m, omega_ee, w])

    initial_guess = [0.5, -0.5]

    mins_omEE = find_mins(h0, fl_name, c_name, params_array, w, omega_m, initial_guess,
                          remap=True, plots=True, name="_fake_omEEcte")

    # todos variaveis
    data = read_fl(fl_name)
    c_lib = config_c_call(c_name)

    def call_c_red_3(xyz):
        omM = xyz[0]
        omEE = xyz[1]
        W = xyz[2]
        return call_c(c_lib, fl_name, h0, omM, omEE, W, 1E-5, len(data[0]))

    initial_guess = [0.5, 0.5, -0.5]

    min_nel, evolution_dots, envolution_min = opt_nelder_mead(call_c_red_3, initial_guess)
    min_sci = minimize(call_c_red_3, np.array(initial_guess), method='nelder-mead').x

    mins_var = {"Min_Nelder": min_nel,
                "Min_SciPy": min_sci}

    print(mins_var)

    head = "param constante, método, valor x, valor y, valor z"
    text = ""

    for i in mins_w.keys():
        text += "    w=-1, {:<10}, {:>8.5f}, {:>8.5f},       -1\n" \
                "".format(i, mins_w[i][1], mins_w[i][0])
    for i in mins_omM.keys():
        text += " omM=0.3, {:<10}, {:>8.5f}, {:>8.5f},      0.3\n" \
                "".format(i, mins_omM[i][1], mins_omM[i][0])
    for i in mins_omEE.keys():
        text += "omEE=0.7, {:<10}, {:>8.5f}, {:>8.5f},      0.7\n" \
                "".format(i, mins_omEE[i][1], mins_omEE[i][0])
    for i in mins_var.keys():
        text += "    none, {:<10}, {:>8.5f}, {:>8.5f}, {:>8.5f}\n" \
                "".format(i, mins_var[i][0], mins_var[i][1], mins_var[i][2])

    np.savetxt("Minimos_fake_data.csv", [text], header=head, fmt="%s")


if __name__ == '__main__':
    main()