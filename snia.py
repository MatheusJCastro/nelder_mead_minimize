###################################################
# Nelder-Mead Minimization (and a lot more)       #
# Matheus J. Castro                               #
# Version 4.5                                     #
# Last Modification: 06/11/2021 (month/day/year)  #
###################################################

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patches as ptc
from scipy.optimize import minimize
from scipy.integrate import quad
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib as mpl
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


def map_chi(h0, data, params_array, c_lib, fl_name, name="", precision=1E-10, prior=False):
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
            part_map.append(call_c(c_lib, fl_name, h0, params[0], params[1], params[2], precision, len(data[0]),
                                   prior=prior))

            i_ind = np.where(params_array[rows_to_map[0]] == i)[0][0]
            j_ind = np.where(params_array[rows_to_map[1]] == j)[0][0]
            if j_ind % 200 == 0:
                percent = 100 * i_ind / len(params_array[rows_to_map[0]])
                percent += 100 * 1 / len(params_array[rows_to_map[0]]) * (j_ind + 1) / len(params_array[rows_to_map[1]])
                print("Progresso: {:>6.2f}%\r".format(percent), end="")
        map_array.append(np.array(part_map))
    map_array = np.array(map_array)

    head = "Mapa de chi2"
    np.savetxt("all_csv_map/Map_chi2{}.csv".format(name), map_array, header=head, fmt="%f", delimiter=",")

    save_params = [["parametro", "min", "max"],
                   ["omega_m", params_array[0][0], params_array[0][-1]],
                   ["omega_ee", params_array[1][0], params_array[1][-1]],
                   ["w", params_array[2][0], params_array[2][-1]]]
    head = "Parâmetros do Mapa de chi2"
    np.savetxt("all_csv_map/Param_map_chi2{}.csv".format(name), save_params, header=head, fmt="%s", delimiter=",")


def map_chi_d(h0, data, omega_m, omega_ee, w, c_lib, fl_name, name="", precision=1E-10):
    map_array = []  # mapa do chi2
    for i in range(len(omega_m)):  # para cada linha que tem as variaveis
        part_map = []  # cada linha do mapa
        for j in w:
            # adiciona os dados das variaveis na linha
            part_map.append(call_c(c_lib, fl_name, h0, omega_m[i], omega_ee[i], j, precision, len(data[0])))

            j_ind = np.where(w == j)[0][0]
            if j_ind % 200 == 0:
                percent = 100 * i / len(omega_m)
                percent += 100 * 1 / len(omega_m) * (j_ind + 1) / len(w)
                print("Progresso: {:>6.2f}%\r".format(percent), end="")

        map_array.append(np.array(part_map))  # adiciona linha no mapa
    map_array = np.array(map_array)  # transforma mapa de lista para array

    head = "Mapa de chi2"
    np.savetxt("all_csv_map/Map_chi2{}.csv".format(name), map_array, header=head, fmt="%f",
               delimiter=",")  # salva mapa num arquivo

    # range de valores usados para construir o mapa de chi2
    save_params = [["parametro", "min", "max"],
                   ["omega_m", omega_m[0], omega_m[-1]],
                   ["omega_ee", omega_ee[0], omega_ee[-1]],
                   ["w", w[0], w[-1]]]
    head = "Parâmetros do Mapa de chi2"
    # salvando esses valores
    np.savetxt("all_csv_map/Param_map_chi2{}.csv".format(name), save_params, header=head, fmt="%s", delimiter=",")


def cov_elipses(cov):
    covx_square = cov[0][0]
    covy_square = cov[1][1]
    covxy = cov[0][1]
    covxy_square = cov[0][1] ** 2

    param_1 = (covx_square + covy_square) / 2
    param_sqrt = np.sqrt((covx_square - covy_square) ** 2 / 4 + covxy_square)

    a = 2 * np.sqrt(param_1 + param_sqrt)
    b = 2 * np.sqrt(param_1 - param_sqrt)

    theta = np.arctan(2 * covxy / (covx_square - covy_square)) / 2
    theta = theta * 180 / np.pi

    return a, b, theta


def plot_map(data, params, cov, cpnm, min_chi=None, min_map=None, triangle=None, show=False,
             save=False, name="", d=False):
    if params[0][0] == params[0][1]:
        im_range = [params[2][0], params[2][1], params[1][0], params[1][1]]
        xlab = "w"
        ylab = "\u03a9\u2091\u2091"
    elif params[1][0] == params[1][1] or d:
        im_range = [params[2][0], params[2][1], params[0][0], params[0][1]]
        xlab = "w"
        ylab = "\u03a9\u2098"
    else:
        im_range = [params[1][0], params[1][1], params[0][0], params[0][1]]
        xlab = "\u03a9\u2091\u2091"
        ylab = "\u03a9\u2098"

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    plt.title("Mapeamento do \u03c7\u00b2", fontsize=18)
    plt.xlabel(xlab, fontsize=18)
    plt.ylabel(ylab, fontsize=18)
    plt.xlim(im_range[0], im_range[1])
    plt.ylim(im_range[2], im_range[3])

    plt.imshow(data, origin="lower", extent=im_range, aspect="auto", interpolation="none",
               cmap=cpnm[0])

    cb = plt.colorbar(mpl.cm.ScalarMappable(cmap=cpnm[0], norm=cpnm[1]))
    cb.set_label(label=r"Intervalos de $\sigma$ - Mapeamento", fontsize=14)

    if min_chi is not None:
        plt.scatter(min_chi[1], min_chi[0], c="black", label="Mínimo Nelder-Mead")
    if min_map is not None:
        plt.scatter(min_map[1], min_map[0], c="blue", label="Mínimo Mapeamento")
    if triangle is not None:
        triangle = np.append(triangle, [triangle[0]], axis=0).T
        plt.plot(triangle[0], triangle[1], "-", c="black", label="nelder-mead")

    a, b, theta = cov_elipses(cov)
    alphas = [1.52, 2.48, 3.44]
    lines = ["-", "--", "-."]

    for i in range(len(alphas)):
        e = ptc.Ellipse((min_chi[1], min_chi[0]), alphas[i] * a, alphas[i] * b, theta, ls=lines[i], zorder=5,
                        fill=False, label=r"{}$\sigma$ - Matriz de Fisher".format(i + 1))
        ax.add_patch(e)

    plt.legend(loc="upper right", bbox_to_anchor=(1, 1), fontsize=14)

    if save:
        plt.savefig("all_mapping/mapping_chi2{}".format(name))
    if show:
        plt.show()
    plt.close()


def plot_movie(data, params, all_dots, cpnm, save_mp4=False, show=False, name="", d=False):
    if params[0][0] == params[0][1]:
        im_range = [params[2][0], params[2][1], params[1][0], params[1][1]]
        xlab = "w"
        ylab = "\u03a9\u2091\u2091"
    elif params[1][0] == params[1][1] or d:
        im_range = [params[2][0], params[2][1], params[0][0], params[0][1]]
        xlab = "w"
        ylab = "\u03a9\u2098"
    else:
        im_range = [params[1][0], params[1][1], params[0][0], params[0][1]]
        xlab = "\u03a9\u2091\u2091"
        ylab = "\u03a9\u2098"

    fig = plt.figure(figsize=(16, 9))

    plt.title("Evolução dos Simplex no Mapeamento do \u03c7\u00b2", fontsize=18)
    plt.xlabel(xlab, fontsize=18)
    plt.ylabel(ylab, fontsize=18)
    plt.xlim(im_range[0], im_range[1])
    plt.ylim(im_range[2], im_range[3])

    plt.imshow(data, origin="lower", extent=im_range, aspect="auto", interpolation="none",
               cmap=cpnm[0])

    cb = plt.colorbar(mpl.cm.ScalarMappable(cmap=cpnm[0], norm=cpnm[1]))
    cb.set_label(label=r"Intervalos de $\sigma$ - Mapeamento", fontsize=14)

    triangle = np.append(all_dots[0], [all_dots[0][0]], axis=0).T
    mov, = plt.plot(triangle[1], triangle[0], "-", c="black", label="Evolução Nelder-Mead")

    def animate(j):
        tri = np.append(all_dots[j], [all_dots[j][0]], axis=0).T
        mov.set_data(tri[1], tri[0])

    ani = animation.FuncAnimation(fig, animate, interval=300, frames=len(all_dots) - 1)
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1))

    if save_mp4:  # salva como mp4
        ani.save("all_movies/evolution_params{}.mp4".format(name), writer="ffmpeg", fps=len(all_dots) / 8)
    if show:
        plt.show()
    plt.close()


def plot_mead(data, params, all_dots, cpnm, save=False, show=False, name="", d=False):
    if params[0][0] == params[0][1]:
        im_range = [params[2][0], params[2][1], params[1][0], params[1][1]]
        xlab = "w"
        ylab = "\u03a9\u2091\u2091"
    elif params[1][0] == params[1][1] or d:
        im_range = [params[2][0], params[2][1], params[0][0], params[0][1]]
        xlab = "w"
        ylab = "\u03a9\u2098"
    else:
        im_range = [params[1][0], params[1][1], params[0][0], params[0][1]]
        xlab = "\u03a9\u2091\u2091"
        ylab = "\u03a9\u2098"

    plt.figure(figsize=(16, 9))

    plt.title("Evolução do Algorítimo de Nelder-Mead a cada duas Iterações", fontsize=18)
    plt.xlabel(xlab, fontsize=18)
    plt.ylabel(ylab, fontsize=18)

    plt.imshow(data, origin="lower", extent=im_range, aspect="auto", interpolation="none",
               cmap=cpnm[0])

    cb = plt.colorbar(mpl.cm.ScalarMappable(cmap=cpnm[0], norm=cpnm[1]))
    cb.set_label(label=r"Intervalos de $\sigma$ - Mapeamento", fontsize=14)

    for i in range(len(all_dots)):
        if i % 2 == 0:
            triangle = np.append(all_dots[i], [all_dots[i][0]], axis=0).T
            plt.plot(triangle[1], triangle[0], "-")

    if save:
        plt.savefig("all_evolution_simplex/evolution_Nelder_Mead{}".format(name))
    if show:
        plt.show()
    plt.close()


def all_plots(evolution_dots, mins, cov, name="", save=True, show=False, d=False):
    print("Plotando Resultados {}.".format(name[1:]))

    min_nel = mins["Min_Nelder"]
    min_map = mins["Min_Map"]

    colors = [(0, 0.5, 1), (0, 0.75, 1), (0, 1, 1)]
    cmap = LinearSegmentedColormap.from_list("rgb", colors, N=3)
    bounds = [0.000, 0.683, 0.954, 0.997]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cpnm = [cmap, norm]

    mapped, params = open_map("Map_chi2", "Param_map_chi2", name=name)

    min_data = np.min(mapped)
    for i in range(len(mapped)):
        for j in range(len(mapped[0])):
            if mapped[i][j] > 11.8 + min_data:
                mapped[i][j] = None
            elif mapped[i][j] > 6.17 + min_data:
                mapped[i][j] = 3
            elif mapped[i][j] > 2.3 + min_data:
                mapped[i][j] = 2
            else:
                mapped[i][j] = 1

    plot_map(mapped, params, cov, cpnm, min_chi=min_nel, min_map=min_map, save=save, show=show, name=name, d=d)
    plot_mead(mapped, params, evolution_dots, cpnm, save=save, show=show, name=name, d=d)
    plot_movie(mapped, params, evolution_dots, cpnm, save_mp4=save, show=show, name=name, d=d)


def open_map(fl_data, fl_param, name=""):
    data = np.loadtxt("all_csv_map/{}{}.csv".format(fl_data, name), skiprows=1, delimiter=",")
    params = np.loadtxt("all_csv_map/{}{}.csv".format(fl_param, name), skiprows=2,
                        delimiter=",", dtype="str").T[1:3].T.astype(np.float)

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


def call_c(c_lib, fl_name, h0, omega_m, omega_ee, w, precision, nrows, prior=False):
    # Subrotina que executa a subrotina em questão no C
    chi2 = c_lib.main_execution(fl_name.encode("utf-8"),
                                h0, omega_m, omega_ee,
                                w, precision,
                                nrows)
    if not prior:
        return chi2
    # se tiver prior omega_k = -0.06 +- 0.05 para os parametros
    else:
        return chi2 + ((1 - omega_m - omega_ee + 0.06) ** 2 / (0.05 ** 2))


def opt_nelder_mead(f, init, eps_desired=1E-5):
    init = np.array(init)
    dots = [init]

    # definicao dos parametros de nelder-mead
    alpha = 1  # reflexão     (alpha>0)
    beta = 1 / 2  # contração    (0<beta<1)
    gamma = 2  # expansão     (gamma>alpha)
    delta = 1 / 2  # encolhimento (0<delta<1)

    # array dos vetores unitarios
    e = np.zeros(len(init))

    # criação do triangulo inicial
    for i in range(len(init)):
        e[i] = e[i] + 1  # vetor unitario do eixo i
        h = 0.00025 if init[i - 1] == 0 else 0.05  # determinação do step a ser dado
        dots.append(init + h * e)
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
        c = (sum(dots) - dots[x_higher]) / (len(dots) - 1)

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
    all_centroids = np.array(all_centroids)

    c = all_centroids[-1]

    return c, all_dots, all_centroids


def find_uncert(cov, mins, name=""):
    a, b, theta = cov_elipses(cov)
    alphas = [1.52, 2.48, 3.44]

    lims = []
    save = ""
    meanxy = None

    for i in range(len(alphas)):
        xmax = mins[1] + alphas[i] * a * np.cos(theta * np.pi / 180) / 2
        ymax = mins[0] + alphas[i] * a * np.sin(theta * np.pi / 180) / 2

        xmin = mins[1] - alphas[i] * a * np.cos(theta * np.pi / 180) / 2
        ymin = mins[0] - alphas[i] * a * np.sin(theta * np.pi / 180) / 2

        # left, right, bottom, top
        lims.append(np.array([xmin, xmax, ymin, ymax]))

        mean_x = np.abs(np.mean([xmax - mins[1], mins[1] - xmin]))
        mean_y = np.abs(np.mean([ymax - mins[0], mins[0] - ymin]))
        save += "{}, {:.2e}, {:.2e}, {:.2e}, {:.2e}\n".format(i + 1, mins[1], mean_x, mins[0], mean_y)

        if i == 0:
            meanxy = [mean_y, mean_x]
    lims = np.array(lims)

    head = "sigma, x, sig_x, y, sig_y"
    np.savetxt("Minimo_Nelder_Incerteza{}.csv".format(name), [save], header=head, fmt="%s")

    return lims, meanxy


def find_mins(h0, fl_name, c_lib, params_array, param0, param1, initial_guess,
              remap=False, integ_precision=1E-5, nelder_precision=1E-5,
              prints=False, name="", d=False, prior=False):
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
        if d:
            omM = xy[0]
            omEE = 1 - xy[0]
            W = xy[1]

        return call_c(c_lib, fl_name, h0, omM, omEE, W, integ_precision, len(data[0]), prior=prior)

    if remap:
        print("Mapeando o chi2;")
        if not d:
            map_chi(h0, data, params_array, c_lib, fl_name, precision=integ_precision, name=name, prior=prior)
        else:
            map_chi_d(h0, data, params_array[0], params_array[1], params_array[2],
                      c_lib, fl_name, precision=integ_precision, name=name)

    try:
        mapped, params = open_map("Map_chi2", "Param_map_chi2", name=name)
    except OSError:
        sys.exit("Arquivo Map_chi2{}.csv não encontrado, coloque a opção remap=True".format(name))

    print("Achando o mínimo do mapeamento;")
    min_map = np.where(mapped == np.min(mapped))
    min_map = [param1[min_map[0][0]], param0[min_map[1][0]]]

    print("Achando o mínimo do algorítimo de Nelder-Mead;")
    min_nel, evolution_dots, evolution_min = opt_nelder_mead(call_c_red,
                                                             initial_guess,
                                                             eps_desired=nelder_precision)
    evol = evolution_min[2:].T
    cov = np.cov(np.stack((evol[1], evol[0]), axis=0))

    print("Achando o mínimo do algorítimo de Nelder-Mead pelo SciPy;")
    min_sci = minimize(call_c_red, np.array(initial_guess), method='Nelder-Mead').x

    print("Calculando Incertezas;")
    lims, meanxy = find_uncert(cov, min_nel, name=name)

    if prints:
        print("Comparação dos resultados:\n"
              "Mapeamento: x={:.4f}, y={:.4f}\n"
              "Nelder-Mead: x={:.4e}+/-{:.4e}, y={:.4e}+/-{:.4e}\n"
              "Scipy Nelder-Mead: x={:.4f}, y={:.4f}\n"
              "".format(min_map[1], min_map[0],
                        min_nel[1], meanxy[1],
                        min_nel[0], meanxy[0],
                        min_sci[1], min_sci[0]))

    results = {"Min_Map": min_map,
               "Min_SciPy": min_sci,
               "Min_Nelder": min_nel}

    return results, evolution_dots, cov


def main():
    fl_name = "fake_data.cat"
    c_name = "chi.so.1"
    h0 = 70
    map_len = 1000

    c_dll = config_c_call(c_name)

    names = []
    all_mins = []
    all_dots = []
    all_covs = []

    # w constante
    print("w constante")
    names.append("_fake_wcte")
    omega_ee = np.linspace(0, 1, map_len)
    omega_m = np.linspace(0, 1, map_len)
    w = -np.ones(map_len)
    params_array = np.array([omega_m, omega_ee, w])

    initial_guess = [0.4, 0.4]  # omega_m, omega_ee

    mins_w, evol_w, cov_w, = find_mins(h0, fl_name, c_dll, params_array, omega_ee, omega_m,
                                       initial_guess, remap=False, prints=False, name=names[0])
    all_mins.append(mins_w)
    all_dots.append(evol_w)
    all_covs.append(cov_w)

    # omega_m constante
    print("omega_m constante")
    names.append("_fake_omMcte")
    omega_ee = np.linspace(0, 1, map_len)
    omega_m = 0.3 * np.ones(map_len)
    w = np.linspace(-2, 0, map_len)
    params_array = np.array([omega_m, omega_ee, w])

    initial_guess = [0.4, -1.3]  # omega_ee, w

    mins_omM, evol_omM, cov_omM = find_mins(h0, fl_name, c_dll, params_array, w, omega_ee,
                                            initial_guess, remap=False, prints=False, name=names[1])
    all_mins.append(mins_omM)
    all_dots.append(evol_omM)
    all_covs.append(cov_omM)

    # omega_ee constante
    print("omega_ee constante")
    names.append("_fake_omEEcte")
    omega_ee = 0.7 * np.ones(map_len)
    omega_m = np.linspace(0, 1, map_len)
    w = np.linspace(-2, 0, map_len)
    params_array = np.array([omega_m, omega_ee, w])

    initial_guess = [0.35, -0.75]  # omega_m, w

    mins_omEE, evol_omEE, cov_omEE = find_mins(h0, fl_name, c_dll, params_array, w, omega_m,
                                               initial_guess, remap=False, prints=False, name=names[2])
    all_mins.append(mins_omEE)
    all_dots.append(evol_omEE)
    all_covs.append(cov_omEE)

    # todos variaveis
    print("Todas variáveis")
    data = read_fl(fl_name)

    def call_c_red_3(xyz):
        omM = xyz[0]
        omEE = xyz[1]
        W = xyz[2]
        return call_c(c_dll, fl_name, h0, omM, omEE, W, 1E-5, len(data[0]))

    initial_guess = [0.5, 0.5, -0.5]  # omega_m, omega_ee, w

    min_nel, evolution_dots, envolution_min = opt_nelder_mead(call_c_red_3, initial_guess)
    min_sci = minimize(call_c_red_3, np.array(initial_guess), method='nelder-mead').x

    mins_var = {"Min_Nelder": min_nel,
                "Min_SciPy": min_sci}

    # salvando os mínimos
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

    np.savetxt("results_files/Minimos_fake_data.csv", [text], header=head, fmt="%s")

    for i in range(len(all_mins)):
        all_plots(all_dots[i], all_mins[i], all_covs[i], name=names[i])


def item_a():
    fl_name = "SN_2021.cat"
    c_name = "chi.so.1"
    h0 = 70  # constante de Hubble

    data = read_fl(fl_name)  # Leitura dos dados de SN_2021.cat
    c_dll = config_c_call(c_name)

    # Calculando χ² para 1 dimensão
    def call_c_red_1(x):
        omEE = x[0]  # Ωee
        omM = 1 - omEE  # Ωm
        W = -1  # parâmetro da equação de estado para Universo dominado por energia escura

        return call_c(c_dll, fl_name, h0, omM, omEE, W, 1E-5, len(data[0]))  # cálculo de χ² com a função em C

    # Função densidade de probabilidade P(Ωee)
    def P(omega_ee):

        # Se Ωee for do tipo array:
        if type(np.array([])) == type(omega_ee):
            soma = 0
            for o in omega_ee:
                # soma os P(Ωee) = exp(-χ²)
                soma += np.exp(-call_c_red_1([o]))
            return [soma]
        # Se não, retorna direto P(Ωee) = exp(-χ²)
        else:
            return np.exp(-call_c_red_1([omega_ee]))

    # método da caçada para encontrar os Ωee dos intervalos de confiança de χ²
    def searching(f, lims, y, eps_desired=1E-3):

        invert = False
        if f([lims[1]]) > f([lims[0]]):
            invert = True

        eps = 1
        x = lims[0] + (lims[1] - lims[0]) / 2

        while eps > eps_desired:
            x_old = x

            if f([x]) > y:
                lims = [lims[0], x] if invert else [x, lims[1]]
            else:
                lims = [x, lims[1]] if invert else [lims[0], x]

            x = lims[0] + (lims[1] - lims[0]) / 2
            eps = np.abs((x - x_old) / x)

        return x

    Omega_ees = np.linspace(0, 1, 1000)  # 100 valores de Ωee igualmente espaçados entre 0 e 1
    chi2 = []  # χ²

    # calculando χ² variando-se Ωee:
    for i in Omega_ees:
        chi2.append(call_c_red_1([i]))

    chi2 = np.array(chi2)  # transformando chi2 em np.array

    initial_guess = [0.5]  # estimativa inicial de Ωee

    min_nel, evolution_dots, evolution_min = opt_nelder_mead(call_c_red_1, initial_guess)  # cálculo do mínimo de χ²

    del_chi = [1, 4, 9]  # Δχ² = [1, 4, 9]: intervalos de confiança
    err_chi = []  # variável que armazena os erros em χ²

    for i in range(len(del_chi)):
        chi = call_c_red_1(min_nel) + del_chi[i]  # calcula χ² de cada intervalo de confiança
        xl = searching(call_c_red_1, [0, min_nel[0]], chi)  # intervalo à esquerda do mínimo χ²
        xr = searching(call_c_red_1, [min_nel[0], 1], chi)  # intervalo à direita do mínimo χ²
        err_chi.append(np.array([i + 1, xl, xr]))  # armazena erro em err_chi

    err_chi = np.array(err_chi)  # transformando err_chi em array

    # probabilidade desses dados indicarem que a densidade
    # da energia escura é maior do que 0.5 :
    prob_num1 = integral_calc(P, [0.5, 1], eps_desired=1E-3)[0]  # numerador
    prob_den1 = integral_calc(P, [0, 1], eps_desired=1E-3)[0]  # denominador

    prob_cumulativa1 = prob_num1 / prob_den1  # probabilidade P(Ωee > 0.5)

    # comparando com o resultado das integrais com o módulo scipy
    prob_num2 = quad(P, 0.5, 1)[0]  # numerador
    prob_den2 = quad(P, 0, 1)[0]  # denominador

    prob_cumulativa2 = prob_num2 / prob_den2  # probabilidade P(Ωee > 0.5)

    # normalização da P(omega_ee)
    p = np.exp(-chi2) / prob_den1  # FDP de Ωee

    lines = ['-', '--', '-.']  # tipos das linhas dos intervalos de Δχ²
    color = ['red', 'green', 'magenta']  # cores das linhas dos intervalos de Δχ²
    plt.figure(figsize=(16, 9))  # tamanho da figura
    plt.title(r"Mapeamento de $\chi^2$ em $\Omega_{ee}$", fontsize=18)  # título do gráfico
    plt.xlabel(r"$\Omega_{ee}$", fontsize=18)  # nome do eixo x
    plt.ylabel(r"$\chi^2$", fontsize=18)  # nome do eixo y
    plt.xlim(min(Omega_ees), max(Omega_ees))  # limites do eixo x entre 0 e 1.
    plt.xticks(np.linspace(min(Omega_ees), max(Omega_ees), 11))

    plt.plot(Omega_ees, chi2, label=r"Curva de $\chi^2$", c='blue')  # plotando curva de χ² x Ωee
    plt.scatter(min_nel[0], min(chi2), label=r'Min. $\chi^2$', c='black', zorder=10)  # ponto do mínimo χ²

    # linhas dos intervalos de confiança
    for i in range(len(err_chi)):
        plt.axvline(err_chi[i][1], ymax=0.2, ls=lines[i], c=color[i], label=r"{:.0f}$\sigma$".format(err_chi[i][0]))
        plt.axvline(err_chi[i][2], ymax=0.2, ls=lines[i], c=color[i])

    plt.legend()  # legendas
    plt.grid()  # grade

    plt.savefig("all_mapping/mapping_chi2_itema")  # salvando a imagem da curva
    plt.close()

    lines = ['-', '--', '-.']  # tipos das linhas dos intervalos de Δχ²
    color = ['red', 'green', 'magenta']  # cores das linhas dos intervalos de Δχ²
    plt.figure(figsize=(16, 9))  # tamanho da figura
    plt.title(r"Probabilidades $P(\Omega_{ee}) \propto exp(-\chi^2)$", fontsize=18)  # título do gráfico
    plt.xlabel(r"$\Omega_{ee}$", fontsize=18)  # nome do eixo x
    plt.ylabel(r"$P(\Omega_{ee})$", fontsize=18)  # nome do eixo y
    plt.xlim(min(Omega_ees), max(Omega_ees))  # limites do eixo x entre 0 e 1.
    plt.xticks(np.linspace(min(Omega_ees), max(Omega_ees), 11))

    plt.plot(Omega_ees, p, label=r"Curva de FDP $P(\Omega_{ee})$", c='blue')  # plotando curva de P(Ωee) x Ωee
    plt.scatter(min_nel[0], max(p), label=r'Max. verossimilhança', c='black', zorder=10)  # ponto de máximo de P(Ωee)

    # linhas dos intervalos de confiança
    for i in range(len(err_chi)):
        plt.axvline(err_chi[i][1], ymax=0.9, ls=lines[i], c=color[i], label=r"{:.0f}$\sigma$".format(err_chi[i][0]))
        plt.axvline(err_chi[i][2], ymax=0.9, ls=lines[i], c=color[i])

    plt.legend()  # legendas
    plt.grid()  # grade

    plt.savefig("all_mapping/fdp_itema")  # salvando a imagem da curva
    plt.close()

    head = 'sigma, x, sig_xl, sig_xr'  # header do arquivo que armazenará os dados
    save = ''
    for i in err_chi:
        save += '{:.0f}, {:.2e}, {:.2e}, {:.2e}\n'.format(i[0], min_nel[0], min_nel[0] - i[1], i[2] - min_nel[0])

    save += "prob_cumulativa, {:.4f},,\n".format(prob_cumulativa1)
    save += "prob_cumulativa_scipy, {:.4f},,\n".format(prob_cumulativa2)

    np.savetxt("results_files/Minimo_Nelder_Incerteza_itema.csv", [save], header=head, fmt="%s")


def item_b():
    fl_name = "SN_2021.cat"
    c_name = "chi.so.1"
    h0 = 70
    map_len = 1000

    c_dll = config_c_call(c_name)

    name_w = "_itemb"
    omega_ee = np.linspace(0, 1, map_len)
    omega_m = np.linspace(0, 1, map_len)
    w = -np.ones(map_len)
    params_array = np.array([omega_m, omega_ee, w])

    initial_guess = [0.5, 0.3]  # omega_m, omega_ee

    mins_w, evol_w, cov_w = find_mins(h0, fl_name, c_dll, params_array, omega_ee, omega_m,
                                      initial_guess, remap=False, prints=False, name=name_w, prior=False)

    all_plots(evol_w, mins_w, cov_w, name=name_w, save=True)


def item_c():
    fl_name = "SN_2021.cat"
    c_name = "chi.so.1"
    h0 = 70
    map_len = 1000

    c_dll = config_c_call(c_name)

    name_w = "_itemc"
    omega_ee = np.linspace(0, 1, map_len)
    omega_m = np.linspace(0, 1, map_len)
    w = -np.ones(map_len)  # w=-1
    params_array = np.array([omega_m, omega_ee, w])

    initial_guess = [0.6, 0.3]  # omega_m, omega_ee

    mins_w, evol_w, cov_w = find_mins(h0, fl_name, c_dll, params_array, omega_ee, omega_m,
                                      initial_guess, remap=False, prints=False, name=name_w, prior=True)

    all_plots(evol_w, mins_w, cov_w, name=name_w, save=True)


def item_d():
    fl_name = "SN_2021.cat"
    c_name = "chi.so.1"
    h0 = 70
    map_len = 1000

    c_dll = config_c_call(c_name)

    name_omEE = "_itemd"
    omega_m = np.linspace(0, 1, map_len)
    omega_ee = 1 - omega_m
    w = np.linspace(-2, 0, map_len)
    params_array = np.array([omega_m, omega_ee, w])

    initial_guess = [0.4, -0.5]  # omega_m, w

    mins_w, evol_w, cov_w = find_mins(h0, fl_name, c_dll, params_array, w, omega_m,
                                      initial_guess, remap=False, prints=False, name=name_omEE,
                                      d=True)

    all_plots(evol_w, mins_w, cov_w, name=name_omEE, save=True, d=True)


def mult_fit():
    fls_names = ["SN_2021.cat", "fake_data.cat"]
    c_name = "chi.so.1"
    h0 = 70

    doublarray = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C')

    c_dll = config_c_call(c_name)
    c_dll.lumin_dist_mod_func.argtypes = [ctypes.c_double, ctypes.c_double,
                                          doublarray, ctypes.c_double]
    c_dll.lumin_dist_mod_func.restype = ctypes.c_double

    for fl_name in fls_names:
        data = read_fl(fl_name)

        def call_c_red_3(xyz):
            omM = xyz[0]
            omEE = xyz[1]
            W = xyz[2]
            return call_c(c_dll, fl_name, h0, omM, omEE, W, 1E-5, len(data[0]))

        initial_guess = np.array([0.5, 0.5, -0.5])  # omega_m, omega_ee, w

        min_nel, evolution_dots, envolution_min = opt_nelder_mead(call_c_red_3, initial_guess)

        modist = []
        x = np.linspace(min(data[0]) * 0.8, max(data[0]) * 1.05, 1000)
        for i in x:
            mu = c_dll.lumin_dist_mod_func(h0, i, min_nel, 1E-3)
            modist.append(mu)

        plt.figure(figsize=(16, 9))

        name = "Ajuste Simultâneo para os dados \"{}\"\n".format(fl_name) + \
               "\u03a9\u2098={:.4f}, \u03a9\u2091\u2091={:.4f}, w={:.4f}".format(min_nel[0], min_nel[1], min_nel[2])

        plt.title(name, fontsize=20)
        plt.xlabel(r"Redshift $z$", fontsize=20)
        plt.ylabel(r"Módulo de Distância $\mu$", fontsize=20)

        plt.errorbar(data[0], data[1], yerr=data[2], fmt="o", zorder=1, markersize=4,
                     ecolor="red", color="blue", label="Dados {}".format(fl_name))
        plt.plot(x, modist, "-", c="black", zorder=2, linewidth=3, label="Curva de Ajuste")

        plt.legend()
        plt.grid()

        plt.savefig("fit_all_{}".format(fl_name[:-4]))
        plt.close()


if __name__ == '__main__':
    # main()
    # item_a()
    # item_b()
    # item_c()
    # item_d()
    mult_fit()

