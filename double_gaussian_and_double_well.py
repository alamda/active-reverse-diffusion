import numpy as np
import matplotlib.pyplot as plt


def compare_plots(a=None, b=None):
    xmin = -5
    xmax = 5
    num_points = 1000

    x_arr = np.linspace(xmin, xmax, num_points)

    # Double Gaussian

    mu_list = [-1.2, 1.2]
    sigma_list = [1.0, 1.0]
    pi_list = [1.0, 1.0]

    gauss_arr = np.zeros(num_points)

    for mu, sigma, pi in zip(mu_list, sigma_list, pi_list):
        gauss_arr += pi*np.exp(-1*(x_arr-mu)**2/(2*sigma**2))

    gauss_arr /= np.sum(gauss_arr)

    # Double Well

    # a = 0.05
    # b = -0.16

    double_well_arr = np.exp(-1*(a*x_arr**4 + b*x_arr**2))

    double_well_arr /= np.sum(double_well_arr)

    # Plotting

    fig, ax = plt.subplots()

    ax.scatter(x_arr, gauss_arr,
               label=r"$P \propto \sum_i \pi_i \exp \left( -\frac{(x-\mu_i)^2}{2\sigma_i^2} \right)$"
               "\t"
                     f"mu={mu_list}, sigma={sigma_list}, pi={pi_list}"
               )

    ax.scatter(x_arr, double_well_arr,
               label=r"$P \propto \exp \left( -(ax^4 + bx^2)\right)$"
               "\t"
                     f"a={a}, b={b}")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25))
    plt.tight_layout()

    png_fname = f"a{a}_b{b}.png"

    plt.savefig(png_fname)

    plt.close(fig)


def fit_double_gaussian():
    xmin = -5
    xmax = 5
    num_points = 1000

    x_arr = np.linspace(xmin, xmax, num_points)

    # Double Gaussian

    mu_list = [-1.2, 1.2]
    sigma_list = [1.0, 1.0]
    pi_list = [1.0, 1.0]

    gauss_arr = np.zeros(num_points)

    for mu, sigma, pi in zip(mu_list, sigma_list, pi_list):
        gauss_arr += pi*np.exp(-1*(x_arr-mu)**2/(2*sigma**2))

    gauss_arr /= np.sum(gauss_arr)

    poly_arr = np.polyfit(x_arr, np.log(gauss_arr), 4)

    title = ""

    fit_arr = np.zeros(num_points)

    for idx, coeff in enumerate(poly_arr):
        if idx > 0:
            title += "+"

        fit_arr += coeff*x_arr**idx
        title += f"{coeff}**x^{idx}"

    y_arr = np.exp(-1*fit_arr)
    y_arr /= np.sum(y_arr)

    fig, ax = plt.subplots()

    ax.scatter(x_arr, np.log(gauss_arr), label="original")
    ax.scatter(x_arr, fit_arr, label="fit")

    ax.set_title(title)

    ax.legend()

    plt.savefig("fit.png")

    plt.close(fig)


if __name__ == "__main__":

    # fit_double_gaussian()

    a_list = [0.034]
    b_list = [-0.09]

    for a in a_list:
        for b in b_list:
            compare_plots(a=a, b=b)
