import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import os

plt.rc("text", usetex=False)


def fancy_plots_2():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = 3.0 / 3.0
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {
        "backend": "ps",
        "axes.labelsize": text_size,
        "legend.fontsize": tick_size,
        "legend.handlelength": 2.5,
        "legend.borderaxespad": 0,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "font.family": "serif",
        "font.size": text_size,
        # Times, Palatino, New Century Schoolbook,
        # Bookman, Computer Modern Roman
        # 'font.serif': ['Times'],
        "ps.usedistiller": "xpdf",
        "text.usetex": True,
        "figure.figsize": fig_size,
        # include here any neede package for latex
        "text.latex.preamble": [
            r"\usepackage{amsmath}",
        ],
    }
    plt.rc(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(
        left=0.13, right=0.98, top=0.97, bottom=0.13, hspace=0.05, wspace=0.02
    )
    plt.ioff()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    return fig, ax1, ax2


def plot_states(fig11, ax11, ax21, x, xd, t, name, path):
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))

    ax11.set_xticklabels([])
    (state_1_e,) = ax11.plot(
        t[0 : t.shape[0]], x[0 : t.shape[0]], color="#C43C29", lw=1.0, ls="-"
    )

    (state_2_e,) = ax21.plot(
        t[0 : t.shape[0]], xd[0 : t.shape[0]], color="#3FB454", lw=1.0, ls="-"
    )
    ax11.set_ylabel(r"$Loss$", rotation="vertical")
    ax11.legend(
        [state_1_e],
        [r"$training$"],
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=False,
        ncol=2,
        borderpad=0.5,
        labelspacing=0.5,
        handlelength=3,
        handletextpad=0.1,
        borderaxespad=0.3,
        columnspacing=2,
    )
    ax11.grid(color="#949494", linestyle="-.", linewidth=0.5)

    ax21.set_ylabel(r"$Accuracy$", rotation="vertical")
    ax21.legend(
        [state_2_e],
        [r"$training$"],
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=False,
        ncol=2,
        borderpad=0.5,
        labelspacing=0.5,
        handlelength=3,
        handletextpad=0.1,
        borderaxespad=0.3,
        columnspacing=2,
    )
    ax21.grid(color="#949494", linestyle="-.", linewidth=0.5)
    ax21.set_xlabel(r"$[Epochs]$", labelpad=5)

    png_file_path = os.path.join(path, name + ".pdf")
    fig11.savefig(png_file_path)
    return None


def plot_states_full(fig11, ax11, ax21, x, xd, t, name, path):
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))

    ax11.set_xticklabels([])
    (state_1_test,) = ax11.plot(
        t[0 : t.shape[0]], x[0, 0 : t.shape[0]], color="#C43C29", lw=1.0, ls="-"
    )

    (state_1_tranning,) = ax11.plot(
        t[0 : t.shape[0]], x[1, 0 : t.shape[0]], color="#C43C29", lw=1.0, ls="--"
    )

    (state_2_test,) = ax21.plot(
        t[0 : t.shape[0]], xd[0, 0 : t.shape[0]], color="#3FB454", lw=1.0, ls="-"
    )

    (state_2_tranning,) = ax21.plot(
        t[0 : t.shape[0]], xd[1, 0 : t.shape[0]], color="#3FB454", lw=1.0, ls="--"
    )
    ax11.set_ylabel(r"$Loss$", rotation="vertical")
    ax11.legend(
        [state_1_test, state_1_tranning],
        [r"$test$", r"$training$"],
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=False,
        ncol=2,
        borderpad=0.5,
        labelspacing=0.5,
        handlelength=3,
        handletextpad=0.1,
        borderaxespad=0.3,
        columnspacing=2,
    )
    ax11.grid(color="#949494", linestyle="-.", linewidth=0.5)

    ax21.set_ylabel(r"$Accuracy$", rotation="vertical")
    ax21.legend(
        [state_2_test, state_2_tranning],
        [r"$test$", r"$training$"],
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=False,
        ncol=2,
        borderpad=0.5,
        labelspacing=0.5,
        handlelength=3,
        handletextpad=0.1,
        borderaxespad=0.3,
        columnspacing=2,
    )
    ax21.grid(color="#949494", linestyle="-.", linewidth=0.5)
    ax21.set_xlabel(r"$[Epochs]$", labelpad=5)

    png_file_path = os.path.join(path, name + ".pdf")
    fig11.savefig(png_file_path)
    return None
