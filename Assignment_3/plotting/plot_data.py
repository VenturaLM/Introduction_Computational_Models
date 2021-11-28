import matplotlib.pyplot as plt
import pandas as pd
import click

#   Note: In case of experiment 2:
#       - Ctrl + h --> Substitute ['r'] for ['eta'].


@click.command()
@click.option('--file_name', '-f', default=None, required=True, help=u'Name of the file with data to plot.')
def main(file_name):
    if file_name == "":
        print("No data file selected.")
        exit(0)

    # Read data file.
    data = pd.read_excel(file_name)
    df = pd.DataFrame(data)

    r, mse_train, mse_test, crr_train, crr_test, time = readData(file_name)

    plotMSE(file_name, r, mse_train, mse_test)
    plotCRR(file_name, r, crr_train, crr_test)
    plotTime(file_name, r, time)


def readData(file_name):
    """
    Read data from a file for plotting.
    """
    data = pd.read_excel(file_name)
    df = pd.DataFrame(data)
    return df['r'], df['MSE_train'], df['MSE_test'], df['CRR_train'], df['CRR_test'], df['time']


def plotMSE(file_name, r, mse_train, mse_test):
    """
    Plot the MSE of the model.
    """
    plt.title(file_name)

    # Labels.
    plt.xlabel('r')
    plt.ylabel('MSE')

    # MSE plotting.
    plt.plot(r, mse_train)
    plt.plot(r, mse_test, color='blueviolet')

    # Set legend.
    plt.legend(['Training MSE', 'Test MSE'])

    # Plot chart.
    plt.show()


def plotCRR(file_name, r, crr_train, crr_test):
    """
    Plot the CRR of the model.
    """
    plt.title(file_name)

    # Labels.
    plt.xlabel('r')
    plt.ylabel('CRR')

    # CRR plotting.
    plt.plot(r, crr_train)
    plt.plot(r, crr_test, color='blueviolet')

    # Set legend.
    plt.legend(['Training CRR', 'Test CRR'])

    # Plot chart.
    plt.show()


def plotTime(file_name, r, time):
    """
    Plot the timing of the model.
    """
    plt.title(file_name)

    # Labels.
    plt.xlabel('r')
    plt.ylabel('time (s)')

    # CRR plotting.
    plt.plot(r, time)

    # Set legend.
    plt.legend(['Time'])

    # Plot chart.
    plt.show()


if __name__ == "__main__":
    main()
