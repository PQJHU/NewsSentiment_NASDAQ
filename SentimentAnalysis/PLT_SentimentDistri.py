from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import numpy as np
from Code.GlobalParams import *

class PlotKernelDensityEstimator:
    """
    A object for plotting a series of KDE with given bandwidths and kernel functions
    """

    def __init__(self, data_points, x_grid_len=None):

        # delete -inf, inf and nan numbers
        data_points = list(filter(lambda x: x != -np.inf, data_points))

        if isinstance(data_points, list):
            data_points = np.asarray(data_points)

        # &(data_points != np.inf)&(data_points != np.nan)

        self.data_points = data_points

        # Default parameters
        self.kernel = 'epanechnikov'
        if x_grid_len is None:
            self.x_grid_len = int(len(data_points)/1)
        else:
            self.x_grid_len = x_grid_len
        self.x_grid = np.linspace(min(data_points), max(data_points), self.x_grid_len)
        # self.x_plot = np.linspace(0, 1, 1000)
        # self.file_name = file_name

    def bandwidth_search(self, method):
        # x_grid = np.linspace(min(self.data_points), max(self.data_points), int(len(self.data_points)/10))
        print('Searching Optimal Bandwidth...')
        if method == 'gridsearch':
            grid = GridSearchCV(KernelDensity(),
                                {
                                    'bandwidth': self.x_grid,
                                },
                                cv=5)
            grid.fit(self.data_points.reshape(-1, 1))
            self.band_width = grid.best_params_['bandwidth']
        elif method == 'silverman':
            std = self.data_points.std()
            n = len(self.data_points)
            self.band_width = 1.06 * std * np.power(n, -1 / 5)
        return self.band_width

    def pdf_calcualtion(self, **kwargs):
        if 'bandwidth' in kwargs:
            self.bandwidth = kwargs['bandwidth']
        else:
            self.bandwidth = self.bandwidth_search(kwargs['method'])
            print(f"Bandwidth search method: {kwargs['method']}")
        kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(X=self.data_points.reshape(-1, 1))
        self.pdf = np.exp(kde.score_samples(self.x_grid.reshape(-1, 1)))
        # self.log_densities[f'{bandwidth}'] = log_dens
        return self.pdf

    def plot_curve_hist_kde(self, bin_num=None, hist_density=True, bandwidth=None, method='silverman'):
        if bandwidth is None:
            self.pdf_calcualtion(method=method)
        else:
            self.pdf_calcualtion(bandwidth=bandwidth)
        fig = plt.figure(figsize=(15, 7))
        plt.hist(self.data_points, bins=bin_num, density=hist_density)
        plt.plot(self.x_grid, self.pdf, '-')
        plt.title(f'Kernel Estimation')
        return fig


sentiment_all_workingdays = pd.read_csv(outdata_path + f'sentiment/Sentiment_all_Workingdays_Agg{article_round_time}.csv', index_col='Time', parse_dates=True)
sentiment_all_holidays = pd.read_csv(outdata_path + f'sentiment/Sentiment_all_Holidays_Agg{article_round_time}.csv', index_col='Time', parse_dates=True)
sentiment_all_weekends = pd.read_csv(outdata_path + f'sentiment/Sentiment_all_Weekends_Agg{article_round_time}.csv', index_col='Time', parse_dates=True)
sentiment_aapl_daily = pd.read_csv(outdata_path + f'sentiment/Sentiment_AAPL_None_Agg{article_round_time}.csv', index_col='Time', parse_dates=True)
sentiment_amzn_daily = pd.read_csv(outdata_path + f'sentiment/Sentiment_AMZN_None_Agg{article_round_time}.csv', index_col='Time', parse_dates=True)



kernel_all_working = PlotKernelDensityEstimator(sentiment_all_workingdays['Sentiment'].values, x_grid_len=100)
pdfs_all_working = kernel_all_working.pdf_calcualtion(bandwidth=0.05)
xgrid_all_working = kernel_all_working.x_grid

kernel_aapl_allday = PlotKernelDensityEstimator(sentiment_aapl_daily['Sentiment'].values, x_grid_len=100)
pdfs_aapl_allday = kernel_aapl_allday.pdf_calcualtion(bandwidth=0.05)
xgrid_aapl_allday = kernel_aapl_allday.x_grid

kernel_amzn_allday = PlotKernelDensityEstimator(sentiment_amzn_daily['Sentiment'].values, x_grid_len=100)
pdfs_amzn_allday = kernel_amzn_allday.pdf_calcualtion(bandwidth=0.05)
xgrid_amzn_allday = kernel_amzn_allday.x_grid

plt.figure(figsize=(15, 7))
plt.plot(xgrid_all_working, pdfs_all_working, color='b', linestyle='-', marker='*', label='Market Workingday')
# plt.scatter(x=xgrid_all_working, y=pdfs_all_working, s=13, c='b', marker='*')
plt.plot(xgrid_aapl_allday, pdfs_aapl_allday, color='r', linestyle='-', marker='*', label='AAPL All-days')
plt.plot(xgrid_amzn_allday, pdfs_amzn_allday, color='g', linestyle='-', marker='*', label='AMZN All-days')
# plt.scatter(x=xgrid_aapl_allday, y=pdfs_aapl_allday, s=13, c='r', marker='*')
plt.xticks(fontsize=14)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.savefig(outplot_path + 'kde_sentiment_plot_market_appl_amzn.png', dpi=300)

kernel_all_holidays = PlotKernelDensityEstimator(sentiment_all_holidays['Sentiment'].values, x_grid_len=100)
pdfs_all_holidays = kernel_all_holidays.pdf_calcualtion(bandwidth=0.05)
xgrid_all_holidays = kernel_all_holidays.x_grid

kernel_all_weekends = PlotKernelDensityEstimator(sentiment_all_weekends['Sentiment'].values, x_grid_len=100)
pdfs_all_weekends = kernel_all_weekends.pdf_calcualtion(bandwidth=0.05)
xgrid_all_weekends = kernel_all_weekends.x_grid

plt.figure(figsize=(15, 7))
plt.plot(xgrid_all_working, pdfs_all_working, color='b', linestyle='-', marker='*', label='Market Workingdays')
plt.plot(xgrid_all_holidays, pdfs_all_holidays, color='r', linestyle='-', marker='*', label='Market Holidays')
plt.plot(xgrid_all_weekends, pdfs_all_weekends, color='g', linestyle='-', marker='*', label='Market weekends')
plt.xticks(fontsize=14)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.savefig(outplot_path + 'kde_sentiment_plot_market_different.png', dpi=300)


