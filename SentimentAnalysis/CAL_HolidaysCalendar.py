from Code.GlobalParams import *
import itertools
import datetime as dt

years = [[2012], [2013], [2014], [2015], [2016], [2017], [2018]]
dates = [
    [[1, 2], [1, 16], [2, 20], [4, 6], [5, 28], [7, 4], [9, 3], [11, 22], [12, 25]],
    [[1, 1], [1, 21], [2, 18], [3, 29], [5, 27], [7, 4], [9, 2], [11, 28], [12, 25]],
    [[1, 1], [1, 20], [2, 17], [4, 18], [5, 26], [7, 4], [9, 1], [11, 27], [12, 25]],
    [[1, 1], [1, 19], [2, 16], [4, 3], [5, 25], [7, 4], [9, 7], [11, 26], [12, 25]],
    [[1, 1], [1, 18], [2, 15], [3, 25], [5, 30], [7, 4], [9, 5], [11, 24], [12, 25]],
    [[1, 2], [1, 16], [2, 20], [4, 14], [5, 29], [7, 4], [9, 4], [11, 23], [12, 25]],
    [[1, 1], [1, 15], [2, 19], [3, 30], [5, 28], [7, 4], [9, 3], [11, 22], [12, 25]]]

def generate_holidays():
    dates_combined = [list(itertools.product(year, date)) for year, date in zip(years, dates)]
    holidays = list()
    for ele in dates_combined:
        holiday = [dt.date(year=t[0], month=t[1][0], day=t[1][1]) for t in ele]
        holidays.extend(holiday)
    return holidays