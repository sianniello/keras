from pandas import read_csv

dataset = read_csv(
    filepath_or_buffer='dataset/globalterrorismdb_0617dist.csv',
    sep=',',
    header=0,
    encoding='ISO-8859-1',
    usecols=['iyear','imonth','iday','country', 'country_txt']
)


print(dataset.tail())
