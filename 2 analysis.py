import pandas as pd
import matplotlib.pyplot as plt
import re

weatherall=pd.read_csv("weather.csv")
weatherall

weatherall['Weather']=weatherall['Weather'].str.replace("转","~")
weatherall

for i in range(len(weatherall['Weather'])):
    if '~' in weatherall['Weather'][i]:
        weatherall['Weather'][i]=weatherall['Weather'][i][:weatherall['Weather'][i].find('~')]
weatherall

weatherall['Date'] = weatherall['Date'].astype('str')
weatherall['Date']=pd.to_datetime(weatherall['Date'],format='%Y-%m-%d')
weatherall['Year']=weatherall['Date'].dt.year
weatherall.head()

weatherall = weatherall.set_index(weatherall['Date'])
weatherall=weatherall.iloc[:,2:8]
weatherall

weatherall.info()

fig,ax=plt.subplots()
fig.set_size_inches([10,5])
ax.plot(weatherall.index,weatherall.DayTemp,color='r',alpha=0.6)
ax.plot(weatherall.index,weatherall.NightTemp,color='b',alpha=0.6)
ax.set_ylabel('Temperature')
ax.set_title('Temperature line chart')
plt.show()

weatherall_year=weatherall.resample('A').mean()
weatherall_year

fig,ax=plt.subplots()
ax.plot(weatherall_year.index,weatherall_year.DayTemp,marker='o',color='r',alpha=0.6)
ax.plot(weatherall_year.index,weatherall_year.NightTemp,marker='o',color='b',alpha=0.6)
ax.set_title('Mean of Temperature')
plt.show()

weatherall_year_max=weatherall.resample('A').max()
weatherall_year_min=weatherall.resample('A').min()
fig,ax=plt.subplots()
ax.plot(weatherall_year_max.index,weatherall_year_max.DayTemp,marker='o',color='r',alpha=0.6)
ax.plot(weatherall_year_min.index,weatherall_year_min.DayTemp,marker='o',color='b',alpha=0.6)
ax.set_title("Temperature difference of years")
plt.show()

weatherall.boxplot(column='DayTemp',by='Year',figsize=(9,7))

weatherall_2016=weatherall['2016']
weatherall_2016

print('白天最高温度:',weatherall_2016['DayTemp'].max())
print('白天最低温度:',weatherall_2016['DayTemp'].min())

weatherall_2016_counts=weatherall_2016['Weather'].value_counts()
weatherall_2016_counts

weatherall_counts=weatherall['Weather'].value_counts()
weatherall_counts

count=weatherall.groupby(['Year','Weather']).count().unstack()
count=count.fillna(0)
count



