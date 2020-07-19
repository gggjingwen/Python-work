import requests
import re
import pandas as pd

url_date="http://lishi.tianqi.com/qingdao/index.html"
header={'user-agent':'Mozilla/5.0'}
r=requests.get(url_date,headers=header)
r.encoding=r.apparent_encoding
html=r.text

hrefurls=re.findall(r'href="http://\S*[qingdao]\S\d{6}.html"',html)
hrefurls
urls=[]
for i in hrefurls:
    url=i.replace('href="','').replace('"','')
    urls.append(url)

weather=[]
for url in urls:
    header={'user-agent':'Mozilla/5.0'}
    r=requests.get(url,headers=header)
    r.encoding=r.apparent_encoding
    html=r.text
    div=re.findall(r'<div class="tqtongji2">.*?</div>',html,re.S)
    ul=re.findall(r'<ul>(.*?)</ul>',div[0],re.S) 
    for i in ul:
        dataall=[]
        li=re.findall(r'<li>(.*)</li>',i)
        for a in li:
            dataall.append(a)
        weather.append(dataall)
weatherall=pd.DataFrame(weather)
weatherall.columns=["Date","DayTemp","NightTemp","Weather","Wind","WindPower"]
date=[]
for url in urls:
    header={'user-agent':'Mozilla/5.0'}
    r=requests.get(url,headers=header)
    r.encoding=r.apparent_encoding
    html=r.text
    div=re.findall(r'<div class="tqtongji2">.*?</div>',html,re.S)
    ul=re.findall(r'<ul>(.*?)</ul>',div[0],re.S)   
    for i in ul:  
        li=re.findall(r'<li><a.*>(\d{4}-\d{2}-\d{2})</a></li>',i)
        for a in li:
            date.append(a)
for i in range(len(weatherall.Date)):
    if len(weatherall.Date[i])>10:
        weatherall.Date[i]=date[i][0:10]
weatherall.to_csv("weather.csv",encoding="utf_8_sig")



