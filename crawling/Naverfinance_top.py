import traceback
import requests
import re
import pandas as pd
import csv
from bs4 import BeautifulSoup
 def crawl(url):
    data = requests.get(url)
    _soap = BeautifulSoup(data.text,'lxml')
    print(data)
    return data.content
def getItem(tr):
    tds = tr.findAll("td")
    rank = tds[0].text
    name = tds[1].find("a")
    href = name["href"]
    name = name.text
#    code = int(href[20:])+1000000
    code = str('"'+href[20:]+'"')
    price = tds[2].text
    up_down = tds[3].find('img')
    up_down = str(up_down)
    up_down_num = tds[3].text
    up_down_num = up_down_num.replace(",","")
    up_down_num = re.findall("\d+",up_down_num)
    up_down_num = up_down_num[0]
    if up_down.find("상승")!=-1:
        up_down = ("+"+up_down_num)
    else:
        if up_down.find("하락")> -1:
            up_down = ("-"+up_down_num)
        else:
            up_down = ("0")
    to = tds[6].text
    fori = tds[8].text
    volume = tds[9].text
    return {"rank":rank,"name":name,"code":code,"price":price,
            "change":up_down,"totalprice":to,"foreign":fori,"volume":volume}
def parse(pageString):
    bsobj = BeautifulSoup(pageString,"html.parser")
    navi = bsobj.find("div",{"class":"box_type_l"})
    navi2 = navi.find("table",{"class":"type_2"})
    navi3 = navi.find("tbody")
    navi4 = navi3.findAll("tr")
    stockinfos = []
    for tr in navi4:
        try:
            stockInfo = getItem(tr)
            stockinfos.append(stockInfo)
            print(stockInfo)
        except Exception as e:
            print("error")
            pass
    return stockinfos
def getsiseMarketSum(sosok,page):
    url="https://finance.naver.com/sise/sise_market_sum.nhn?sosok={}&page={}".format(sosok,page)
    pageString = crawl(url)
    list = parse(pageString)
    return list
result = []
for page in range(1,3):  
    list = getsiseMarketSum(0,page)
    result += list
    
time_table = pd.DataFrame(result)
# os import
import os
import datetime
path_dir = 'data\\totalmarket'
if not os.path.exists(path_dir):
    os.makedirs(path_dir)
str_dateto = datetime.datetime.strftime(datetime.datetime.today(), '%Y.%m.%d')
path = os.path.join(path_dir,'{date}Top_total_market.csv'.format(date=str_dateto))
time_table.to_csv(path,index=False)
