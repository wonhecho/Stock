import requests
import datetime
#삼성전자
code = '005930'
url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)
# headers 코드 없을 시에 페이지를 찾을 수 없는 block이 걸림.
res = requests.get(url,headers={'User-agent':'Mozilla/5.0'})
res.encoding = 'utf-8'

from bs4 import BeautifulSoup
soap = BeautifulSoup(res.text,'lxml')
el_table_navi = soap.find("table", class_="Nnavi")
el_td_last = el_table_navi.find("td",class_="pgRR")

#총 페이지수 확인
pg_last = el_td_last.a.get('href').rsplit('&')[1]
pg_last = pg_last.split('=')[1]
pg_last = int(pg_last)

#테이블의 정보를 가져오는 함수
def parsing(code,page):
   try:
      url = 'http://finance.naver.com/item/sise_day.nhn?code={code}&page={page}'.format(code=code,page=page)
      res = requests.get(url,headers={'User-agent':'Mozilla/5.0'})
      _soap = BeautifulSoup(res.text, 'lxml')
        _df = pd.read_html(str(_soap.find("table")), header=0)[0]
        _df = _df.dropna()
        return _df
    except Exception as e:
        traceback.print_exc()
    return None
#시작지점과 종료지점을 정함
str_datefrom = datetime.datetime.strftime(datetime.datetime(year=2021, month=1, day=1), '%Y.%m.%d')
str_dateto = datetime.datetime.strftime(datetime.datetime.today(), '%Y.%m.%d')
df = None
# parsing
for page in range(1, pg_last+1):
    _df = parse_page(code, page)
    _df_filtered = _df[_df['날짜'] > str_datefrom]
    if df is None:
        df = _df_filtered
    else:
        df = pd.concat([df, _df_filtered])
    if len(_df) > len(_df_filtered):
       break
import os
import os
path_dir = 'data\\'+code
if not os.path.exists(path_dir):
    os.makedirs(path_dir)
path = os.path.join(path_dir, '{code}_{date_from}_{date_to}.csv'.format(code=code, date_from=str_datefrom, date_to=str_dateto))
df.to_csv(path, index=False)

   
