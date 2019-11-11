import re
import urllib.request
import urllib
import requests
from bs4 import BeautifulSoup


#获取该网页的源代码
def getPage(url):
    try:
        user_agent = 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36'
        header = {'User-Agent': user_agent}
        request = urllib.request.Request(url,headers=header)
        response = urllib.request.urlopen(request)
        pageCode = response.read().decode('utf-8')
        return pageCode
    except Exception as e:
        print('Get Page Fail')
        print("An error has occurred:"+e)

#将双色球的信息写入txt文件
def writePage(date,dateNum,num):
    #日期的列表会比期号长1，因为最后加入了一组页数统计，所以不能以日期列表长度来写数据
    with open('./DCnumber.txt','a+') as f:
        for x in range(0,len(num)):
            f.write('('+date[x]+';'+dateNum[x]+';'+num[x]+')'+'\n')

#获取总页数
def getPageNum(url):
    try:
        pageCode = getPage(url)
        soup = BeautifulSoup(pageCode,'lxml')
        td_code = soup.find('td',colspan='7')       #获取表格中包含页数的列
        result = td_code.get_text().split(' ')      #将该列转换成列表，我们
        #result = ['\n', '共125', '页', '/2484', '条记录', '首页', '上一页', '下一页', '末页', '当前第', '2', '页']
        #用正则表达式从reslut[1]中提取出数字
        list_num = re.findall("[0-9]{1}",result[1])
        #将提取出来的数字转换成十进制的数值
        page_num = int(list_num[0]+list_num[1]+list_num[2])
        return page_num
    except Exception as e:
        print('Get Page Number Fail')
        print("An error has occurred:" + e)

#获取单页中的双色球中奖信息
def getDC(url):
    #循环读取每一页的信息
    for num in range(1,getPageNum(url)+1):
        print('begin get page:'+str(num))
        href = 'http://kaijiang.zhcw.com/zhcw/html/ssq/list_'+str(num)+'.html'
        page = BeautifulSoup(getPage(href),'lxml')
        em_list = page.find_all('em')   #获取该页面中的em内容,即中奖编号所在
        td_list = page.find_all('td',{'align':'center'})    #获取该页面中的开奖日期，期号等信息

        i = 1   #计数器，每七个号码为一组
        DCnum = ''      #存放一期中奖号码
        DCnum_list = []     #存放该页每期的中奖号码
        for em in em_list:
            emnum = em.get_text()
            if i == 7:
                DCnum = DCnum + emnum
                DCnum_list.append(DCnum)
                DCnum = ''          #重置一期的号码
                i = 1               #重置计数器
            else:
                DCnum = DCnum + emnum +','
                i += 1

        DCdate = []         #存放开奖日期
        DCdateN = []        #存放期号
        t = 1              #计数器，每5个为一组，我们只需要每组的前两个td内容
        for td in td_list:
            td_text = td.get_text()
            if t == 1:
                DCdate.append(td_text)
                t += 1
            elif t == 2:
                DCdateN.append(td_text)
                t += 1
            elif t == 5:
                t = 1
            else:
                t+=1
        writePage(DCdate,DCdateN,DCnum_list)


if __name__ == '__main__':
    getDC('http://kaijiang.zhcw.com/zhcw/html/ssq/list_2.html')

