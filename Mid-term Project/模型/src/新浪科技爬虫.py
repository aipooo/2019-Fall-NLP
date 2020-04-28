import requests
from bs4 import BeautifulSoup

headers = {"User-Agent":"Mozilla/5.0"}


def getNewsDetailUrlList(url):
    '''
    根据给定的url(这里为新浪科技首页)抓取网页的新闻链接
    并从中筛选出符合要求的连接，返回链接列表
    '''
    #利用request模拟HTTP请求
    wbdata = requests.get(url, headers=headers)
    wbdata.decoding = 'utf-8'
    #对网站信息进行解析
    soup = BeautifulSoup(wbdata.content, 'lxml')
    news_link_data = soup.select("li > a")
    news_link_list = []
    for item in news_link_data:
        link = item.get('href')
        #筛选符合要求的链接
        if 'tech' in link and '2019' in link and 'photo' not in link:
            news_link_list.append(link)
    return news_link_list


#获取新闻标题
def getNewsTitle(url):
    wbdata = requests.get(url, headers=headers)
    wbdata.decoding = 'utf-8'
    soup = BeautifulSoup(wbdata.content, 'lxml')
    news_title_list = soup.select('body > div.main-content.w1240 > h1')
    if len(news_title_list):
        news_title = news_title_list[0].get_text()
    else:
        news_title = ''
    return news_title


# 获取新闻详情内容
def getNewsContent(url):
    wbdata = requests.get(url, headers=headers)
    wbdata.decoding = 'utf-8'
    soup = BeautifulSoup(wbdata.content, 'lxml')
    news_data = soup.select('#artibody > p')
    news_content = []
    for item in news_data:
        news_content.append(item.get_text()) 
    return news_content


#将新闻内容保存为txt
def saveTXT(url, index):
    file_name = str(index)+'.txt'
    file = open(file_name, 'w', encoding='utf-8')
    title = getNewsTitle(url)
    content = getNewsContent(url)
    file.write(title.strip())
    file.write('\n')
    for para in content:
        if len(para):
            file.write(para.strip())
            file.write('\n')
    file.close()
    print(str(index)+'.txt is finished.')


def saveLinkToTXT(url_list):
    file = open('link.txt', 'w')
    for url in url_list:
        file.write(url)
        file.write('\n')
    file.close()


def readLinkTXT():
    file = open('link.txt', 'r')
    url_list = []
    for line in file:
        url_list.append(line.strip())
    file.close()
    return url_list

def getTXTSize(index):
    file_name = str(index)+'.txt'
    file = open(file_name, 'r', encoding='utf-8')
    content = file.read()
    size = len(content)
    file.close()
    return size
    

if __name__ == '__main__':
#    url = "https://tech.sina.com.cn/"
#    news_url_list = readLinkTXT()
#    url_num = len(news_url_list)
#    for news_url in getNewsDetailUrlList(url):
#        if news_url not in news_url_list:
#            news_url_list.append(news_url)
#            url_num += 1
#            saveTXT(news_url_list[url_num-1], url_num)
#    saveLinkToTXT(news_url_list)
    
    url = "https://tech.sina.com.cn/"
    news_url_list = readLinkTXT()
    url_num = len(news_url_list)
    not_valid = []
    for index in range(1, url_num+1):
        if getTXTSize(index) < 200:
            not_valid.append(index)      
    
    for news_url in getNewsDetailUrlList(url):
        if news_url not in news_url_list:
            if len(not_valid)>0 :
                saveTXT(news_url, not_valid[-1])
                news_url_list[not_valid[-1]-1] = news_url
                not_valid.pop()
            else:
                news_url_list.append(news_url)
                url_num += 1
                saveTXT(news_url_list[url_num-1], url_num)
    saveLinkToTXT(news_url_list)    
    