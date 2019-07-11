# -*- coding:utf-8 -*-
import time
import json
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_titles_in_cctv(file_path):
    """
    :param file_path:爬取结果存储路径
    :return: 无
    从央视网的滚动新闻上爬取近期NBA所有新闻的标题，并进行存储
    """
    url = "http://sports.cctv.com/nba/scrollnews/"
    driver = webdriver.PhantomJS(executable_path=r'F:\PhantomJS\phantomjs-2.1.1-windows\bin\phantomjs.exe')
    driver.get(url)
    f = open(file_path, "w")
    for page in range(10):
        print("Getting data from the page " + str(page))
        # 等到页面加载完成，再继续进行程序
        time.sleep(2)
        # 抓取标题数据
        for i in range(29):
            driver.find_element_by_id('contentELMT1412839887613320')
            ul = driver.find_element_by_xpath('//div[@id="contentELMT1412839887613320"]/ul/li['
                                              + str(page*30+i+1) + ']/a').text
            f.write(ul+'\n')
        # 触发下一页按钮
        if page == 0:
            driver.find_element_by_xpath('//div[@id="changeELMT1412839887613320"]/span[2]/a[6]').click()
        elif 0 < page < 5:
            driver.find_element_by_xpath('//div[@id="changeELMT1412839887613320"]/span[2]/a[7]').click()
        elif page < 9:
            driver.find_element_by_xpath('//div[@id="changeELMT1412839887613320"]/span[2]/a[9]').click()
    f.close()


def get_news_title_in_baidu(news_title, file_opened, driver):
    """:
    根据新闻标题，在百度中搜索相关新闻，并存储搜索到的新闻内容
    """
    url = "http://www.baidu.com"
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'kw')))
    # ctrl+a全选输入框内容
    driver.find_element_by_id('kw').send_keys(Keys.CONTROL, 'a')
    # ctrl+x剪切输入框内容
    driver.find_element_by_id('kw').send_keys(Keys.CONTROL, 'x')
    # 输入框重新输入内容
    driver.find_element_by_id('kw').send_keys(news_title)
    # 模拟Enter回车键
    driver.find_element_by_id('kw').send_keys(Keys.RETURN)
    # 百度需要一定的时间来进行搜索，并刷新页面
    time.sleep(3)

    for i in range(1, 6):
        # 循环等待到对应新闻标题渲染结束
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, str(i))))
        # 单击新闻标题，打开新的窗口
        new_link = driver.find_element_by_xpath('//div[@id="'+str(i)+'"]/h3/a')
        ActionChains(driver).move_to_element(new_link).click().perform()
        # 切换窗口至新打开的窗口
        change_window(driver)
        # 循环等待到标题出现
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'title')))
        except:
            change_window(driver, close=True)
            continue
        article_title = driver.title
        article_url = driver.current_url
        # 切换回搜索界面，同时关闭当前窗口
        change_window(driver, close=True)
        if not title_filter(article_title):
            continue
        file_opened.write(article_title+" "+article_url+"\n")
        print(article_title + " " + article_url)

    print("over")


def get_news_title_in_google(title_list):
    url = "https://www.google.com/"
    proxy = [
        '--proxy=%s' % "127.0.0.1:1080",  # 设置的代理ip
        '--proxy-type=http',  # 代理类型
        '--ignore-ssl-errors=true',  # 忽略https错误
    ]

    driver = webdriver.PhantomJS(executable_path=r'F:\PhantomJS\phantomjs-2.1.1-windows\bin\phantomjs.exe',
                                 service_args=proxy)

    file = open("SearchResultInGoogle.txt", "a", encoding='utf-8')
    driver.get(url)

    for index, news_title in enumerate(title_list):
        if index <= 242:
            continue
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, 'f')))
        # 输入框输入内容
        driver.find_element_by_xpath('//input[@title="Google 搜索"]').send_keys(news_title)
        # 模拟Enter回车键
        driver.find_element_by_xpath('//input[@title="Google 搜索"]').send_keys(Keys.RETURN)
        driver.save_screenshot("error.png")
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'center_col')))
        search_title = driver.title
        for i in range(1, 6):
            attempts = 0
            success = False
            while attempts < 3 and not success:
                try:
                    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'center_col')))
                    success = True
                except:
                    attempts += 1
                    driver.refresh()
                    if attempts == 3:
                        print("can not access")
                        continue

            # 单击新闻标题，刷新页面
            try:
                new_link = driver.find_element_by_xpath('//div[@id="ires"]/ol/div['+str(i)+']/h3/a')
            except:
                continue
            ActionChains(driver).move_to_element(new_link).click().perform()

            try:
                WebDriverWait(driver, 10).until_not(EC.title_is(search_title))
            except:
                driver.back()
                continue
            # 获取当前页面title以及url
            article_title = driver.title
            article_url = driver.current_url
            print(article_title + " " + article_url)
            file.write(article_title + " " + article_url + '\n')
            driver.save_screenshot('test4.png')
            # 后退至谷歌搜索页面
            driver.back()
        driver.get(url)
        print("title "+str(index))

    file.close()


# 两个窗口之间进行相互切换
def change_window(_driver, close=False):
    handles = _driver.window_handles
    print(handles)
    for handle in handles:
        if handle != _driver.current_window_handle:
            if close:
                _driver.close()
            _driver.switch_to.window(handle)
            break


def get_news_in_cctv(_driver):
    div_num = 10
    try:
        try:
            WebDriverWait(_driver, 10).until(EC.presence_of_element_located((By.XPATH, '/html/body/div['+str(div_num)+']/div[1]/div[1]/h1')))
        except:
            div_num = 9
            WebDriverWait(_driver, 10).until(EC.presence_of_element_located((By.XPATH, '/html/body/div['+str(div_num)+']/div[1]/div[1]/h1')))
    except:
        print("wrong cctv website")
        return None, None, None, None
    article_title = _driver.find_element_by_xpath('/html/body/div['+str(div_num)+']/div[1]/div[1]/h1').text
    article_info = _driver.find_element_by_xpath('/html/body/div['+str(div_num)+']/div[1]/div[1]/div/span[1]/i').text
    article_source = "央视网"
    article_date = article_info.split(" ")[1]
    print(article_title + '\n' + article_date)
    i = 2
    article_content = ""
    while True:
        try:
            buf = _driver.find_element_by_xpath('/html/body/div['+str(div_num)+']/div[1]/div[1]/p[' + str(i) + ']').text.strip(" ")
            article_content += buf
        except:
            break
        i += 1
    print(article_content)
    return article_title, article_date, article_source, article_content


def get_news_in_baijiahao(_driver):
    WebDriverWait(_driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'article-title')))
    article_title = _driver.find_element_by_class_name('article-title').text
    article_date = _driver.find_element_by_class_name('date').text
    article_source = _driver.find_element_by_class_name('source').text
    article_time = _driver.find_element_by_class_name('time').text
    article_content = _driver.find_element_by_class_name('article-content').text.replace('\n', '')
    print(article_title)
    print(article_source + " " + article_date + " " + article_time)
    print(article_content)
    return article_title, article_date, article_source, article_content


def get_news_in_nba(_driver):
    try:
        WebDriverWait(_driver, 10).until(EC.presence_of_element_located((By.ID, 'C-Main-Article-QQ')))
    except:
        return None, None, None, None
    article_title = _driver.find_element_by_xpath('//div[@id="C-Main-Article-QQ"]/div[1]/h1').text
    article_date = _driver.find_element_by_class_name('article-time').text.split(' ')[0]
    article_content = _driver.find_element_by_id('Cnt-Main-Article-QQ').text.replace('\n', '')
    article_source = "NBA中国官方网站"
    print(article_title)
    print(article_date)
    print(article_content)
    return article_title, article_date, article_source, article_content


def get_news_in_tencent(_driver):
    try:
        WebDriverWait(_driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'a_time')))
    except:
        print("wrong tencent website.")
        return None, None, None, None
    article_title = _driver.find_element_by_xpath('//div[@id="Main-Article-QQ"]/div/div[1]/div[1]/div[1]/h1').text
    article_date = _driver.find_element_by_class_name('a_time').text.split(' ')[0]
    print(article_title)
    print(article_date)
    article_content = ""
    i = 2
    while True:
        try:
            buf = _driver.find_element_by_xpath('//div[@id="Cnt-Main-Article-QQ"]/p[' + str(i) + ']').text
            article_content += buf
        except:
            break
        i += 1
    print(article_content)
    article_source = "腾讯新闻"
    return article_title, article_date, article_source, article_content


def get_news_in_sohu(_driver):
    WebDriverWait(_driver, 10).until(EC.presence_of_element_located((By.ID, 'news-time')))
    article_title = _driver.find_element_by_xpath('//div[@id="article-container"]/div[2]/div[1]/div[1]/div[1]/h1').text
    article_date = _driver.find_element_by_xpath('//*[@id="news-time"]').text
    article_content = _driver.find_element_by_xpath('//*[@id="mp-editor"]').text.replace('\n', '')
    article_source = '搜狐网'
    print(article_title)
    print(article_date)
    print(article_content)
    return article_title, article_date, article_source, article_content


def get_news_in_163(_driver):
    WebDriverWait(_driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'title')))
    article_title = _driver.find_element_by_class_name('title').text
    article_date = _driver.find_element_by_css_selector("[class = 'time js-time']").text.split(' ')[0]
    article_content = _driver.find_element_by_xpath('/html/body/main/article/div[2]/div').text.replace('\n', '')
    article_source = "网易新闻"
    print(article_title)
    print(article_date)
    print(article_content)
    return article_title, article_date, article_source, article_content


def get_news_in_sina(_driver):
    try:
        WebDriverWait(_driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'art_tit_h1')))
    except:
        print("Wrong sina website")
        return None, None, None, None
    article_title = _driver.find_element_by_class_name('art_tit_h1').text
    article_date = _driver.find_element_by_xpath('/html/body/main/section[1]/section/article/time').text
    article_date = article_date.split(' ')[0]
    article_content = _driver.find_element_by_css_selector("[class = 'art_pic_card art_content']").text
    article_content = article_content.replace('\n', '')
    article_source = '新浪体育'
    print(article_title)
    print(article_date)
    print(article_content)
    return article_title, article_date, article_source, article_content


def get_news_from_baidu(news_title, driver, tag):
    """
    根据新闻标题，在百度中搜索相关新闻，并存储搜索到的新闻内容
    """
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'kw')))
    # 输入框输入内容
    driver.find_element_by_id('kw').send_keys(news_title)
    # 模拟Enter回车键
    driver.find_element_by_id('kw').send_keys(Keys.RETURN)
    # 百度需要一定的时间来进行搜索，并刷新页面
    log = open('spiderLog.txt', 'a')
    # articles = []
    # buf = {}
    source = ""
    for i in range(1, 6):
        # 循环等待到对应新闻标题渲染结束
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, str(i))))
        # 单击新闻标题，打开新的窗口
        new_link = driver.find_element_by_xpath('//div[@id="'+str(i)+'"]/h3/a')
        ActionChains(driver).move_to_element(new_link).click().perform()
        # 切换窗口至新打开的窗口
        change_window(driver)
        try:  # 可能存在网站服务器问题，无法打开网站，直接跳过网站
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'title')))
        except:
            change_window(driver, True)
            log.write(news_title + str(i) + '\n' + "can not open website in baidu" + '\n')
            print("Can not open website in baidu. Reason: can not get title")
            continue
        time.sleep(5)
        source = source + driver.execute_script('return document.documentElement.outerHTML') + '\n\n\n'
        change_window(driver, True)
        # current_url = driver.current_url
        # source, cur_tag = source_judge(current_url)
        # if source is not None:
        #     if tag[cur_tag] == 0:
        #         tag[cur_tag] = 1
        #         article_title, article_data, article_source, article_content = source(driver)
        #         if article_title is None:
        #             change_window(driver, True)
        #             continue
        #         buf['article_title'] = article_title
        #         buf['article_data'] = article_data
        #         buf['article_source'] = article_source
        #         buf['article_content'] = article_content.replace('·', '')
        #         print(buf['article_content'])
        #         articles.append(buf)
        #         buf = {}
    driver.get("https://www.baidu.com")
    log.close()
    return source
    # return articles


def get_news_from_google(news_title, driver, tag):
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, 'f')))
    # 输入框输入内容
    driver.find_element_by_xpath('//input[@title="Google 搜索"]').send_keys(news_title)
    # 模拟Enter回车键
    driver.find_element_by_xpath('//input[@title="Google 搜索"]').send_keys(Keys.RETURN)
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'center_col')))
    search_title = driver.title
    log = open('spiderLog.txt', 'a')
    articles = []
    buf = {}
    source = ''
    for i in range(0, 5):
        attempts = 0
        success = False
        while attempts < 3 and not success:
            try:
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'center_col')))
                success = True
            except:
                attempts += 1
                driver.refresh()
                if attempts == 3:
                    log.write(news_title + str(i) + '\n' + 'can not access to google search' + '\n')
                    print("can not access")
        if not success:
            break

        # 单击新闻标题，刷新页面
        try:
            new_link = driver.find_element_by_xpath('//div[@id="ires"]/ol/div['+str(i)+']/h3/a')
        except:
            log.write(news_title + str(i) + '\n' + "can not find the link by xpath" + '\n')
            print("can not find the link by xpath")
            continue
        ActionChains(driver).move_to_element(new_link).click().perform()

        try:
            WebDriverWait(driver, 20).until_not(EC.title_is(search_title))
        except:
            log.write(news_title + str(i) + '\n' + "can not access to website in Google" + '\n')
            print("page jump failed")
            driver.back()
            continue
        time.sleep(5)
        source = source + driver.execute_script('return document.documentElement.outerHTML') + '\n==========================================================\n'
        # 获取当前页面title以及url
        # article_url = driver.current_url
        # source, cur_tag = source_judge(article_url)
        # if source is not None:
        #     if tag[cur_tag] == 0:
        #         tag[cur_tag] = 1
        #         article_title, article_data, article_source, article_content = source(driver)
        #         if article_title is None:
        #             driver.back()
        #             continue
        #         buf['article_title'] = article_title
        #         buf['article_data'] = article_data
        #         buf['article_source'] = article_source
        #         buf['article_content'] = article_content.replace('·', '')
        #         articles.append(buf)
        #         buf = {}
        # 后退至谷歌搜索页面
        driver.back()
    driver.get('https://www.google.com/')
    # return articles
    log.close()
    return source

# 判断网页属于哪一个站点
def source_judge(_url):
    if "cctv" in _url:
        return get_news_in_cctv, 'cctv'
    elif "baijiahao.baidu.com" in _url:
        return get_news_in_baijiahao, 'baidu'
    elif "nbachina.qq.com" in _url or "china.nba.com" in _url:
        return get_news_in_nba, 'nba'
    elif "sports.qq.com" in _url:
        return get_news_in_tencent, 'qq'
    elif "www.sohu.com" in _url:
        return get_news_in_sohu, 'sohu'
    elif "3g.163.com" in _url:
        return get_news_in_163, '163'
    elif "sports.sina.cn" in _url:
        return get_news_in_sina, 'sina'
    else:
        return None, ''


# 对新闻标题进行一定的筛选
def title_filter(_title):
    if "百度搜索" in _title or "百度资讯搜索" in _title:
        return False
    else:
        return True


# 对最终的爬虫结果进行一定的统计
def count():
    a = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(1, 290):
            try:
                json_file = open('corpus/News'+str(i)+'.json')
                load_json = json.load(json_file)
                json_file.close()
            except:
                print(i, "load failed")
                continue
            length = len(load_json['articles'])
            print(length)
            a[length-1] = a[length-1] + 1
            if length == 1:
                print(i, length)
    print(a)


# 同时爬取谷歌与百度上的新闻
def get_news_from_google_and_baidu():
    driver_baidu = webdriver.PhantomJS(executable_path=r'F:\PhantomJS\phantomjs-2.1.1-windows\bin\phantomjs.exe')
    driver_baidu.get('https://www.baidu.com/')

    proxy = [
        '--proxy=%s' % "127.0.0.1:1080",  # 设置的代理ip
        '--proxy-type=http',  # 代理类型
        '--ignore-ssl-errors=true',  # 忽略https错误
    ]
    driver_google = webdriver.PhantomJS(executable_path=r'F:\PhantomJS\phantomjs-2.1.1-windows\bin\phantomjs.exe',
                                        service_args=proxy)
    driver_google.get('https://www.google.com/')
    f = open("NewsList.txt", "r")
    num = 0
    for news in f:
        num += 1
        if num < 280 or num == 222 or num == 229 or num == 254:
            continue
        source_file = open("Source"+str(num)+'.txt', 'w', encoding='utf-8')
        # news_file = open("News"+str(num)+'.json', 'w')
        url_tag = {'cctv': 0, 'baidu': 0, 'nba': 0, 'qq': 0, 'sohu': 0, '163': 0, 'sina': 0}
        print("From Biadu:")
        source_baidu = get_news_from_baidu(news, driver_baidu, url_tag)
        print("From Google:")
        source_google = get_news_from_google(news, driver_google, url_tag)
        print(url_tag)
        sources = source_baidu + source_google
        source_file.write(sources)
        # res = {}
        # articles_baidu.extend(articles_google)
        # res['articles'] = articles_baidu
        # json_str = json.dump(res, news_file, ensure_ascii=False)
        source_file.close()
    f.close()


# 处理spiderLog记录的爬虫错误记录
def handle_errors():
    logs = []
    with open("spiderLog.txt", 'r') as f:
        line_num = 1
        log = {'title': '', 'num': 0, 'error': '', 'source_num': 0}
        for line in f:
            if line_num == 1:
                log['title'] = line.replace('\n', '')
                line_num = 2
            elif line_num == 2:
                log['num'] = str(line.replace('\n', ''))
                line_num = 3
            elif line_num == 3:
                log['error'] = line.replace('\n', '')
                line_num = 1
                logs.append(log)
                log = {'title': '', 'num': 0, 'error': '', 'source_num': 0}

    with open('NewsList.txt', 'r') as f:
        line_num = 0
        for line in f:
            line_num = line_num + 1
            for log in logs:
                if line.replace('\n', '') == log['title']:
                    log['source_num'] = line_num

    print(len(logs))
    # driver_baidu = webdriver.PhantomJS(executable_path=r'F:\PhantomJS\phantomjs-2.1.1-windows\bin\phantomjs.exe')
    # driver_baidu.get('https://www.baidu.com/')
    proxy = [
        '--proxy=%s' % "127.0.0.1:1080",  # 设置的代理ip
        '--proxy-type=http',  # 代理类型
        '--ignore-ssl-errors=true',  # 忽略https错误
    ]
    driver_google = webdriver.PhantomJS(executable_path=r'F:\PhantomJS\phantomjs-2.1.1-windows\bin\phantomjs.exe',
                                        service_args=proxy)
    driver_google.get('https://www.google.com/')
    log_file = open('spiderLog.txt', 'a')
    for log in logs:
        # if log['error'] == 'can not open website in baidu':
        #     WebDriverWait(driver_baidu, 10).until(EC.presence_of_element_located((By.ID, 'kw')))
        #     # 输入框输入内容
        #     driver_baidu.find_element_by_id('kw').send_keys(log['title'])
        #     # 模拟Enter回车键
        #     driver_baidu.find_element_by_id('kw').send_keys(Keys.RETURN)
        #     # 循环等待到对应新闻标题渲染结束
        #     WebDriverWait(driver_baidu, 10).until(EC.presence_of_element_located((By.ID, log['num'])))
        #     # 单击新闻标题，打开新的窗口
        #     new_link = driver_baidu.find_element_by_xpath('//div[@id="' + log['num'] + '"]/h3/a')
        #     ActionChains(driver_baidu).move_to_element(new_link).click().perform()
        #     # 切换窗口至新打开的窗口
        #     change_window(driver_baidu)
        #     WebDriverWait(driver_baidu, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'title')))
        #     # 充分加载网页内容
        #     time.sleep(5)
        #     source = driver_baidu.execute_script('return document.documentElement.outerHTML') + '\n==========================================================\n'
        #     change_window(driver_baidu, True)
        #     f = open('page_source/Source' + str(log['source_num']) + '.txt', 'a', encoding='utf-8')
        #     f.write(source)
        #     f.close()
        # if log['error'] == 'can not access to Google search':
        #     print(log)
        #     WebDriverWait(driver_google, 20).until(EC.presence_of_element_located((By.NAME, 'f')))
        #     # 输入框输入内容
        #     driver_google.find_element_by_xpath('//input[@title="Google 搜索"]').send_keys(log['title'])
        #     # 模拟Enter回车键
        #     driver_google.find_element_by_xpath('//input[@title="Google 搜索"]').send_keys(Keys.RETURN)
        #     WebDriverWait(driver_google, 20).until(EC.presence_of_element_located((By.ID, 'center_col')))
        #     search_title = driver_google.title
        #     source = ''
        #     for i in range(int(log['num']), 6):
        #         WebDriverWait(driver_google, 20).until(EC.presence_of_element_located((By.ID, 'center_col')))
        #         # 单击新闻标题，刷新页面
        #         new_link = driver_google.find_element_by_xpath('//div[@id="ires"]/ol/div[' + str(i) + ']/h3/a')
        #         ActionChains(driver_google).move_to_element(new_link).click().perform()
        #         try:
        #             WebDriverWait(driver_google, 20).until_not(EC.title_is(search_title))
        #         except:
        #             log_file.write(log['title'] + str(i) + '\n' + "can not access to website in Google" + '\n')
        #             print("page jump failed")
        #             driver_google.back()
        #             continue
        #         time.sleep(10)
        #         source = source + driver_google.execute_script(
        #             'return document.documentElement.outerHTML') + '\n==========================================================\n'
        #         driver_google.back()
        #     f = open('page_source/Source' + str(log['source_num']) + '.txt', 'a', encoding='utf-8')
        #     f.write(source)
        #     f.close()
        #     driver_google.get('https://www.google.com/')
        if log['error'] == 'can not access to website in Google':
            print(log)
            WebDriverWait(driver_google, 20).until(EC.presence_of_element_located((By.NAME, 'f')))
            # 输入框输入内容
            driver_google.find_element_by_xpath('//input[@title="Google 搜索"]').send_keys(log['title'])
            # 模拟Enter回车键
            driver_google.find_element_by_xpath('//input[@title="Google 搜索"]').send_keys(Keys.RETURN)
            WebDriverWait(driver_google, 20).until(EC.presence_of_element_located((By.ID, 'center_col')))
            search_title = driver_google.title
            # 单击新闻标题，刷新页面
            new_link = driver_google.find_element_by_xpath('//div[@id="ires"]/ol/div[' + str(log['num']) + ']/h3/a')
            ActionChains(driver_google).move_to_element(new_link).click().perform()
            try:
                WebDriverWait(driver_google, 20).until_not(EC.title_is(search_title))
            except:
                print(str(log['source_num']) + "page jump failed")
                driver_google.get('https://www.google.com/')
                continue
            time.sleep(10)
            source = driver_google.execute_script(
                'return document.documentElement.outerHTML') + '\n==========================================================\n'
            f = open('page_source/Source' + str(log['source_num']) + '.txt', 'a', encoding='utf-8')
            f.write(source)
            f.close()
            driver_google.get('https://www.google.com/')

    log_file.close()


if __name__ == '__main__':
    # handle_errors()
    count()
