from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
import csv
import time

profile_path = r'C:\Users\micha\AppData\Roaming\Mozilla\Firefox\Profiles\ahh11qox.default-release-1623166099488'
options = Options()
options.set_preference('profile', profile_path)

options.set_preference('browser.download.folderList', 2)
options.set_preference('browser.download.dir', 'D:\\downloadfiles\\files')
options.set_preference('browser.helperApps.neverAsk.saveToDisk','application/zip,application/octet-stream,application/x-zip-compressed,multipart/x-zip,application/x-rar-compressed, application/octet-stream,application/msword,application/vnd.ms-word.document.macroEnabled.12,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/rtf,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel,application/vnd.ms-word.document.macroEnabled.12,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/xls,application/msword,text/csv,application/vnd.ms-excel.sheet.binary.macroEnabled.12,text/plain,text/csv/xls/xlsb,application/csv,application/download,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/octet-stream')
options.set_preference('general.warnOnAboutConfig', False)
options.set_preference('network.cookie.cookieBehaviour', 2)

service = Service(r'C:\Users\micha\Documents\geckodriver.exe')

browser = webdriver.Firefox(service=service, options=options)


browser.implicitly_wait(10)
filename = 'D:\downloadfiles\output2.csv'

with open(filename, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        print(row[0])
        try:
            browser.get(row[0])
            time.sleep(3)
            element = browser.find_element_by_xpath("//*[@id=\"__layout\"]/div/div[2]/div[2]/div/div[1]/div[1]/span[2]/a")
            ActionChains(browser).key_down(Keys.ALT).click(element).key_up(Keys.ALT).perform()
            time.sleep(3)
        except:
            print('gaming')


