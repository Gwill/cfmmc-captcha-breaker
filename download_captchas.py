# coding: utf8
import sys
import datetime
import numpy as np
import PIL
import requests
from io import BytesIO
import seleniumrequests
from selenium.webdriver.common.keys import Keys
import network
import CaptchaGenerator.generate_captcha as capgen


def download_captchas(model, browser, user_id, password):
    while True:
        browser.get('https://investorservice.cfmmc.com/login.do')

        img = browser.find_element_by_xpath("//img[@id='imgVeriCode']")
        url = img.get_attribute('src')
        img, vericode = predict_captcha(model, browser, url)

        txt_user_id = browser.find_element_by_name('userID')
        txt_user_id.send_keys(user_id)
        txt_password = browser.find_element_by_name('password')
        txt_password.send_keys(password)
        txt_vericode = browser.find_element_by_name('vericode')
        txt_vericode.send_keys(vericode)

        txt_vericode.submit()

        btn_logout = browser.find_elements_by_xpath("//input[@name='logout']")
        if btn_logout:
            img.save('captchas/{}.jpg'.format(vericode))
            btn_logout[0].submit()


def predict_captcha(model, browser, url):
    response = browser.request('GET', url)
    img = PIL.Image.open(BytesIO(response.content))
    img_bin = capgen.binarization(img)
    X1 = np.zeros((1, 25, 96, 3), dtype=np.uint8)
    X1[0] = img_bin
    y1 = model.predict(X1)
    vericode = network.decode(y1)
    return img, vericode


model = network.create_model()
model.load_weights('my_model_weights_gen.h5')

user_id = input("Enter username:")
password = input("Enter password:")
browser = seleniumrequests.Chrome()
download_captchas(model, browser, user_id, password)

#browser.get('https://investorservice.cfmmc.com/login.do')


#img = browser.find_element_by_xpath("//img[@id='imgVeriCode']")
#url = img.get_attribute('src')
#vericode = predict_captcha(model, browser, url)



browser.quit()
