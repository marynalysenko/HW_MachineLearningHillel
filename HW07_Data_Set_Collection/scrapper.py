import os
import urllib.request
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import re



def create_images_folder():
    if not os.path.exists("workouters"):
        os.mkdir("workouters")


def save_image(image_url):
    pattern = re.compile(r'(https://i.pinimg.com/)\d+x(/.*\..*)')
    url = pattern.sub(r'\1originals\2', image_url)
    urllib.request.urlretrieve(url, f'workouters/image_{url.split("/")[-1]}')


def scrap_images(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        wait = WebDriverWait(driver, 10)
        images = wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, "img")))
        for image in images:
            try:
                image_url = image.get_attribute("src")
                if image_url.startswith("https") and "RS" not in image_url:
                    save_image(image_url)
            except StaleElementReferenceException:
                pass
            except Exception as e:
                print(f'Error type: {type(e)} \n with message: {str(e)}')
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def main():
    create_images_folder()
    driver = webdriver.Chrome()
    #driver.get("https://www.pinterest.com/marinalsnk/running-and-cardio/")
    driver.get("https://www.pinterest.com/marinalsnk/strength-training/")
    #driver.get("https://www.pinterest.com/marinalsnk/yoga-and-mindfulness/")

    scrap_images(driver)
    driver.quit()


if __name__ == "__main__":
    main()


#Error type: <class 'urllib.error.HTTPError'>
# with message: HTTP Error 403: Forbidden
#Error type: <class 'urllib.error.HTTPError'>
# with message: HTTP Error 403: Forbidden

#Process finished with exit code 0