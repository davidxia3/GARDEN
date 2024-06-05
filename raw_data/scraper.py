from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
import time
import os
import requests


options = Options()
options.headless = False


def download_image(url, folder_path, file_name):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(folder_path, "image" + file_name + ".jpg")
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded: {url}")
    except:
        pass


save_path = 'raw_data/diseased_images'

for starting_index in range(0, 1001, 200):

    link = 'https://universe.roboflow.com/hydromac/spinach-diseased/browse?queryText=&pageSize=200&startingIndex=' + str(starting_index) + '&browseQuery=true'

    driver = webdriver.Chrome(options=options)
    driver.get(link)
    time.sleep(20)

    change_view_button = driver.find_element(By.CSS_SELECTOR, '.far.fa-list-ul.rounded-l-md.flex.h-9.w-9.items-center.justify-center.border.pt-0\\.5.transition.duration-200.focus\\:outline-none.z-0.border-gray-600.bg-gray-900.text-gray-400.hover\\:border-gray-500.hover\\:bg-gray-800.hover\\:text-gray-200')
    change_view_button.click()
    time.sleep(3)


    x = driver.find_element(By.CSS_SELECTOR, '.grid.grid-cols-1.gap-1.md\\:grid-cols-1.lg\\:grid-cols-2')
    image_sections = x.find_elements(By.XPATH, './div')
    counter = starting_index
    for image_section in image_sections:
        hover_element = image_section.find_element(By.CSS_SELECTOR, '.absolute.right-7.top-0.z-30.h-5.w-5.rounded-md.border.invisible.border-aquavision.bg-aquavision-500.bg-opacity-10.hover\\:bg-opacity-40.group-hover\\:visible.group-hover\\:cursor-copy')
        actions = ActionChains(driver)
        actions.move_to_element(hover_element).perform()
        time.sleep(1)

        open_button = image_section.find_element(By.CSS_SELECTOR, '.previewImageControls.flex.h-full.w-full.items-center.justify-center.rounded.text-xs.transition-opacity.focus\\:opacity-100.focus\\:outline-none.focus\\:ring-2.focus\\:ring-purboflow-300\\/20.focus\\:ring-offset-2.focus\\:ring-offset-purboflow-300\\/20.fas.fa-expand-alt')
        open_button.click()
        time.sleep(2)

        image_element = driver.find_element(By.CSS_SELECTOR, '.h-full.transform.object-contain.transition-all')
        image_url = image_element.get_attribute('src')
        if image_url:
            download_image(image_url, save_path, str(counter))
        time.sleep(2)

        close_button = driver.find_element(By.CSS_SELECTOR, '.absolute.-right-12.-top-0.flex.h-7.w-7.items-center.justify-center.rounded-full.bg-white.shadow-2xl')
        close_button.click()
        time.sleep(2)
        counter=counter+1
