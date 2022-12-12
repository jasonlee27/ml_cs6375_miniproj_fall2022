from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import time

URL = "https://www.utdallas.edu/directory/"
names_filename = "final.csv"

names = pd.read_csv(names_filename)["name"]

driver = webdriver.Firefox()
driver.get(URL)

select = Select(driver.find_element(By.ID, "dirAffil"))
select.select_by_visible_text('Faculty')
search_bar = driver.find_element(By.CLASS_NAME, "quicksearch")

results = []

for name in names:

    search_bar.clear()
    search_bar.send_keys()
    search_bar.send_keys(Keys.RETURN)

    time.sleep(.5)
    try:
        profile = driver.find_element(By.CLASS_NAME, "resultPage")
    except NoSuchElementException:
        department = None 
    else:
        profile_info = profile.text.split("\n")
        department = " ".join(profile_info[5].split()[1:])
    
    results.append([name, department])

driver.close()

df = pd.DataFrame(results)
df.columns = ["name","department"]        
df.to_csv("departments.csv", index=False)