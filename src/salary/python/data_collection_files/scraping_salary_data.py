import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# df = pd.DataFrame
# df.columns = ["Name","Title","Salary"]
#df = np.array(["Name","Title","Salary"])

salary_data = []

URL = "https://www.openthebooks.com/members/employer-detail/?Id=8212&tab=1&Year_S=2021&pg=1"
page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")
results = soup.find(class_="employer-detail-table")

for i in range(1,101+1):
    print(str(i) + "th page")
    URL = "https://www.openthebooks.com/members/employer-detail/?Id=8212&tab=1&Year_S=2021&pg=" + str(i)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(class_="employer-detail-table")
    employee_details = results.find_all("tr")

    for employee in employee_details:
        data = employee.find_all("td")
        row = []
        for item in data:
            row.append(item.contents[0])
        if len(row) > 0:
            salary_data.append(row[2:5])


df = pd.DataFrame(salary_data)
df.columns = ["Name","Title","Salary"]

df.to_csv("salary_data.csv", index=False)
