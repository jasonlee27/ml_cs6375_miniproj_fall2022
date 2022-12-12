import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

results = []

URL = 'https://profiles.utdallas.edu/schools/'

schools = ["AHT", "BBS", "ECS", "EPPS", "IS", "JSOM", "NSM"]

num_school_pages = {"AHT":11, "BBS":5, "ECS":10, "EPPS":5, "IS":1, "JSOM":8, "NSM":11}

for school in schools:

    for page_num in range(1, num_school_pages[school] + 1):

        # Getting specific page
        page = requests.get(URL + school + "?page=" + str(page_num))
        page_soup = BeautifulSoup(page.content, "html.parser")
        page_results = page_soup.find_all(class_="card-title profile-name")

        # Iterating through results on page 
        for page_result in page_results:
            row = []
            page_result = page_result.find("a")
            profile_url = page_result.get('href')

            # Adding name and school
            row.append(page_result.string)
            row.append(school)

            # Navigating to profile page to get year of graduation
            profile_page = requests.get(profile_url)
            profile_soup = BeautifulSoup(profile_page.content, "html.parser")
            profile_results = profile_soup.find(id="preparation")

            # Iterating to see if any years listed
            if profile_results:
                profile_results = profile_results.find_all(class_="entry")
                years = []
                for result in profile_results:
                    for entry in result.contents:
                        matches = re.findall(r'.*([1-2][0-9]{3})', str(entry))
                        years += matches
                if len(years) > 0:
                    years = [int(year) for year in years]
                    row.append(min(years))
                else:
                    row.append(None)
            else:
                row.append(None)

            results.append(row)

df = pd.DataFrame(results)
df.columns = ["name","school","year"]        
df.to_csv("profiles.csv", index=False)