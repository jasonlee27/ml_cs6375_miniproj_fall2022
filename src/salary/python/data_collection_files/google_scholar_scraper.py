import pandas as pd
from scholarly import scholarly

UTD_scholar_instance = scholarly.search_author_by_organization(6037664147916511730)
data = {
    "Name": [],
    "citedby5y" : [],
    "hindex" : [],
    "hindex5y": [],
    "i10index": [],
    "i10index5y": []
}

while True:
    try:
        professor_instance = scholarly.fill(next(UTD_scholar_instance))
        for e in data:
            data[e].append(professor_instance[e])
        
    except:
        df = pd.DataFrame(data)
        df.to_csv("scholar_data.csv")
