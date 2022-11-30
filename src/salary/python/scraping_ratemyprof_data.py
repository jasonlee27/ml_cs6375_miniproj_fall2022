import requests
import json
import math
import os
import pandas as pd

class Professor:
    def __init__(self, ratemyprof_id: int, first_name: str, last_name: str, num_of_ratings: int, overall_rating):
        self.ratemyprof_id = ratemyprof_id

        self.name = f"{first_name} {last_name}"
        self.first_name = first_name
        self.last_name = last_name
        self.num_of_ratings = num_of_ratings

        if self.num_of_ratings < 1:
            self.overall_rating = 0

        else:
            self.overall_rating = float(overall_rating)


class ProfessorNotFound(Exception):
    def __init__(self, search_argument, search_parameter: str = "Name"):

        self.search_argument = self.search_argument

        self.search_parameter = search_parameter

    def __str__(self):

        return (
            f"Proessor not found"
            + f" The search argument {self.search_argument} did not"
            + f" match with any professor's {self.search_parameter}"
        )


class RateMyProfApi:
    def __init__(self, school_id: str = "1273", testing: bool = False):
        self.UniversityId = school_id
        # dict of Professor
        self.professorlist = []
        self.professors= self.scrape_professors(testing)
        self.indexnumber = False

    def scrape_professors(self, testing: bool = False):
        professors = dict()
        num_of_prof = self.get_num_of_professors(self.UniversityId)
        num_of_pages = math.ceil(num_of_prof / 20)

        for i in range(1, num_of_pages + 1): 
            page = requests.get(
                "http://www.ratemyprofessors.com/filter/professor/?&page="
                + str(i)
                + "&filter=teacherlastname_sort_s+asc&query=*%3A*&queryoption=TEACHER&queryBy=schoolId&sid="
                + str(self.UniversityId)
            )
            json_response = json.loads(page.content)
            temp_list = json_response["professors"]


            for json_professor in json_response["professors"]:
                # print(json_professor)
                self.professorlist.append({
                    "Fname": json_professor["tFname"],
                    "Lname": json_professor["tLname"],
                    "Dept": json_professor["tDept"],
                    "rating_class": json_professor["rating_class"],
                    "total_Ratings": json_professor["tNumRatings"],
                    "overall_rating" : json_professor["overall_rating"]
                })
                professor = Professor(
                    json_professor["tid"],
                    json_professor["tFname"],
                    json_professor["tLname"],
                    json_professor["tNumRatings"],
                    json_professor["overall_rating"])
                
                

                professors[professor.ratemyprof_id] = professor
            data = json.dumps(self.professorlist, indent=4)
            df = pd.read_json(data)
            df.to_csv("ratemyprof.csv")

            # for test cases, limit to 2 iterations
            if testing and (i > 1): break

        return professors

    def get_num_of_professors(self, id):  # function returns the number of professors in the university of the given ID.
        page = requests.get(
            "http://www.ratemyprofessors.com/filter/professor/?&page=1&filter=teacherlastname_sort_s+asc&query=*%3A*&queryoption=TEACHER&queryBy=schoolId&sid="
            + str(id)
        )  # get request for the specific result page
        temp_jsonpage = json.loads(page.content)
        num_of_prof = (temp_jsonpage["remaining"] + 20)  # get the number of professors at The University of Texas at Dallas
        return num_of_prof


if __name__ == '__main__':

    # Gets UTD data
    # UTD id = 1273
    uci = RateMyProfApi(1273)
