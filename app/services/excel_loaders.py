import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

from app.core.exceptions import KnowledgeSourceError
from app.core.logging import get_logger

logger = get_logger(__name__)


def dataframe_document_loader(df, page_content_column_name):
    try:
        return DataFrameLoader(df, page_content_column=page_content_column_name).load()
    except Exception as e:
        logger.exception("Loading documents from dataframe failed")
        raise KnowledgeSourceError("Failed to load documents") from e


class LevelCompetencyKnowledgeLoader:
    def create_levels_json(self, l1, l2, l3, l4, l5, l6, l7):
        return {
            "Level1NewtotheIssue__c": l1,
            "Level2Beginner__c": l2,
            "Level3BasicUnderstanding__c": l3,
            "Level4ProfoundUnderstanding__c": l4,
            "Level5Master__c": l5,
            "Level6Expertise__c": l6,
            "Level7Excellence__c": l7,
        }

    def create_page_content(self, description, level_competency):
        levels_description = []
        for key, value in level_competency.items():
            levels_description.append(f"{key} : {value}\n")
        levels_description_text = "".join(levels_description)
        return f"Description : {description}\n {levels_description_text}"

    def load_and_processed_excel(self, filepath, file_type):
        try:
            if file_type == "csv":
                df = pd.read_csv(filepath, header=2)
            else:
                df = pd.read_excel(filepath, header=2)
            df.drop(
                columns=[
                    "Answer alignes also to the following competencies (Category)",
                    "Answers",
                    "Keywords",
                    "Description_Available__c",
                    "Languages__c",
                    "Level_Description_Available__c",
                ],
                axis=1,
                inplace=True,
            )
            df.rename(
                columns={"Main Competency": "Competency", "Definition of Competency": "Description"},
                inplace=True,
            )
            df = df.ffill().drop_duplicates().reset_index(drop=True)
            df["LevelofCompetency"] = df.apply(
                lambda row: self.create_levels_json(
                    row["Level1NewtotheIssue__c"],
                    row["Level2Beginner__c"],
                    row["Level3BasicUnderstanding__c"],
                    row["Level4ProfoundUnderstanding__c"],
                    row["Level5Master__c"],
                    row["Level6Expertise__c"],
                    row["Level7Excellence__c"],
                ),
                axis=1,
            )
            df["page_content"] = df.apply(
                lambda row: self.create_page_content(row["Description"], row["LevelofCompetency"]),
                axis=1,
            )
            return df[["Competency", "Catagory", "LevelofCompetency", "page_content"]]
        except Exception as e:
            logger.exception("Parsing the competency sheet failed")
            raise KnowledgeSourceError("Failed to parse competency sheet") from e


class JobProfileKnowledgeLoader:
    def create_page_content(self, responsiblity):
        return f"Responsiblities : {responsiblity}"

    def load_and_processed_excel(self, filepath, file_type):
        try:
            if file_type == "csv":
                df = pd.read_csv(filepath, header=0)
            else:
                df = pd.read_excel(filepath, header=0)
            df["page_content"] = df["Business_Responsibilities__c"].apply(
                lambda x: "Responsibilities: " + str(x) if pd.notnull(x) else "Responsibilities: "
            )
            df.drop(["Business_Responsibilities__c"], axis=1, inplace=True)
            return df
        except Exception as e:
            logger.exception("Parsing the job profile sheet failed")
            raise KnowledgeSourceError("Failed to parse job profile sheet") from e


class MotivationKnowledgeLoader:
    def create_page_content(self, definaton, key_dim, description):
        return definaton + "\n" + key_dim + "\n" + "Description/Keywords : \n" + description

    def load_and_processed_excel(self, filepath, file_type):
        try:
            if file_type == "csv":
                df = pd.read_csv(filepath, header=1)
            else:
                df = pd.read_excel(filepath, header=1)
            df.drop(["Language", "Unnamed: 4"], axis=1, inplace=True)
            df["page_content"] = df.apply(
                lambda row: self.create_page_content(
                    row["Definition of Motivational Factor"],
                    row["Key Dimensions"],
                    row["Description / Key Words"],
                ),
                axis=1,
            )
            df.drop(
                ["Definition of Motivational Factor", "Key Dimensions", "Description / Key Words"],
                axis=1,
                inplace=True,
            )
            return df
        except Exception as e:
            logger.exception("Parsing the motivation sheet failed")
            raise KnowledgeSourceError("Failed to parse motivation sheet") from e


def get_loader_by_type(source_type: str):
    if source_type == "competency":
        return LevelCompetencyKnowledgeLoader()
    if source_type == "job_profile":
        return JobProfileKnowledgeLoader()
    return MotivationKnowledgeLoader()
