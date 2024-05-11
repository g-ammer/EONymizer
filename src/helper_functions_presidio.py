from presidio_analyzer import RecognizerResult
from typing import List
import re
import os
import concurrent.futures


##################################################################################################################################
## This function loads .txt files from a local folder and converts them into a dict of the format {filename: text}
def convert_txt_to_dict(directory_path):
    output_dict = {}
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            output_dict[filename] = content

    return output_dict
    
##################################################################################################################################


## This function loads .txt files from the mounted google drive folder and converts them into a dict of the format {filename: text}
def convert_txt_to_dict_from_shared_gdrive(folder_id):
    # Dictionary to store filename and text content
    txt_dict = {}

    # Function to download and process a file
    def process_file(file):
        if file['title'].endswith('.txt'):
            # Download the txt file
            file.GetContentFile(file['title'])
            # Read the text content of the file
            with open(file['title'], 'r') as txt_file:
                text = txt_file.read()
            # Add filename and text content to the dictionary
            txt_dict[file['title']] = text
            # Remove the downloaded txt file
            os.remove(file['title'])

    # List files in the shared folder
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks for each file
        futures = [executor.submit(process_file, file) for file in file_list]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

    return txt_dict

##################################################################################################################################


## This function loads .txt files from the mounted google drive folder and converts them into a dict of the format {filename: text}
def convert_txt_to_dict_from_mounted_gdrive(folder_path):
  
  texts_dict = {}

  txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

  for file_name in txt_files:
      file_path = os.path.join(folder_path, file_name)
      
      with open(file_path, 'r') as file:
          file_content = file.read()
          texts_dict[file_name] = file_content

  return texts_dict

##################################################################################################################################


## Function that merges entities of the same type when separated by <= 2 characters
def merge_adjacent_entities(
                              self,
                              text: str,
                              analyzer_results: List[RecognizerResult]
                              ) -> List[RecognizerResult]:
    """Merge adjacent entities if separated by 2 characters."""
    merged_results = []
    prev_result = None
    for result in analyzer_results:
        if prev_result is not None:
            if prev_result.entity_type == result.entity_type:
                if re.search(r'(?s)^.{1,2}$', text[prev_result.end:result.start]):
                    merged_results.remove(prev_result)
                    result.start = prev_result.start
        merged_results.append(result)
        prev_result = result
    return merged_results

##################################################################################################################################


## Function that merges different entities to single entity "ADDRESS"
def merge_to_address(
                      self,
                      text: str,
                      analyzer_results: List[RecognizerResult]
                      ) -> List[RecognizerResult]:
    """Merge adjacent entities if separated by 2 characters."""
    merged_results = []
    to_merge_list = ["STRASSE", "ORT", "PLZ", "ADDRESS", "ADRESSE"]
    prev_result = None
    for result in analyzer_results:
        if prev_result is not None:
            if prev_result.entity_type in to_merge_list:
              if result.entity_type in to_merge_list:
                if re.search(r'(?s)^.{1,2}$', text[prev_result.end:result.start]):
                #if result.start - prev_result.end < 3:
                  merged_results.remove(prev_result)
                  result.start = prev_result.start
                  result.entity_type="ADRESSE"
        merged_results.append(result)
        prev_result = result
    return merged_results

##################################################################################################################################


# Convert labeled data into list of dicts with start and end key
def convert_labels_list_to_results_list(input_list):
  #input_dict = input_dict
  output_list = []

  for item in input_list:
      category = item['category']
      start = item['offset']
      end = item['offset'] + item['length']
      score = 1.0
      output_list.append({'entity_type': category, 'start': start, 'end': end, 'score': score})

  return output_list

##################################################################################################################################


# Convert labeled data list of dicts into list of presidio_analyzer.recognizer_result files
# for further processing
def convert_list_to_recognizer_type(input_list):
  results_labeled_recognizer = []

  for i in range(len(input_list)):
    if input_list[i]['start'] != None:
      list_interm_recog = RecognizerResult(
                                          #entity_type='DEFAULT',  # converts all entity types to default
                                          entity_type=input_list[i]['entity_type'],  # keep entity type
                                          start=input_list[i]['start'],
                                          end=input_list[i]['end'],
                                          score=input_list[i]['score']
                                          )
      results_labeled_recognizer.append(list_interm_recog)

  return results_labeled_recognizer

##################################################################################################################################


# Convert results from presidio analyzer into list of dicts
def results_recognizer_to_list(results_recognizer):

  results_list = []
  for i in range(len(results_recognizer)):
    original_dict = results_recognizer[i].__dict__
    new_dict = {key: original_dict[key] for key in ['entity_type', 'start', 'end', 'score']}
    results_list.append(new_dict)

  return results_list

##################################################################################################################################

##################################################################################################################################

##################################################################################################################################

##################################################################################################################################

##################################################################################################################################

##################################################################################################################################
