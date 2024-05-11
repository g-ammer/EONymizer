# This script provides functions for data conversion, score calculation and plotting
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# This function convertes the dict from the json file that was generated with Azure AI languague studio
# and converts it into a dict of the format that is accepted by the score_calculator function
def convert_labels_dict(input_dict):
   
  num_labeled_texts = sum('location' in d for d in input_dict.get('assets', {}).get('documents', []))
  filenames_labeled = []

  for i in range(num_labeled_texts):
    filenames_labeled.append(input_dict['assets']['documents'][i]['location'])
  
  result_dict = {}
  for location in filenames_labeled:
      if 'assets' in input_dict and 'documents' in input_dict['assets']:
          labels_list = []
          converted_list = []
          for document in input_dict['assets']['documents']:
              if document.get('location') == location:
                  for entity in document.get('entities', []):
                      labels_list.extend(entity.get('labels', []))

                      for item in labels_list:
                        category = item['category']
                        start = item['offset']
                        end = item['offset'] + item['length']
                        score = 1.0
                        converted_list.append({'entity_type': category, 'start': start, 'end': end, 'score': score})

                  break
          result_dict[location] = converted_list
      else:
          result_dict[location] = []
  return result_dict

#########################################################################################################################################


# This function merges the entities defined in the entities_to_merge list when they are separated by less than 3 characters
def address_merger_result_list(results_list, entities_to_merge = ['STRASSE', 'PLZ', 'ORT', 'ADRESSE']):
    
    results_list = sorted(results_list, key=lambda x: x['start'] if x['start'] is not None else float('inf'))
    merged_results = []
    prev_result = None
    for result in results_list:
        if prev_result is not None:
            if prev_result['entity_type'] in entities_to_merge:
              if result['entity_type'] in entities_to_merge:
                if result['start'] is not None and prev_result['start'] is not None and result['end'] is not None and prev_result['end'] is not None:
                  if result['start'] - prev_result['end'] < 3:
                    merged_results.remove(prev_result)
                    result['start'] = prev_result['start']
                    result['entity_type'] = "ADRESSE"
        merged_results.append(result)
        prev_result = result

    return merged_results

#########################################################################################################################################

# This function applies the address_merger_result_list function to the full results_dict
def dict_address_merger(input_dict):

    merged_dict = {}
    for key, value in input_dict.items():
        merged_dict[key] = address_merger_result_list(value)

    return merged_dict


#########################################################################################################################################

# This function calculates a list of true positive, false positive and false negative entities and
# precision, recall and f1-score when comparing the result_dict from 2 texts

def score_calculator(
                    predicted_list,
                    true_list,
                    fuzzy = 3,
                    score = False
                    ):

  true_positive = 0
  false_positive = 0
  false_negative = 0

  true_positive_list = []
  false_positive_list = []
  false_negative_list = []
  true_positive_scores = []
  false_positive_scores = []

  conf_matrix_dict = {}

  for dict_a in predicted_list:
      
      match_found = False

      for dict_b in true_list:

          if dict_a['start'] is not None and dict_b['start'] is not None and dict_a['end'] is not None and dict_b['end'] is not None:

            if abs(dict_a['start'] - dict_b['start']) <= fuzzy and abs(dict_a['end'] - dict_b['end']) <= fuzzy:

                # Create dictionary for confusion matrix
                if dict_a['entity_type'] not in conf_matrix_dict.keys():
                  conf_matrix_dict[dict_a['entity_type']] = []
                conf_matrix_dict[dict_a['entity_type']].append(dict_b['entity_type'])

                true_positive += 1
                true_positive_list.append(dict_a['entity_type'])
                if score == True:
                  true_positive_scores.append(float(dict_a['score']))
                match_found = True
                break

      if not match_found:
          false_positive += 1
          false_positive_list.append(dict_a['entity_type'])
          if score == True:
            false_positive_scores.append(float(dict_a['score']))

  for dict_b in true_list:

      match_found = False
      for dict_a in predicted_list:
          if dict_a['start'] is not None and dict_b['start'] is not None and dict_a['end'] is not None and dict_b['end'] is not None:
            if abs(dict_b['start'] - dict_a['start']) <= fuzzy and abs(dict_b['end'] - dict_a['end']) <= fuzzy:
                match_found = True
                break

      if not match_found:
          false_negative += 1
          false_negative_list.append(dict_b['entity_type'])

  precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
  recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
  f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

  metrics_dict = {
      "precision": precision,
      "recall": recall,
      "f1_score": f1_score,
      "true_positive": true_positive,
      "false_positive": false_positive,
      "false_negative": false_negative,
      "true_positive_list": true_positive_list,
      "false_positive_list": false_positive_list,
      "false_negative_list": false_negative_list,
      "true_positive_scores": true_positive_scores,
      "false_positive_scores": false_positive_scores,
      "conf_matrix_dict": conf_matrix_dict
      }

  return metrics_dict

#########################################################################################################################################


# This function retrieves all the entity types that occur in the result_dict
def get_unique_entity_types(data_dict):

    unique_entities = set()
    for file_dict in data_dict.values():
        for nested_dict in file_dict:
            entity_type = nested_dict.get('entity_type')
            if entity_type:
                unique_entities.add(entity_type)
    return list(unique_entities)

#########################################################################################################################################


# This function calculates model scores for entire batches of predictions
def calculate_model_score(
                          prediction_dict_input,
                          true_dict_input,
                          fuzzy = 3,
                          entity_mapping_dict = {},
                          address_merge = False,
                          rec_score = False
                          ):
   
  # Change name of entities according to entity_mapping_dict
  for result_list in prediction_dict_input.values():
    for dict_a in result_list:
      entity_type = dict_a['entity_type']
      if dict_a['entity_type'] in entity_mapping_dict:
        dict_a['entity_type'] = entity_mapping_dict[entity_type]

  for result_list in true_dict_input.values():
    for dict_b in result_list:
      entity_type = dict_b['entity_type']
      if dict_b['entity_type'] in entity_mapping_dict:
        dict_b['entity_type'] = entity_mapping_dict[entity_type]

  # Merge several entites (defined in dict_address_merger) to single entity 'ADRESSE'
  if address_merge == True:
    prediction_dict = dict_address_merger(prediction_dict_input)
    true_dict = dict_address_merger(true_dict_input)

  if address_merge == False:
    prediction_dict = prediction_dict_input
    true_dict = true_dict_input

  # Calculate score_dict_all
  score_dict_all = {}

  for key in true_dict.keys():

    score_dict = score_calculator(
                                  prediction_dict[key],
                                  true_dict[key],
                                  fuzzy = fuzzy,
                                  )
    
    score_dict_all[key] = score_dict

  # Use score_dict to calculate all metrics
  sum_true_positive = 0
  sum_false_positive = 0
  sum_false_negative = 0
  perfect_prediction = 0
  precision_list = []
  recall_list = []
  f1_score_list = []
  true_positive_list = []
  false_positive_list = []
  false_negative_list = []
  true_positive_scores_list = []
  false_positive_scores_list = []
  conf_matrix_dict_all = {}

  for score in score_dict_all.values():
      sum_true_positive += score['true_positive']
      sum_false_positive += score['false_positive']
      sum_false_negative += score['false_negative']

      precision_list.append(score['precision'])
      recall_list.append(score['recall'])
      f1_score_list.append(score['f1_score'])

      true_positive_list.extend(score['true_positive_list'])
      false_positive_list.extend(score['false_positive_list'])
      false_negative_list.extend(score['false_negative_list'])
      
      if rec_score == True:
        true_positive_scores_list.extend(score['true_positive_scores'])
        false_positive_scores_list.extend(score['false_positive_scores'])

      if score['false_positive'] == 0 and score['false_negative'] == 0 and score['precision'] == 1.0 and score['recall'] == 1.0 and score['f1_score'] == 1.0:
        perfect_prediction += 1

      # Create dict for confusion matrix for all texts
      for key, values in score['conf_matrix_dict'].items():
          if key not in conf_matrix_dict_all:
              conf_matrix_dict_all[key] = []
          conf_matrix_dict_all[key].extend(values)


  precision_all = sum_true_positive / (sum_true_positive + sum_false_positive) if sum_true_positive + sum_false_positive > 0 else 0
  recall_all = sum_true_positive / (sum_true_positive + sum_false_negative) if sum_true_positive + sum_false_negative > 0 else 0
  f1_score_all = 2 * (precision_all * recall_all) / (precision_all + recall_all) if precision_all + recall_all > 0 else 0

  perfect_prediction_percent = "{:.0%}".format(perfect_prediction/len(score_dict_all))

  # Count true positives, false positives and false negatives for each entity
  from collections import Counter

  TP_list = [entity_mapping_dict.get(item, item) for item in true_positive_list]
  FP_list = [entity_mapping_dict.get(item, item) for item in false_positive_list]
  FN_list = [entity_mapping_dict.get(item, item) for item in false_negative_list]

  TP_counts = Counter(TP_list)
  FP_counts = Counter(FP_list)
  FN_counts = Counter(FN_list)

  # Get unique identity types from input dictionaries
  unique_entity_types_predictions = get_unique_entity_types(prediction_dict)
  unique_entity_types_true = get_unique_entity_types(true_dict)

  # Create dicitionary with all relevant output information
  metrics_dict_all = {
                      "true_positive_sum": sum_true_positive,
                      "false_positive_sum": sum_false_positive,
                      "false_negative_sum": sum_false_negative,
                      "precision_all": precision_all,
                      "recall_all": recall_all,
                      "f1_score_all": f1_score_all,
                      "precision_list": precision_list,
                      "recall_list": recall_list,
                      "f1_score_list": f1_score_list,
                      "true_positive_list": true_positive_list,
                      "false_positive_list": false_positive_list,
                      "false_negative_list": false_negative_list,
                      "true_positive_scores_list": true_positive_scores_list,
                      "false_positive_scores_list": false_positive_scores_list,
                      "conf_matrix_dict_all": conf_matrix_dict_all,
                      "TP_list": TP_list,
                      "FP_list": FP_list,
                      "FN_list": FN_list,
                      "TP_counts": TP_counts,
                      "FP_counts": FP_counts,
                      "FN_counts": FN_counts,
                      "unique_entity_types_predictions": unique_entity_types_predictions,
                      "unique_entity_types_true": unique_entity_types_true
                      }

  print(f"{perfect_prediction_percent} of texts ({perfect_prediction}/{len(score_dict_all)}) were perfectly anonymized by the model")

  print("Model Precision:", "{:.2%}".format(precision_all))
  print("Model Recall:", "{:.2%}".format(recall_all))
  print("Model F1_score:", "{:.2%}".format(f1_score_all))
  print("True Positives:", sum_true_positive)
  print("False Positives:", sum_false_positive)
  print("False Negatives:", sum_false_negative)
  
  return metrics_dict_all, score_dict_all

#########################################################################################################################################

# This function calculates and plots a confusion_matrix
def make_confusion_matrix(metrics_dict_all):

  unique_entities_predictions = list(set(metrics_dict_all['TP_list'] + metrics_dict_all['FP_list']))
  
  conf_matrix = np.zeros((len(unique_entities_predictions), len(metrics_dict_all['unique_entity_types_true'])))

  for predicted_label, true_labels in metrics_dict_all["conf_matrix_dict_all"].items():
      predicted_label_index = unique_entities_predictions.index(predicted_label)
      for true_label in true_labels:
          true_label_index = metrics_dict_all['unique_entity_types_true'].index(true_label)
          conf_matrix[predicted_label_index][true_label_index] += 1

  # Rearrange confusion matrix
  row_sums = np.max(conf_matrix, axis=1)
  col_sums = np.max(conf_matrix, axis=0)

  row_indices = np.argsort(row_sums)[::-1]
  col_indices = np.argsort(col_sums)[::-1]

  rearranged_matrix = conf_matrix[row_indices][:, col_indices]

  # Rearrange labels
  x_labels = metrics_dict_all['unique_entity_types_true'].copy()
  x_labels = np.array(x_labels)[col_indices]

  y_labels = unique_entities_predictions.copy()
  y_labels = np.array(y_labels)[row_indices]

  FP_array = np.array([metrics_dict_all['FP_counts'].get(c, 0) for c in y_labels])
  FN_array = np.array([metrics_dict_all['FN_counts'].get(c, 0) for c in x_labels])
  FP_array = np.append(FP_array, 0)
  
  confusion_matrix = np.vstack((rearranged_matrix, FN_array))
  confusion_matrix = np.column_stack((confusion_matrix, FP_array))

  x_labels = np.append(x_labels, 'FALSE POSITIVES')
  y_labels = np.append(y_labels, 'FALSE NEGATIVES')

  return confusion_matrix, x_labels, y_labels

#########################################################################################################################################


# This function takes a quadratic confusion matrix with matched entities and calculates adapted performance metrics based on the
# entity type (--> words that were detected but labeled with the wrong entity_type are considered false positives)
def score_from_confusion_matrix(confusion_matrix):
  TP = np.sum(confusion_matrix[0][:-1,:-1])
  TP_entities = np.sum(np.diag(confusion_matrix[0][:-1,:-1]))
  FP = np.sum(confusion_matrix[0][:-1,-1])
  FP_entities = FP + TP - TP_entities
  FN_entities = np.sum(confusion_matrix[0][-1,:-1])

  precision = TP_entities / (TP_entities + FP_entities) if TP_entities + FP_entities > 0 else 0
  recall = TP_entities / (TP_entities + FN_entities) if TP_entities + FN_entities > 0 else 0
  f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

  metrics_dict_entities = {
                          "TP_entities": TP_entities,
                          "FP_entities": FP_entities,
                          "FN_entities": FN_entities,
                          "Precision_entities": precision,
                          "Recall_entities": recall,
                          "f1_score_entities": f1_score
                          }


  print("Model Performance based on Entity Type:")
  print("Model Precision:", "{:.2%}".format(precision))
  print("Model Recall:", "{:.2%}".format(recall))
  print("Model F1_score:", "{:.2%}".format(f1_score))
  print("True Positives:", int(TP_entities))
  print("False Positives:", int(FP_entities))
  print("False Negatives:", int(FN_entities))

  return metrics_dict_entities

#########################################################################################################################################

# This function plots the confusion_matrix as a seaborn heatmap

def plot_confusion_matrix(confusion_matrix):

# Create confusion matrix figure
  fig_cm, ax = plt.subplots(1,1)

  sns.heatmap( 
              confusion_matrix[0],
              annot = True,
              fmt = 'g',
              xticklabels = confusion_matrix[1],
              yticklabels = confusion_matrix[2]
              )
  
  ax.set_ylabel("Predicted Entities", fontsize=12)
  ax.set_xlabel("True Entities", fontsize=12)

  return fig_cm

#########################################################################################################################################


# This function creates a figure with a barplot for all entities
def plot_entity_barplot(metrics_dict_all):
  
  # Get all unique keys from both dictionaries
  keys_set = set(list(metrics_dict_all['TP_counts'].keys()) + list(metrics_dict_all['FP_counts'].keys()) + list(metrics_dict_all['FN_counts'].keys()))
  all_keys = sorted(keys_set, key=lambda x: sum([metrics_dict_all['TP_counts'].get(x, 0), metrics_dict_all['FP_counts'].get(x, 0), metrics_dict_all['FN_counts'].get(x, 0)]), reverse=True)

  # Create lists of values for each dictionary, filling in missing keys with 0
  values_TP = [metrics_dict_all['TP_counts'].get(key, 0) for key in all_keys]
  values_FP = [metrics_dict_all['FP_counts'].get(key, 0) for key in all_keys]
  values_FN = [metrics_dict_all['FN_counts'].get(key, 0) for key in all_keys]
  values_TPFP = [x + y for x, y in zip(values_TP, values_FP)]

  # Plotting the stacked bar chart
  fig_bar, ax = plt.subplots(1,1, figsize=(6,4))
  bar_width = 0.5
  index = np.arange(len(all_keys))

  ax.bar(index, values_TP, bar_width, label='True Positive', color='C0')
  ax.bar(index, values_FP, bar_width, label='False Positive', color='C3', bottom=values_TP)
  ax.bar(index, values_FN, bar_width, label='False Negative', color='C2', bottom=values_TPFP)

  ax.set_xlabel('Entity Type')
  ax.set_ylabel('Nr. of Entities')
  ax.set_title('Analysis of PII Entities')
  ax.set_xticks(index, all_keys, rotation=45, ha='right')
  plt.tick_params(axis='x', which='major', labelsize=9)
  plt.legend(fontsize=9)

  return fig_bar

#########################################################################################################################################


# This function creates a figure where precision, recall, f1-score and their distributions are plotted
def plot_scores(metrics_dict_all):

  # Calculate cumulative distributions
  sorted_precision = np.sort(metrics_dict_all['precision_list'])
  cdf_precision = np.arange(sorted_precision.size) / float(sorted_precision.size)

  sorted_recall = np.sort(metrics_dict_all['recall_list'])
  cdf_recall = np.arange(sorted_recall.size) / float(sorted_recall.size)

  sorted_f1_score = np.sort(metrics_dict_all['f1_score_list'])
  cdf_f1_score = np.arange(sorted_f1_score.size) / float(sorted_f1_score.size)

  ## PLOT RESULTS FIGURE
  fig_score, ax = plt.subplots(3,3, figsize=(10,8))

  # Plot piecharts
  ax[0,0].set_position([0.118, 0.61, 0.25, 0.25])
  ax[0,0].pie([metrics_dict_all['precision_all'], 1 - metrics_dict_all['precision_all']],
                          radius=1,
                          colors=['C0', 'none'],
                          labels=['', ''], autopct='',
                          pctdistance=0.0,
                          labeldistance=0,
                          startangle=105,
                          wedgeprops=dict(width=0.45, edgecolor='white'))
  ax[0,0].text(0.5, 0.5, f"{metrics_dict_all['precision_all']:.1%}", fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax[0,0].transAxes)
  ax[0,0].text(0.5, 1, "Precision", fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax[0,0].transAxes)

  ax[0,1].set_position([0.387, 0.61, 0.25, 0.25])
  ax[0,1].pie([metrics_dict_all['recall_all'], 1 - metrics_dict_all['recall_all']],
                          radius=1,
                          colors=['C1', 'none'],
                          labels=['', ''], autopct='',
                          pctdistance=0.0,
                          labeldistance=0,
                          startangle=105,
                          wedgeprops=dict(width=0.45, edgecolor='white'))
  ax[0,1].text(0.5, 0.5, f"{metrics_dict_all['recall_all']:.1%}", fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax[0,1].transAxes)
  ax[0,1].text(0.5, 1, "Recall", fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax[0,1].transAxes)

  ax[0,2].set_position([0.662, 0.61, 0.25, 0.25])
  ax[0,2].pie([metrics_dict_all['f1_score_all'], 1 - metrics_dict_all['f1_score_all']],
                          radius=1,
                          colors=['C2', 'none'],
                          labels=['', ''], autopct='',
                          pctdistance=0.0,
                          labeldistance=0,
                          startangle=105,
                          wedgeprops=dict(width=0.45, edgecolor='white'))
  ax[0,2].text(0.5, 0.5, f"{metrics_dict_all['f1_score_all']:.1%}", fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax[0,2].transAxes)
  ax[0,2].text(0.5, 1, "F1-score", fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax[0,2].transAxes)

  # Plot histograms
  bins = np.histogram(np.hstack((metrics_dict_all['precision_list'], metrics_dict_all['recall_list'], metrics_dict_all['f1_score_list'])), bins=10)[1]

  ax[1,0].hist(metrics_dict_all['precision_list'], bins, color='C0', weights=np.ones(len(metrics_dict_all['precision_list'])) / len(metrics_dict_all['precision_list'])*100)
  ax[1,0].set_xlim(-0.02, 1.02)
  #ax[1,0].set_ylim(0, 100)
  ax[1,0].set_ylabel('Texts [%]')

  ax[1,1].hist(metrics_dict_all['recall_list'], bins, color='C1', weights=np.ones(len(metrics_dict_all['recall_list'])) / len(metrics_dict_all['recall_list'])*100)
  ax[1,1].set_xlim(-0.02, 1.02)
  #ax[1,1].set_ylim(0, 100)

  ax[1,2].hist(metrics_dict_all['f1_score_list'], bins, color='C2', weights=np.ones(len(metrics_dict_all['f1_score_list'])) / len(metrics_dict_all['f1_score_list'])*100)
  ax[1,2].set_xlim(-0.02, 1.02)
  #ax[1,2].set_ylim(0, 100)

  # Plot cumulative distributions
  ax[2,0].plot(sorted_precision, cdf_precision, color='C0', linewidth=2)
  ax[2,0].set_xlim(-0.02, 1.02)
  ax[2,0].set_ylim(0, 1)
  ax[2,0].set_ylabel('Cumulative Probability')

  ax[2,1].plot(sorted_recall, cdf_recall, color='C1', linewidth=2)
  ax[2,1].set_xlim(-0.02, 1.02)
  ax[2,1].set_ylim(0, 1)

  ax[2,2].plot(sorted_f1_score, cdf_f1_score, color='C2', linewidth=2)
  ax[2,2].set_xlim(-0.02, 1.02)
  ax[2,2].set_ylim(0, 1)

  return fig_score