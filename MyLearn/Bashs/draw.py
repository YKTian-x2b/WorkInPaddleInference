import matplotlib.pyplot as plt
import re
import numpy as np

proj_path = '/tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_704/'

def parseResult_single(before_dataPath, split_num):
  before_content_list = []
  with open(before_dataPath, 'r') as file:
    before_content = file.read()
    total_lines = len(before_content)  
    head_line_index = total_lines // split_num  
    for ii in range(1, split_num):
      before_content_list.append(before_content[head_line_index*ii: head_line_index*(ii+1)])
 
  seq_len_list = [[] for _ in range(split_num-1)]
  before_time_cost_list = [[] for _ in range(split_num-1)]

  for ii in range(split_num-1):
    before_content = before_content_list[ii]

    idx = 0
    for line in before_content.splitlines():
        pattern = 'seq_lens: \[(.*)\],      time_cost: (.*)ms'
        r = re.search(pattern, line)
        if r is not None:
            seq_len = int(r.groups()[0])
            time_cost = float(r.groups()[1])
            seq_len_list[ii].append(seq_len)
            before_time_cost_list[ii].append(time_cost)
            idx += 1
  
  before_time_cost_list_out = np.mean(np.array(before_time_cost_list), axis=0)
  return seq_len_list[0], before_time_cost_list_out


def parseResult(before_dataPath, after_dataPath, split_num):
  before_content_list = []
  after_content_list = []
  with open(before_dataPath, 'r') as file:
    before_content = file.read()
    total_lines = len(before_content)  
    head_line_index = total_lines // split_num  
    for ii in range(1, split_num):
      before_content_list.append(before_content[head_line_index*ii: head_line_index*(ii+1)])
  with open(after_dataPath, 'r') as file:
    after_content = file.read() 
    total_lines = len(after_content)  
    head_line_index = total_lines // split_num  
    for ii in range(1, split_num):
      after_content_list.append(after_content[head_line_index*ii: head_line_index*(ii+1)])
  
  seq_len_list = [[] for _ in range(split_num-1)]
  before_time_cost_list = [[] for _ in range(split_num-1)]
  after_time_cost_list = [[] for _ in range(split_num-1)]
  for ii in range(split_num-1):
    before_content = before_content_list[ii]
    after_content = after_content_list[ii]

    idx = 0
    for line in before_content.splitlines():
        pattern = 'seq_lens: \[(.*)\],      time_cost: (.*)ms'
        r = re.search(pattern, line)
        if r is not None:
            seq_len = int(r.groups()[0])
            time_cost = float(r.groups()[1])
            seq_len_list[ii].append(seq_len)
            before_time_cost_list[ii].append(time_cost)
            idx += 1
    idx = 0
    for line in after_content.splitlines():
        pattern = 'seq_lens: \[(.*)\],      time_cost: (.*)ms'
        r = re.search(pattern, line)
        if r is not None:
            seq_len = int(r.groups()[0])
            time_cost = float(r.groups()[1])
            assert seq_len == seq_len_list[ii][idx]
            after_time_cost_list[ii].append(time_cost)
            idx += 1
  
  before_time_cost_list_out = np.mean(np.array(before_time_cost_list), axis=0)
  after_time_cost_list_out = np.mean(np.array(after_time_cost_list), axis=0)
  return seq_len_list[0], before_time_cost_list_out, after_time_cost_list_out


def draw(seq_len_list, before_time_cost_list, after_time_cost_list,  after_2_time_cost_list, end_dataPath):
  plt.clf()
  save_path = proj_path + 'pic/'

  x_axis_data = seq_len_list
  y_before_time_cost = before_time_cost_list
  y_after_time_cost = after_time_cost_list

  plt.plot(x_axis_data, y_before_time_cost, marker='s', markersize=2, color='tomato',
            linestyle='-', label='before')
  plt.plot(x_axis_data, y_after_time_cost, marker='o', markersize=2, color='y',
            linestyle='-', label='after_withSPLIT')
  plt.plot(x_axis_data, after_2_time_cost_list, marker='*', markersize=4, color='m',
            linestyle='-', label='after_withoutSPLIT')
  # plt.plot(x_axis_data, y_host_memory_percent_data, marker='x', markersize=4, color='g',
  #           linestyle='--', label='host_memory_percent')

  plt.legend(loc='upper left')
  plt.xlabel('seq_len') 
  plt.ylabel('time_cost')
  # plt.title('BlackBoxResourceUtilization')
  
  plt.savefig(save_path + 'mmha_llm_pref' +  end_dataPath + '.jpg')


def draw2(seq_len_list, time_cost_lists, labels, end_dataPath):
  plt.clf()
  save_path = proj_path + 'pic/'

  x_axis_data = seq_len_list

  assert len(time_cost_lists) == len(labels) == 5
  markers = ['s', 'o', '^', 'v', '<']
  colors = ['tomato', 'y', 'm', 'g', 'r']
  
  for ii in range(len(time_cost_lists)):
    y_time_cost = time_cost_lists[ii]
    plt.plot(x_axis_data, y_time_cost, marker=markers[ii], markersize=2, color=colors[ii],
            linestyle='-', label=labels[ii])

  # plt.plot(x_axis_data, y_before_time_cost, marker='s', markersize=2, color='tomato',
  #           linestyle='-', label='before')
  # plt.plot(x_axis_data, y_after_time_cost, marker='o', markersize=2, color='y',
  #           linestyle='-', label='after_withSPLIT')
  # plt.plot(x_axis_data, after_2_time_cost_list, marker='*', markersize=4, color='m',
  #           linestyle='-', label='after_withoutSPLIT')
  # plt.plot(x_axis_data, y_host_memory_percent_data, marker='x', markersize=4, color='g',
  #           linestyle='-', label='host_memory_percent')

  plt.legend(loc='upper left')
  plt.xlabel('seq_len') 
  plt.ylabel('time_cost')
  # plt.title('BlackBoxResourceUtilization')
  
  plt.savefig(save_path + 'mmha_llama_pref' +  end_dataPath + '.jpg')

if __name__ == "__main__":
  # before_dataPath_base = proj_path + "log_short/time_before"
  # after_dataPath_base = proj_path + "log_short/time_after"
  # after_2_dataPath_base = proj_path + "log_short/time_after"

  r"""
  # for list 
  all_innerDataPath_list = ["_3072_1", "_2048_2", "_2048_1", 
                           "_1024_8", "_1024_4", "_1024_2", 
                           "_1024_1"] 

  before_end_path_str = "_short.txt"
  after_end_path_str = "_dirtyV4_509V2_short.txt"
  
  after_split_num_list_base = [8, 4, 8, 1, 2, 4, 8]
  after_split_num_list = [item * 6 for item in after_split_num_list_base]
  before_split_num_list = after_split_num_list
  assert len(before_split_num_list) == len(after_split_num_list)
 
  for i in range(len(before_split_num_list)):
    before_dataPath = before_dataPath_base + all_innerDataPath_list[i] + before_end_path_str
    after_dataPath = after_dataPath_base + all_innerDataPath_list[i] + after_end_path_str
    # seq_len_list, before_time_cost_list, after_time_cost_list \
    #                       = parseResult(before_dataPath, after_dataPath, split_num_list[i])
    before_seq_len_list, before_time_cost_list = parseResult_single(before_dataPath, before_split_num_list[i])
    after_seq_len_list, after_time_cost_list = parseResult_single(after_dataPath, after_split_num_list[i])
    assert before_seq_len_list == after_seq_len_list

    draw(before_seq_len_list, before_time_cost_list, after_time_cost_list, all_innerDataPath_list[i] + after_end_path_str)
  """ 
  
  # for single
  

  dataPath_base = proj_path + "log_short/"

  logShort_file_path_list = ["time_before_1024_1_704_all_2_seq1_short.txt", 
                            "time_625_after_1024_1_704_all_2_seq1_short.txt", 
                            "time_after_false_2128_1024_1_704_all_2_seq1_short.txt", 
                            "time_after_false_4128_1024_1_704_all_2_seq1_short.txt",
                            "time_after_true_1024_1_704_all_2_seq1_short.txt"]

  labels = ["before", "after_625", "after_false_2128", "after_false_4128", "after_true"]
  end_dataPath = "_1024_1_704_all_2_seq1_short.txt"

  seq_len_lists = []
  time_cost_lists = []
  for i in range(len(logShort_file_path_list)):
    dataPath = dataPath_base + logShort_file_path_list[i]
    seq_len_tmp, time_cost_tmp = parseResult_single(dataPath, 6)
    seq_len_lists.append(seq_len_tmp)
    time_cost_lists.append(time_cost_tmp)
  
  for i in range(len(seq_len_lists)):
    assert seq_len_lists[i] == seq_len_lists[0]
  draw2(seq_len_lists[0], time_cost_lists, labels, end_dataPath)
  

  

  # before_split_num = 6
  # after_625_split_num = 6
  # after_false_2128_split_num = 6
  # after_false_4128_split_num = 6
  # after_true_split_num = 6

  # before_end_path_str = "_1024_1_625_seq1_short.txt"
  # after_end_path_str = "_1024_1_625_seq1_short.txt"
  # after_2_end_path_str = "_1024_1_624_seq1_short.txt"

  # before_dataPath = before_dataPath_base + before_end_path_str
  # after_dataPath = after_dataPath_base + after_end_path_str
  # after_2_dataPath = after_2_dataPath_base + after_2_end_path_str

  # before_seq_len_list, before_time_cost_list = parseResult_single(before_dataPath, before_split_num)
  # after_seq_len_list, after_time_cost_list = parseResult_single(after_dataPath, after_split_num)
  # after_2_seq_len_list, after_2_time_cost_list = parseResult_single(after_2_dataPath, after_2_split_num)

  # assert before_seq_len_list == after_seq_len_list == after_2_seq_len_list
