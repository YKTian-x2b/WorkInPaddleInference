import re

dir_pth = "/tyk/PaddleNLP/llm/kai/mmha_res/mmhaRes_715/"

# logOrigin_file_path_list = ["time_before_3072_1_511.txt", "time_before_2048_2_511.txt",
#                           "time_before_2048_1_511.txt",  # "time_before_1024_8_511.txt", 
#                           "time_before_1024_4_511.txt", "time_before_1024_2_511.txt", 
#                           "time_before_1024_1_511.txt"] 
# logShort_file_path_list = ["time_before_3072_1_511_short.txt", "time_before_2048_2_511_short.txt", 
#                           "time_before_2048_1_511_short.txt", # "time_before_1024_8_511_short.txt", 
#                           "time_before_1024_4_511_short.txt", "time_before_1024_2_511_short.txt", 
#                           "time_before_1024_1_511_short.txt"] 
# logOrigin_file_path_list = ["time_after_3072_1_V5dirty_2128_512.txt", "time_after_2048_2_V5dirty_2128_512.txt", 
#                             "time_after_2048_1_V5dirty_2128_512.txt", # "time_after_1024_8_V5dirty_2128_512.txt", 
#                             "time_after_1024_4_V5dirty_2128_512.txt", "time_after_1024_2_V5dirty_2128_512.txt", 
#                             "time_after_1024_1_V5dirty_2128_512.txt"]
# logShort_file_path_list = ["time_after_3072_1_V5dirty_2128_512_short.txt", "time_after_2048_2_V5dirty_2128_512_short.txt", 
#                             "time_after_2048_1_V5dirty_2128_512_short.txt", # "time_after_1024_8_V5dirty_2128_512_short.txt", 
#                             "time_after_1024_4_V5dirty_2128_512_short.txt", "time_after_1024_2_V5dirty_2128_512_short.txt", 
#                             "time_after_1024_1_V5dirty_2128_512_short.txt"]
# logOrigin_file_path_list = [ "time_after_1024_4_V5_512.txt", "time_after_1024_2_V5_512.txt", 
#                             "time_after_1024_1_V5_512.txt"]
# logShort_file_path_list = ["time_after_1024_4_V5_512_short.txt", "time_after_1024_2_V5_512_short.txt", 
#                             "time_after_1024_1_V5_512_short.txt"]
logOrigin_file_path_list = ["time_before_1024_1_704_all_2.txt", 
                            "time_625_after_1024_1_704_all_2.txt", 
                            "time_after_false_2128_1024_1_704_all_2.txt", 
                            "time_after_false_4128_1024_1_704_all_2.txt",
                            "time_after_true_1024_1_704_all_2.txt"]
logOrigin_seq1_file_path_list = ["time_before_1024_1_704_all_2_seq1.txt", 
                            "time_625_after_1024_1_704_all_2_seq1.txt", 
                            "time_after_false_2128_1024_1_704_all_2_seq1.txt", 
                            "time_after_false_4128_1024_1_704_all_2_seq1.txt",
                            "time_after_true_1024_1_704_all_2_seq1.txt"]
logShort_file_path_list = ["time_before_1024_1_704_all_2_seq1_short.txt", 
                            "time_625_after_1024_1_704_all_2_seq1_short.txt", 
                            "time_after_false_2128_1024_1_704_all_2_seq1_short.txt", 
                            "time_after_false_4128_1024_1_704_all_2_seq1_short.txt",
                            "time_after_true_1024_1_704_all_2_seq1_short.txt"]

def cut2():
  source_file_path = dir_pth + 'time_1024_1_704_all_2.txt'
  target_file_path_base = dir_pth + "log_origin/"

  target_file_path_list = logOrigin_file_path_list
  log_count_list = [2, 2, 2, 2, 2]
  
  i = 0
  with open(source_file_path, 'r') as source_file:
      count = log_count_list[i]
      target_file = open(target_file_path_base + target_file_path_list[i], 'w')
      for line in source_file:  
          if 'total costs' in line:  
              count = count - 1
              if count == 0:
                target_file.write(line) 
                i += 1
                if i >= len(log_count_list):
                    break
                count = log_count_list[i]
                target_file.close()
                target_file = open(target_file_path_base + target_file_path_list[i], 'w')
                continue

          if count > 0:  
              target_file.write(line)  

      if not target_file.closed:
        target_file.close()

  print(f"Lines before and including 'total cost' have been written to target_file_path")


def getSeq1():
  source_file_path_base = dir_pth + "log_origin/"
  target_file_path_base = dir_pth + "log_origin/"

  for i in range(len(logOrigin_file_path_list)):
    source_file_path = source_file_path_base + logOrigin_file_path_list[i]
    target_file_path = target_file_path_base + logOrigin_seq1_file_path_list[i]
    target_file = open(target_file_path, 'w')
    with open(source_file_path, 'r') as source_file:
      for line in source_file:  
        if 'total costs' in line:  
          target_file.write(line) 
          break
        target_file.write(line) 

    if not target_file.closed:
      target_file.close()

def cut():
  source_file_path = dir_pth + 'time_after_704_all.txt'  # 'time_before_511.txt'  
  target_file_path_base = dir_pth + "log_origin/"

  target_file_path_list = logOrigin_file_path_list
  # log_count_list = [8, 4, 8, 1, 2, 4, 8]
  log_count_list = [4, 2, 4, 1, 2, 4]
  # log_count_list = [1, 2, 4]
  
  i = 0
  with open(source_file_path, 'r') as source_file:
      count = log_count_list[i]
      target_file = open(target_file_path_base + target_file_path_list[i], 'w')
      for line in source_file:  
          if 'total costs' in line:  
              count = count - 1
              if count == 0:
                target_file.write(line) 
                i += 1
                if i >= len(log_count_list):
                    break
                count = log_count_list[i]
                target_file.close()
                target_file = open(target_file_path_base + target_file_path_list[i], 'w')
                continue

          if count > 0:  
              target_file.write(line)  

      if not target_file.closed:
        target_file.close()

  print(f"Lines before and including 'total cost' have been written to target_file_path")


def cut_short_single(source_file_path,target_file_path ):
  with open(source_file_path, 'r') as source_file, \
      open(target_file_path, 'w') as target_file:
      for line in source_file:  
          r = re.search('seq_lens: \[(.*)\]', line)
          if r is not None:
            new_res = int(r.groups()[0])
            if new_res % 20 == 0:
              target_file.write(line) 


def cut_short():
  source_file_path_base = dir_pth + "log_origin/"
  target_file_path_base = dir_pth + "log_short/"

  # source_file_path = source_file_path_base + "time_before_3072_1.txt"  
  # target_file_path = target_file_path_base + "time_before_3072_1_short.txt" 

  # source_file_path = source_file_path_base + "time_after_1024_1_704_seq1.txt"  
  # target_file_path = target_file_path_base + "time_after_1024_1_704_seq1_short.txt" 
  # source_file_path = source_file_path_base + "time_before_1024_1_704_seq1.txt"  
  # target_file_path = target_file_path_base + "time_before_1024_1_704_seq1_short.txt" 
  # source_file_path = source_file_path_base + "time_after_1024_1_704_seq1.txt"  
  # target_file_path = target_file_path_base + "time_after_1024_1_704_seq1_short.txt" 
  # cut_short_single(source_file_path, target_file_path)
  
  
  # for list
  source_file_path_list = logOrigin_seq1_file_path_list
  target_file_path_list = logShort_file_path_list 
  
  assert len(source_file_path_list) == len(target_file_path_list)

  for i in range(len(source_file_path_list)):
    source_file_path = source_file_path_base + source_file_path_list[i]
    target_file_path = target_file_path_base + target_file_path_list[i]
    cut_short_single(source_file_path, target_file_path)
  

def endToEnd(res:str):
  avg_time = 0.
  cnt = 0
  for line in res.splitlines():
      r = re.search('total costs: (.*)ms', line)
      if r is not None:
          new_res = float(r.groups()[0])
          print(new_res)
          cnt += 1
          avg_time += new_res
  print(avg_time/cnt)
  
def getEndToEndAvg():
  source_file_path_base = dir_pth + "log_origin/"
  # source_file_path_list = ["time_after_3072_1_V5_512.txt"]
  source_file_path_list = logOrigin_file_path_list
  
  for source_file_pth in source_file_path_list:
    with open(source_file_path_base + source_file_pth, "r") as source_file:
      content = source_file.read()
      endToEnd(content)
      print("============")


def getSpeedup(before_opt, after_opt):
  return (before_opt - after_opt) / after_opt * 100


def getSpeedupList():
  # before_opt_list = [14411.513400000002, 14539.564000000002, 14804.6976, 15661.997200000002, 30181.4698, 30856.331599999998, 45911.787399999994]
  # after_opt_list = [14291.422400000001, 14449.8068, 14671.2302, 15596.517799999998, 28447.657400000004, 30225.583599999994, 43407.6858]
  # [14351.0198, 14166.8236, 14867.02, 15756.929200000002, 29421.264399999996, 29396.880599999997, 43189.526399999995]
  # before_opt_list = [30208, 31173, 36432, 45948, 58376, 68579, 96743]
  # after_opt_list = [31936, 33446, 37103, 45222, 64813, 74910, 100512]


  before_opt_list = [14278.211, 14439.488799999997, 14601.794999999998, 15585.3856, 28425.4312, 30008.855, 43832.052599999995,]
  after_opt_list = [13835.151800000001, 14284.6236, 14690.3984, 15708.560800000001, 28059.643399999994, 29022.238400000002, 42909.31]

  assert len(before_opt_list) == len(before_opt_list)
  for i in range(len(before_opt_list)):
    print(getSpeedup(before_opt_list[i], after_opt_list[i]))


if __name__ == "__main__":
  # cut2()
  # getSeq1()
  # cut_short()

  # getEndToEndAvg()

  getSpeedupList()


  # res1 = [9151,8923,9311,9603,]
  # res2 = [8856,9296,9034,10237,]
  # baseline1 = 9366
  # baseline2 = 8705
  # for i in res1:
  #   print(getSpeedup(baseline1, i))
  # for i in res2:
  #   print(getSpeedup(baseline2, i))

  # list_kai = [38230.321,33125.936,32902.488,35183.097,37197.372]
  # sum = 0.
  # for i in list_kai:
  #   sum += i
  # print(sum/5.)
  # print(getSpeedup(59858,58376))


  source_file_path = dir_pth + "time_bestR_1024_4_716_none_1237.txt"
  # source_file_path = dir_pth + "time_715_none.txt"
  cnt = 6
  res_sum = 0.
  res_list = []
  all_cnt = 0
  with open(source_file_path, 'r') as source_file:
    for line in source_file:  
      r = re.search('total costs: (.*)ms', line)
      if r is not None:
        cnt -= 1
        all_cnt += 1
        # print(float(r.groups()[0]))
        if(cnt < 5):
          res_sum += float(r.groups()[0])

        if(cnt == 0):
          res_list.append(res_sum / 5.)
          res_sum = 0.
          cnt = 6
          # print("\n")
  print(res_list)
  print(all_cnt)  



