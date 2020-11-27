#!/bin/python
# shuaijiang@202010

import math
from optparse import OptionParser


def compute_p(in_file):
    '''
    compute p for age prediction
    :param in_file: echo line contains true and predict ages
    :return: the p
    '''
    age_true_list = []
    age_pred_list = []
    tmp_mean_true = 0
    tmp_var_true  = 0
    tmp_mean_pred = 0
    tmp_var_pred  = 0
    total_num = 0
    with open(in_file) as IN:
        line_list = IN.readlines()
        for line in line_list:
            age_true = float(line.strip().split()[1])
            age_pred = float(line.strip().split()[0])
            age_true_list.append(age_true)
            age_pred_list.append(age_pred)
            total_num += 1 
            tmp_mean_true += age_true
            tmp_var_true += age_true * age_true
            tmp_mean_pred += age_pred
            tmp_var_pred += age_pred * age_pred

    age_true_mean = tmp_mean_true / total_num
    age_true_var  = tmp_var_true / total_num - age_true_mean * age_true_mean
    age_pred_mean = tmp_mean_pred / total_num
    age_pred_var  = tmp_var_pred / total_num - age_pred_mean * age_pred_mean
    print(age_true_mean, age_true_var)
    print(age_pred_mean, age_pred_var)
    age_true_stdvar = math.sqrt(age_true_var)
    age_pred_stdvar = math.sqrt(age_pred_var)
    age_list = zip(age_true_list, age_pred_list)
    p = 0
    for i, (age_true, age_pred) in enumerate(age_list):
        p += ((age_true - age_true_mean) / age_true_stdvar) * ((age_pred - age_pred_mean) / age_pred_stdvar)
    p = p / (total_num - 1)
    print(p)

def main(argv=None):
    usage = "python %prog age.txt"
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    if len(args) != 1:
        print(usage)
        return
    in_file = args[0]
    compute_p(in_file)


if __name__ == '__main__':
    main()
