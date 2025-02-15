import csv, re, copy, os, time, itertools, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
from os.path import join
from lifelines import statistics
from scipy import stats

def get_km_curve(times, events, clip_time=60):
    # get km curve
    unique_times = np.asarray(list(set(times)))
    sorted_unique_times = np.sort(unique_times)
    S_list = [1.]
    time_list = [0.]
    censor_list = [False]
    at_risk_list = [len(times)]
    live_at_the_start = len(times)
    S_t = 1.
    start_time = 0
    RMST = 0.
    for i in range(len(sorted_unique_times)):
        end_time = sorted_unique_times[i]
        event_num = np.sum(events[times==end_time])
        at_risk_list.append(live_at_the_start)
        live_at_the_start = np.sum(times >= end_time)
        if end_time <= clip_time:
            RMST += (S_t * (end_time - start_time))
        S_list.append(S_t)
        S_t *= (1. - event_num/live_at_the_start)
        S_list.append(S_t)
        time_list.append(end_time)
        time_list.append(end_time)
        censor_list.append(0 in events[times==end_time])
        censor_list.append(0 in events[times==end_time])
        at_risk_list.append(live_at_the_start)
        start_time = end_time
    if np.amax(times) < clip_time:
        RMST += (S_t * (60 - end_time))
    return S_list, time_list, censor_list, at_risk_list, RMST

@jit
def get_rmst_custom(times, events, clip_time=60):
    # get km curve
    unique_times = np.asarray(list(set(times)))
    sorted_unique_times = np.sort(unique_times)
    S_list = [1.]
    time_list = [0.]
    S_t = 1.
    start_time = 0
    RMST = 0.
    for i in range(len(sorted_unique_times)):
        end_time = sorted_unique_times[i]
        event_num = np.sum(events[times==end_time])
        live_at_the_start = np.sum(times >= end_time)
        if end_time <= clip_time:
            RMST += (S_t * (end_time - start_time))
        S_t *= (1. - event_num/live_at_the_start)
        S_list.append(S_t)
        time_list.append(end_time)
        start_time = end_time
    if np.amax(times) < clip_time:
        RMST += (S_t * (60 - end_time))
    return RMST

def get_rmst(time, event, label):
    kmf = KaplanMeierFitter().fit(time, event, label=label)
    return restricted_mean_survival_time(kmf, t=60)

def get_delta_rmst(time, event, assignments, k, return_RMST=False, case2=False):
    rmst_list = []
    for i in range(k):
        if np.sum(assignments==i) <= 3:
            return_list = [-10.] * (k - 1) if not return_RMST else [-10.] * (k - 1 + k)
            return return_list
        rmst = get_rmst_custom(
            time[assignments==i],
            event[assignments==i], 
        )
        rmst_list.append(rmst)
    if k == 3:
        if case2:
            return_list = [rmst_list[0] - rmst_list[2], rmst_list[1] - rmst_list[2]]
        else:
            return_list = [rmst_list[0] - rmst_list[1], rmst_list[1] - rmst_list[2]]
    elif k == 2:
        return_list = [rmst_list[0] - rmst_list[1]]
    if return_RMST:
        return_list += rmst_list
    return return_list

def get_reward(os_month, os_status, assignments, combine12=False):
    rmst_list = []
    if combine12:
        assignments[assignments<2] = 0
        assignments[assignments==2] = 1
        for subtype in range(2):
            if np.sum(assignments==subtype) <= 2:
                return -2. * np.ones_like(assignments)
            os = os_month[assignments==subtype]
            status = os_status[assignments==subtype]
            rmst = get_rmst_custom(os, status)
            rmst_list.append(rmst)
        r = rmst_list[0] - rmst_list[1]
    else:
        for subtype in range(3):
            if np.sum(assignments==subtype) <= 3:
                return -2. * np.ones_like(assignments)
            os = os_month[assignments==subtype]
            status = os_status[assignments==subtype]
            rmst = get_rmst_custom(os, status)
            rmst_list.append(rmst)
        r = min(rmst_list[0] - rmst_list[1], rmst_list[1] - rmst_list[2])/10.
    return (r * np.ones_like(assignments)).astype(np.float32)

def multivariate_logrank_test(times, events, assignments):
    logrank_results = statistics.multivariate_logrank_test(times, assignments, events)
    p = logrank_results.p_value + 1e-100
    return p

def filter_patients(df, input_string):
    # Split the input string into components
    parts = input_string.split('_')
    
    # Initialize cohort, group, and risk variables
    cohorts = []
    ex_valid_condition = None
    risk_condition = None
    
    # Check if the parts contain cohort, group, and risk
    for part in parts:
        if part.lower() in ['jiang', 'sh', 'gz', 'fz', 'gao']:
            cohorts.append(part)

    if 'cv-valid' in input_string:
        ex_valid_condition = 0
    elif 'ex-valid' in input_string:
        ex_valid_condition = 1
    if 'low-risk' in input_string:
        risk_condition = 0
    
    # Filter based on cohort
    if cohorts:
        df_filtered = df[df['cohort'].str.lower().isin([c.lower() for c in cohorts])]
    else:
        df_filtered = df.copy()

    # Filter based on group if specified
    if ex_valid_condition is not None:
        df_filtered = df_filtered[df_filtered['ex_valid'] == ex_valid_condition]
    
    # Filter based on risk if specified
    if risk_condition is not None:
        df_filtered = df_filtered[df_filtered['risk'] == risk_condition]
    
    return df_filtered

def decode_prob(encoded_prob):
    prob_str = str(encoded_prob).zfill(6)
    class1_prob = int(prob_str[:2]) / 100
    class2_prob = int(prob_str[2:4]) / 100
    class3_prob = int(prob_str[4:6]) / 100
    return [class1_prob, class2_prob, class3_prob]

def normalize(probs):
    total = sum(probs)
    return [p / total for p in probs]

def process_results(row, mode='retrain'):
    fold = row["fold"]

    if fold in range(0, 5):
        return row.get(f"assignment_fold-{int(fold)}", None), row.get(f"prob_fold-{int(fold)}", None)
    elif fold == -1 and mode =='retrain':
        return int(row['assignment_fold-5']), int(row['prob_fold-5'])

    return None, None