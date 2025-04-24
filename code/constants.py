#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants

local working dir: /Users/jingjingtang/Documents/backfill-recheck/paper
@author: jingjingtang
"""

from delphi_utils import GeoMapper
gmpr = GeoMapper()


data_dir = "./data/"
fig_dir = "./figs/"

filtered_states = ["gu", "as", "pr", "mp", "dc", "vi"]
signals = ["Insurance claims", "Antigen tests", "COVID-19 cases"]
taus = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

MA_POP = 1e6

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']


map_list = [
    '','','','','','wi', '','','vt', 'nh', 'me',
    'wa', 'id', 'mt', "nd", "mn", "il", "mi", '', 'ny', 'ma', '',
    'or', 'nv', 'wy', 'sd', 'ia', 'in', 'oh', 'pa', 'nj', 'ct', 'ri',
    'ca', 'ut', 'co', 'ne', 'mo', 'ky', 'wv', 'va', 'md', 'de', '',
    '', 'az', 'nm', 'ks', 'ar', 'tn', 'nc', 'sc', 'dc', '', '',
    '', '', '', 'ok', 'la', 'ms', 'al', 'ga', '', '', '',
    'hi', 'ak', '', 'tx', '', '', '', '', 'fl', '','']
