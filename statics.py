#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
@author: Yizhu Chen
"""

import numpy as np


data_molhiv = {
    "validation": {
        "diffpool": [0.806520062, 0.797934426, 0.756194273, 0.791158387, 0.807940795, 0.788574123, 0.761935381, 0.812677592, 0.797211812, 0.776632618],
        "gin": [0.793904933, 0.807561116, 0.789495762, 0.792609739, 0.792122893, 0.804698217, 0.812968474, 0.802098643, 0.785600995, 0.803807197],
        "smoothpool": [0.81501384, 0.81550681, 0.805635166, 0.823201426, 0.815292475, 0.817144939, 0.825543798, 0.811174187, 0.836686141, 0.806400647],
        "topk": [0.81919796, 0.769971402, 0.752745015, 0.802626825, 0.798741243, 0.823982216, 0.817195461, 0.784382349, 0.810332158, 0.770814962],
        "lapool": [0.780325299, 0.799431707, 0.819652655, 0.81386868, 0.78505291, 0.809955541, 0.793228248, 0.806627229, 0.778978052, 0.809061459],
    },
    "test": {
        "diffpool": [0.763031345, 0.713323934, 0.694974797, 0.748807432, 0.781795709, 0.732899438, 0.703650128, 0.763977674, 0.763127909, 0.761314432],
        "gin": [0.723129068, 0.728057707, 0.751320033, 0.754242067, 0.742482474, 0.722557407, 0.763143359, 0.757308948, 0.737797176, 0.744809672],
        "smoothpool": [0.750329284, 0.758465787, 0.757077193, 0.72628865, 0.773218873, 0.746874215, 0.764149559, 0.749313428, 0.764178528, 0.743538886],
        "topk": [0.743813129, 0.706885031, 0.722144112, 0.761808841, 0.720483207, 0.725294038, 0.749153132, 0.724323567, 0.727896445, 0.695610189],
        "lapool": [0.749747967, 0.760159717, 0.749877557, 0.76739431, 0.736209853, 0.726267599, 0.735668901, 0.767023504, 0.720807663, 0.769785048],
    }
}

data_moltoxcast= {
    "validation": {
        "diffpool": [0.710316599, 0.701562054, 0.704068001, 0.690366309, 0.708232546],
        "gin": [0.680733196, 0.679300241, 0.679587943, 0.67911581, 0.669750316, 0.671670834, 0.675307695, 0.654267059, 0.677519282, 0.677359202],
        "smoothpool": [0.723872617, 0.715140596, 0.71304247, 0.728080868, 0.71792878, 0.722578905, 0.714706467, 0.710457021, 0.719817467, 0.709611399],
        "topk": [0.634500953, 0.659023933, 0.627376409, 0.634577564, 0.63088951],
        "lapool": [0],
    },
    "test": {
        "diffpool": [0.649604792, 0.655352405, 0.655104044, 0.639145373, 0.65725957],
        "gin": [0.629233894, 0.629108616, 0.634248931, 0.62870361, 0.622925588, 0.626826568, 0.627802154, 0.600761729, 0.610827116, 0.634383539],
        "smoothpool": [0.670693488, 0.666246, 0.660342309, 0.664222274, 0.653469852, 0.662745974, 0.64768796, 0.652086112, 0.655239383, 0.660383241],
        "topk": [0.602703009, 0.613602242, 0.592194215, 0.607853936, 0.580713215],
        "lapool": [0],
    }
}

pool_methods = ["gin", "smoothpool", "topk", "diffpool", "lapool"]

for method in pool_methods:
    validation = np.array(data_molhiv["validation"][method])
    test = np.array(data_molhiv["test"][method])
    print(f"{method}:\nvalidation: mean:{validation.mean()} std:{validation.std()}\ntest: mean:{test.mean()} std:{test.std()}")


print("--------------------")

for method in pool_methods:
    validation = np.array(data_moltoxcast["validation"][method])
    test = np.array(data_moltoxcast["test"][method])
    print(f"{method}:\nvalidation: mean:{validation.mean()} std:{validation.std()}\ntest: mean:{test.mean()} std:{test.std()}")
